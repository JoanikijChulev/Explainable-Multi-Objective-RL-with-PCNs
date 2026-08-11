"""CFZOO_Rollout — interactive UI server.

Stdlib HTTP server; dependencies come from the repository's Conda environment.
"""
import base64
import io
import json
import os
import shutil
import sys
import threading
import traceback
# Single-threaded server on purpose: MuJoCo's offscreen GL context is bound to
# the thread that created it — rendering from ThreadingHTTPServer worker
# threads produces black frames (mo-reacher). All work is serialized anyway.
from http.server import BaseHTTPRequestHandler, HTTPServer

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(APP_DIR, "MODELS")
sys.path.insert(0, APP_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(1, REPO_DIR)

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import interactive_pcn_zoo_cf as cf
from device_utils import preferred_device
from pcn.env_setup import build_experiment_setup
from pcn.pcn import action_mask_for_env, apply_action_mask
from env_info import ENV_INFO, OBJECTIVE_SHORT, VARIANT_INFO

DEVICE = preferred_device()
SEED = 0
# Per-env reset-seed overrides. breakable-bottles: seed 0's random stream drops
# a bottle during the front plan's carry-2 run; seed 1 is drop-free for both
# the R and RH models (verified: exact (-10, 50, 0), terminated).
ENV_SEEDS = {"breakable-bottles-v0": 1}


def env_seed(env_name=None):
    return ENV_SEEDS.get(env_name if env_name is not None else S.get("env_name"), SEED)
# App-level defaults (the vendored paper code keeps its own untouched):
# margin (kappa) 0 = the foil strictly wins the argmax; 30k query budget per
# stage; CW capped at 1k iterations per binary-search round.
APP_ZOO_DEFAULTS = dict(cf.DEFAULT_ZOO_SETTINGS, margin=0.0, max_queries=30000, random_directions=30000)
APP_CW_OVERRIDES = {"max_iter": 1000}
ROLLOUT_CAP = 1200
LANDING_TOL = {"minecart": 5e-3}
DEFAULT_TOL = 1e-3
VERIFIED_FRONT_TOL = 1e-2
VERIFIED_FRONT_BENCHMARKS = {
    ("mo-reacher-v5", "R"): {
        "state_coverage": "197/200 (98.5%)",
        "zoo_realization": "50/50 (100%)",
        "zoo_max_queries": 500,
    },
    ("mo-reacher-v5", "RH"): {
        "state_coverage": "200/200 (100%)",
        "zoo_realization": "50/50 (100%)",
        "zoo_max_queries": 500,
    },
}

LOCK = threading.RLock()
S = {}    # loaded model session
RT = {}   # rollout state
CF = {}   # last counterfactual result
CONT = {} # last verified-realization result (for the saved report)


class HorizonFrozenAdapter(nn.Module):
    """Expose an RH model as model(obs, desired_return) with the horizon frozen,
    so the CF ZOO / C&W code (which searches return-command space only) runs on
    it unchanged."""

    def __init__(self, inner, horizon):
        super().__init__()
        self.inner = inner
        self.horizon = float(horizon)

    def forward(self, state, desired_return):
        h = torch.full(
            (desired_return.shape[0], 1),
            self.horizon,
            dtype=torch.float32,
            device=desired_return.device,
        )
        return self.inner(state, desired_return, h)


def jsonable(x):
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _load_numeric_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(v) for v in line.split(",")])
    if not rows:
        raise ValueError(f"front file has no data rows: {path}")
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError(f"front file contains malformed or non-finite rows: {path}")
    return arr


def load_front_file(path, variant):
    arr = _load_numeric_rows(path)
    if variant == "RH":
        if arr.shape[1] < 2:
            raise ValueError(f"RH front needs return columns plus a horizon: {path}")
        return arr[:, :-1], arr[:, -1]
    return arr, None


def load_verified_front_file(path, variant, n_obj=None):
    """Load desired witness commands separately from achieved Pareto targets."""
    arr = _load_numeric_rows(path)
    horizon_columns = 1 if variant == "RH" else 0
    paired_columns = arr.shape[1] - horizon_columns
    if n_obj is None:
        if paired_columns < 2 or paired_columns % 2:
            raise ValueError(
                f"cannot infer objective count from verified {variant} front with "
                f"{arr.shape[1]} columns: {path}"
            )
        n_obj = paired_columns // 2
    expected = 2 * int(n_obj) + (1 if variant == "RH" else 0)
    if arr.shape[1] != expected:
        raise ValueError(
            f"verified {variant} front for {n_obj} objectives needs {expected} columns; "
            f"got {arr.shape[1]} in {path}"
        )
    commands = arr[:, :n_obj].astype(np.float32)
    if variant == "RH":
        horizons = arr[:, n_obj].astype(np.float32)
        targets = arr[:, n_obj + 1:].astype(np.float32)
        if not np.all(horizons == np.rint(horizons)) or np.any(horizons < 1):
            raise ValueError(f"verified RH front has a non-positive or non-integral horizon: {path}")
    else:
        horizons = None
        targets = arr[:, n_obj:].astype(np.float32)
    return commands, targets, horizons


def verified_front_path(env_name, variant):
    """Return the sole built-in front stored beside its model checkpoint."""
    return os.path.join(
        MODELS_DIR,
        env_name,
        f"{env_name}_{variant}_achievable_within_1pct.txt",
    )


INCLUDED_WALKROOMS = ("walkroom2", "walkroom3")
CUSTOM_DIR = os.path.join(MODELS_DIR, "custom")


def list_custom_models():
    out = []
    if not os.path.isdir(CUSTOM_DIR):
        return out
    for d in sorted(os.listdir(CUSTOM_DIR)):
        meta_path = os.path.join(CUSTOM_DIR, d, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            continue
        info = ENV_INFO.get(meta.get("env", ""), {})
        out.append({
            "id": f"custom::{d}",
            "env": meta.get("env", "?"),
            "variant": meta.get("variant", "R"),
            "n_front": int(meta.get("n_front", 0)),
            "n_obj": int(meta.get("n_obj", 0)),
            "title": f"{meta.get('name', d)} (custom)",
            "tagline": f"your model on {info.get('title', meta.get('env', '?'))} — {meta.get('front_source', '')}",
            "custom": True,
        })
    return out


def list_models():
    out = []
    for d in sorted(os.listdir(MODELS_DIR)):
        p = os.path.join(MODELS_DIR, d)
        if not os.path.isdir(p) or d == "custom":
            continue
        if d.startswith("walkroom") and d not in INCLUDED_WALKROOMS:
            continue
        for variant in ("R", "RH"):
            pt = os.path.join(p, f"{d}_{variant}.pt")
            ft = verified_front_path(d, variant)
            if not (os.path.isfile(pt) and os.path.isfile(ft)):
                continue
            _commands, front, _horizons = load_verified_front_file(ft, variant)
            info = ENV_INFO.get(d, {})
            out.append({
                "id": f"{d}::{variant}",
                "env": d,
                "variant": variant,
                "n_front": int(len(front)),
                "n_obj": int(front.shape[1]) if front.ndim == 2 else 0,
                "title": info.get("title", d),
                "tagline": info.get("tagline", ""),
            })
    return out


def build_env_setup(env_name):
    """Build the env, preferring pixel rendering; returns (setup, render_mode)."""
    if env_name.startswith("walkroom"):
        # WalkRoom has no pixel renderer; the app draws its own frames.
        return build_experiment_setup(env_name, device=DEVICE, include_model=False, render_mode=None), None
    try:
        return build_experiment_setup(env_name, device=DEVICE, include_model=False, render_mode="rgb_array"), "rgb_array"
    except Exception:
        return build_experiment_setup(env_name, device=DEVICE, include_model=False, render_mode=None), None


def load_model_file(path):
    """torch.load with a TorchScript fallback and readable error messages."""
    try:
        model = torch.load(path, map_location=DEVICE, weights_only=False)
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"the checkpoint references Python module '{exc.name}' which is not available here. "
            "Full pickled models must be saved against the pcn package classes (or exported as TorchScript)."
        )
    except Exception as first_error:
        try:
            model = torch.jit.load(path, map_location=str(DEVICE))
        except Exception:
            raise ValueError(f"file is not a loadable torch checkpoint: {first_error}")
    if isinstance(model, dict):
        raise ValueError(
            "this file is a state_dict (bare weights), not a full model. "
            "Save with torch.save(model, path) so the architecture is included, or export TorchScript."
        )
    if not callable(model):
        raise ValueError(f"loaded object of type {type(model).__name__} is not a callable model.")
    model = model.to(DEVICE)
    if hasattr(model, "scaling_factor") and torch.is_tensor(model.scaling_factor):
        model.scaling_factor = model.scaling_factor.to(DEVICE)
    try:
        model.eval()
    except Exception:
        pass
    return model


def load_model(env_name, variant, custom=None):
    close_session()
    if custom is not None:
        meta_path = os.path.join(CUSTOM_DIR, custom, "meta.json")
        if not os.path.isfile(meta_path):
            raise ValueError(f"unknown custom model '{custom}'")
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        env_name, variant = meta["env"], meta["variant"]
        pt = os.path.join(CUSTOM_DIR, custom, "model.pt")
        front_path = os.path.join(CUSTOM_DIR, custom, "front.txt")
        front_source = meta.get("front_source") or "Uploaded custom front"
    else:
        pt = os.path.join(MODELS_DIR, env_name, f"{env_name}_{variant}.pt")
        front_path = verified_front_path(env_name, variant)
        front_source = "Achievable Pareto front"

    setup, render_mode = build_env_setup(env_name)
    model = load_model_file(pt)
    bounds = cf.command_bounds(setup)
    scale = cf.command_scale(bounds)
    n_obj = int(np.asarray(setup.max_return).shape[0])
    if custom is None:
        front_commands, front, horizons = load_verified_front_file(front_path, variant, n_obj)
        landing_front = front.copy()
        fidelity = np.linalg.norm((front_commands - front) / scale, axis=1)
        if np.any(fidelity > VERIFIED_FRONT_TOL + 1e-7):
            raise ValueError(
                f"verified front contains {int(np.sum(fidelity > VERIFIED_FRONT_TOL + 1e-7))} "
                f"commands outside the 1% fidelity threshold"
            )
        front_verified = True
        landing_tol = VERIFIED_FRONT_TOL
    else:
        front, horizons = load_front_file(front_path, variant)
        if front.shape[1] != n_obj:
            raise ValueError(f"front has {front.shape[1]} objectives; model environment has {n_obj}")
        front_commands = front.copy()
        landing_front = front.copy()
        fidelity = np.zeros(len(front), dtype=np.float32)
        front_verified = False
        landing_tol = LANDING_TOL.get(env_name, DEFAULT_TOL)
    n_actions = int(setup.n_actions)
    labels = [cf.action_label(setup.env, a) for a in range(n_actions)]
    custom_names = ENV_INFO.get(env_name, {}).get("action_names")
    if custom_names and len(custom_names) == n_actions:
        labels = [str(x) for x in custom_names]

    S.clear(); RT.clear(); CF.clear()
    S.update(dict(
        env_name=env_name, variant=variant, setup=setup, env=setup.env, model=model,
        front=front, front_commands=front_commands, front_horizons=horizons,
        landing_front=landing_front, front_fidelity=fidelity,
        front_source=front_source,
        front_verified=front_verified, landing_tol=landing_tol,
        bounds=bounds, scale=scale,
        n_actions=n_actions, action_labels=labels, render_mode=render_mode,
        max_return=np.asarray(setup.max_return, dtype=np.float32),
        original_outcomes={}, custom=custom,
    ))
    return env_name, variant


def close_session():
    env = S.get("env")
    if env is not None:
        try:
            env.close()
        except Exception:
            pass
    S.clear(); RT.clear(); CF.clear(); CONT.clear()


def policy_at(obs, rem_return, rem_horizon):
    if S["variant"] == "R":
        return cf.greedy_action(S["model"], S["env"], obs, np.asarray(rem_return, np.float32), DEVICE)
    obs_b = np.asarray([obs])
    ret_b = np.asarray([rem_return], dtype=np.float32)
    hor_b = np.asarray([[float(rem_horizon)]], dtype=np.float32)
    with torch.no_grad():
        log_probs = S["model"](
            torch.as_tensor(obs_b).to(DEVICE),
            torch.as_tensor(ret_b, dtype=torch.float32).to(DEVICE),
            torch.as_tensor(hor_b, dtype=torch.float32).to(DEVICE),
        )
    log_probs = log_probs.detach().cpu().numpy()[0].astype(np.float32)
    mask = action_mask_for_env(S["env"])
    masked = apply_action_mask(log_probs, mask)
    return {
        "action": int(np.argmax(masked)),
        "masked_log_probs": masked,
        "probs": cf.probabilities_from_log_probs(masked),
        "action_mask": mask,
    }


def cf_model(rem_horizon):
    if S["variant"] == "R":
        return S["model"]
    return HorizonFrozenAdapter(S["model"], rem_horizon)


def update_command(rem_return, reward):
    return np.clip(rem_return - reward, None, S["max_return"]).astype(np.float32)


TRAIL = []  # visited base-env positions of the current episode (walkroom visuals)

try:
    from PIL import ImageDraw, ImageFont
    _WFONT = ImageFont.truetype("arial.ttf", 15)
    _WFONT_B = ImageFont.truetype("arialbd.ttf", 17)
except Exception:
    from PIL import ImageDraw, ImageFont
    _WFONT = _WFONT_B = ImageFont.load_default()


def _png_b64(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def walkroom2_frame(env=None, trail=None):
    base = (env if env is not None else S["env"]).unwrapped
    if trail is None:
        trail = TRAIL
    size, room, pos = base.size, base.room, np.asarray(base.pos)
    cell, m = 25, 42
    W = m + size * cell + 14
    H = m + size * cell + 30
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for x0 in range(size):
        limit = int(room[x0])
        for x1 in range(size):
            px, py = m + x0 * cell, m + x1 * cell
            if x1 > limit:
                fill = (247, 219, 219)
            elif x1 == limit:
                fill = (214, 92, 92)
            else:
                fill = (255, 255, 255)
            d.rectangle([px, py, px + cell, py + cell], fill=fill, outline=(205, 210, 220))
    for tx in trail[:-1]:
        px, py = m + tx[0] * cell, m + tx[1] * cell
        d.rectangle([px + 7, py + 7, px + cell - 7, py + cell - 7], fill=(170, 205, 250))
    px, py = m + int(pos[0]) * cell, m + int(pos[1]) * cell
    d.ellipse([px + 4, py + 4, px + cell - 4, py + cell - 4], fill=(35, 90, 220))
    d.text((m + 4, m - 22), "dim 0  (each step right costs obj-0 one unit) ->", fill="black", font=_WFONT)
    d.text((4, m + 2), "d\ni\nm\n\n1", fill="black", font=_WFONT)
    d.text((m, H - 24), "red = border cells: entering a column at/below its red cell ends the episode",
           fill=(160, 60, 60), font=_WFONT)
    d.text((4, 4), f"WalkRoom 2D  pos=({int(pos[0])},{int(pos[1])})  border here: {int(room[int(pos[0])])}",
           fill="black", font=_WFONT_B)
    return img


def walkroom3_frame(env=None, trail=None):
    """Isometric 3-D view of the WalkRoom cube. The border is drawn as solid
    terrain (columns rising from the floor up to their limit depth); the agent
    is a floating cube that must sink onto the terrain of its column."""
    base = (env if env is not None else S["env"]).unwrapped
    if trail is None:
        trail = TRAIL
    size, room, pos = base.size, base.room, np.asarray(base.pos)
    ax0, ax1, az = int(pos[0]), int(pos[1]), int(pos[2])
    limit_here = int(room[ax0, ax1])

    TW, TH, ZH = 14, 7, 8
    W, H = 660, 566
    OX, OY = W // 2, 66
    img = Image.new("RGB", (W, H), (16, 19, 26))
    d = ImageDraw.Draw(img)

    def proj(x0, x1, z):
        return (OX + (x0 - x1) * TW, OY + (x0 + x1) * TH + z * ZH)

    def diamond(x0, x1, z, k=1.0):
        nx, ny = proj(x0 + (1 - k) / 2, x1 + (1 - k) / 2, z)
        return [(nx, ny), (nx + TW * k, ny + TH * k), (nx, ny + 2 * TH * k), (nx - TW * k, ny + TH * k)]

    def shade(color, f):
        return tuple(int(c * f) for c in color)

    def lerp(c1, c2, t):
        return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

    SHALLOW, DEEP = (246, 196, 110), (66, 108, 196)

    # faint surface rim (depth 0 plane) for reference
    rim = [proj(0, 0, 0), proj(size, 0, 0), proj(size, size, 0), proj(0, size, 0)]
    d.polygon(rim, outline=(90, 100, 120))

    trail_by_col = {}
    for t in trail[:-1]:
        trail_by_col.setdefault((t[0], t[1]), []).append(t[2])

    # Screen coords are the 180°-rotated physical coords so the terrain rises
    # AWAY from the viewer (start corner in front, deep basin visible).
    flip = lambda v: size - 1 - v
    pxa0, pxa1 = flip(ax0), flip(ax1)
    agent_label = None

    floor_z = size
    for s in range(2 * size - 1):
        for p0 in range(max(0, s - size + 1), min(size, s + 1)):
            p1 = s - p0
            x0, x1 = flip(p0), flip(p1)
            limit = int(room[x0, x1])
            top = diamond(p0, p1, limit)
            n, e, so, w = top
            drop = (floor_z - limit) * ZH
            col = lerp(SHALLOW, DEEP, limit / (size - 1))
            d.polygon([w, so, (so[0], so[1] + drop), (w[0], w[1] + drop)], fill=shade(col, 0.52))
            d.polygon([e, so, (so[0], so[1] + drop), (e[0], e[1] + drop)], fill=shade(col, 0.74))
            d.polygon(top, fill=col, outline=shade(col, 0.82))

            for tz in trail_by_col.get((x0, x1), []):
                cx, cy = proj(p0, p1, tz)
                cy += TH
                d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 190, 90))

            if x0 == ax0 and x1 == ax1:
                # highlight the landing cell on the terrain
                d.polygon(top, outline=(255, 150, 40))
                d.line([top[0], top[2]], fill=(255, 150, 40))
                # dashed plumb line from the agent down to its landing cell
                cx, cy_a = proj(p0, p1, az)
                cy_a += TH
                cy_t = proj(p0, p1, limit)[1] + TH
                yy = cy_a + 6
                while yy < cy_t - 2:
                    d.line([(cx, yy), (cx, min(yy + 4, cy_t - 2))], fill=(255, 170, 60), width=2)
                    yy += 8
                # agent cube (drawn after its column so nearer terrain can occlude it)
                ch = int(1.1 * ZH)
                bot = diamond(p0, p1, az, k=0.72)
                topc = [(p[0], p[1] - ch) for p in bot]
                bn, be, bs, bw = bot
                tn, te, ts, tw_ = topc
                d.polygon([bw, bs, ts, tw_], fill=(214, 108, 16))
                d.polygon([be, bs, ts, te], fill=(238, 132, 26))
                d.polygon(topc, fill=(255, 166, 52), outline=(255, 220, 170))
                agent_label = (min(cx + 26, W - 150), cy_a - ch - 20, limit)

    # agent labels drawn last, on a dark backing so they stay readable on sand
    if agent_label is not None:
        lx, ly, lim = agent_label
        d.rectangle([lx - 5, ly - 4, lx + 118, ly + 37], fill=(16, 19, 26), outline=(255, 150, 40))
        d.text((lx, ly), f"you: depth {az}", fill=(255, 210, 140), font=_WFONT)
        d.text((lx, ly + 18), f"land at {lim}", fill=(255, 160, 60), font=_WFONT)

    # axis arrows off the front (bottom) rim corner = the physical origin/start
    bx, by = proj(size, size, 0)
    e0 = proj(size - 3.2, size, 0)
    e1 = proj(size, size - 3.2, 0)
    d.line([(bx, by), e0], fill=(200, 210, 230), width=2)
    d.line([(bx, by), e1], fill=(200, 210, 230), width=2)
    d.line([(bx, by), (bx, by + int(2.6 * ZH))], fill=(200, 210, 230), width=2)
    d.text((e0[0] - 48, e0[1] - 8), "dim 0", fill=(210, 220, 240), font=_WFONT)
    d.text((e1[0] + 6, e1[1] - 8), "dim 1", fill=(210, 220, 240), font=_WFONT)
    d.text((bx + 8, by + int(2.6 * ZH) - 8), "dim 2 = depth", fill=(210, 220, 240), font=_WFONT)
    d.text((bx - 24, by + 4), "start", fill=(160, 170, 190), font=_WFONT)

    d.text((10, 8), f"WalkRoom 3D   pos=({ax0}, {ax1}, {az})   border of this column: depth {limit_here}",
           fill=(235, 240, 250), font=_WFONT_B)
    d.text((10, 30), "terrain = the termination border: the episode ends when the orange cube sinks onto its column",
           fill=(150, 160, 180), font=_WFONT)
    d.text((10, H - 44), "sand-colored columns are tall (border near the surface — episode ends after few dim-2 steps);",
           fill=(150, 160, 180), font=_WFONT)
    d.text((10, H - 26), "blue columns are deep. Dots = trail. Every step costs −1 on the dimension you moved along.",
           fill=(150, 160, 180), font=_WFONT)
    return img


def frame_b64():
    try:
        if S["env_name"].startswith("walkroom"):
            img = walkroom2_frame() if S["env_name"] == "walkroom2" else walkroom3_frame()
            return _png_b64(img)
        fr = S["env"].render()
        if fr is None:
            return None
        img = Image.fromarray(np.asarray(fr).astype(np.uint8))
        if max(img.size) > 900:
            img.thumbnail((900, 900))
        return _png_b64(img)
    except Exception:
        return None


def env_reset():
    out = S["env"].reset(seed=env_seed())
    del TRAIL[:]
    if S["env_name"].startswith("walkroom"):
        TRAIL.append(tuple(np.asarray(S["env"].unwrapped.pos).tolist()))
    return out[0] if isinstance(out, tuple) else out


def env_step(action):
    result = cf.step_env(S["env"], int(action))
    if S["env_name"].startswith("walkroom"):
        TRAIL.append(tuple(np.asarray(S["env"].unwrapped.pos).tolist()))
    return result


def replay(actions):
    """Deterministic save-state restore: fixed-seed reset + replay the prefix."""
    obs = env_reset()
    rem = np.asarray(RT["desired"], np.float32).copy()
    hor = RT["desired_horizon"]
    collected = np.zeros_like(rem)
    term = trunc = False
    for a in actions:
        obs, r, term, trunc, _ = env_step(a)
        collected = collected + r
        rem = update_command(rem, r)
        if hor is not None:
            hor = max(float(hor) - 1.0, 1.0)
        if term or trunc:
            break
    return obs, rem, hor, collected, term, trunc


def start_rollout(desired, horizon, front_index=None):
    RT.clear(); CF.clear(); CONT.clear()
    front_target = None
    if front_index is not None:
        front_target = np.asarray(S["front"][int(front_index)], np.float32).copy()
    RT.update(dict(
        desired=np.asarray(desired, np.float32),
        desired_horizon=None if horizon is None else float(horizon),
        selected_front_index=None if front_index is None else int(front_index),
        front_target=front_target,
        actions=[], history=[],
    ))
    obs = env_reset()
    RT.update(dict(
        obs=obs,
        rem=RT["desired"].copy(),
        hor=RT["desired_horizon"],
        collected=np.zeros_like(RT["desired"]),
        terminated=False, truncated=False,
    ))


def state_payload():
    if not RT:
        return {"active": False}
    done = RT["terminated"] or RT["truncated"]
    pol = None
    if not done:
        p = policy_at(RT["obs"], RT["rem"], RT["hor"])
        mask = p["action_mask"]
        valid = list(range(S["n_actions"])) if mask is None else [int(a) for a in np.flatnonzero(mask)]
        pol = {
            "action": p["action"],
            "label": S["action_labels"][p["action"]],
            "probs": [
                {"action": a, "label": S["action_labels"][a], "p": float(p["probs"][a]),
                 "valid": a in valid}
                for a in range(S["n_actions"])
            ],
            "valid_actions": valid,
        }
    return jsonable({
        "active": True,
        "t": len(RT["actions"]),
        "frame": frame_b64(),
        "obs": cf.to_jsonable(np.asarray(RT["obs"])),
        "desired": RT["desired"],
        "front_target": RT.get("front_target"),
        "selected_front_index": RT.get("selected_front_index"),
        "desired_horizon": RT["desired_horizon"],
        "remaining": RT["rem"],
        "remaining_horizon": RT["hor"],
        "collected": RT["collected"],
        "terminated": bool(RT["terminated"]),
        "truncated": bool(RT["truncated"]),
        "policy": pol,
        "history": RT["history"][-25:],
        "variant": S["variant"],
        "env": S["env_name"],
    })


def do_step():
    if RT["terminated"] or RT["truncated"]:
        return
    p = policy_at(RT["obs"], RT["rem"], RT["hor"])
    a = int(p["action"])
    obs, r, term, trunc, _ = env_step(a)
    RT["history"].append({
        "t": len(RT["actions"]), "action": a, "label": S["action_labels"][a],
        "reward": jsonable(r), "p": float(p["probs"][a]),
    })
    RT["actions"].append(a)
    RT["obs"] = obs
    RT["collected"] = RT["collected"] + r
    RT["rem"] = update_command(RT["rem"], r)
    if RT["hor"] is not None:
        RT["hor"] = max(float(RT["hor"]) - 1.0, 1.0)
    RT["terminated"], RT["truncated"] = bool(term), bool(trunc)


def do_back():
    if not RT["actions"]:
        return
    RT["actions"].pop()
    RT["history"].pop()
    obs, rem, hor, collected, term, trunc = replay(RT["actions"])
    RT.update(obs=obs, rem=rem, hor=hor, collected=collected, terminated=term, truncated=trunc)


def query_result_payload(res):
    if res is None:
        return None
    return jsonable({
        "success": bool(res.success),
        "command": res.command,
        "delta": res.delta,
        "l2_norm": float(res.l2_norm),
        "scaled_distance": cf.scaled_command_distance(res.command, RT["rem"], S["scale"]),
        "objective": float(res.objective),
        "hinge": float(res.hinge),
        "target_margin": float(res.target_margin),
        "greedy_action": int(res.greedy_action),
        "greedy_label": S["action_labels"][int(res.greedy_action)],
        "probs": [
            {"action": a, "label": S["action_labels"][a], "p": float(res.probs[a])}
            for a in range(S["n_actions"])
        ],
    })


def objective_semantics():
    """Per-channel (short name, kind). kind='cost' when the channel's command
    range is entirely non-positive (time/fuel/step budgets); 'gain' otherwise.
    This inference classifies every bundled env correctly and degrades safely
    for custom models."""
    high = np.asarray(S["bounds"][1], np.float32)
    names = OBJECTIVE_SHORT.get(S["env_name"])
    if not names or len(names) != len(high):
        names = [f"objective {i}" for i in range(len(high))]
    kinds = ["cost" if float(h) <= 0 else "gain" for h in high]
    return list(names), kinds


def explain_cf(rep, rem_t, collected, hor_t):
    """Plain-language reading of a counterfactual result. The causal claim is
    kept honest: the *request* causes the action, the change is one joint
    adjustment, minimality means a knife-edge flip, and only the NEXT action
    is claimed (realization is a separate step). Returns (detailed, simple):
    the simple variant carries the same content without numbers or jargon."""
    names, kinds = objective_semantics()
    lines, simple = [], []
    a_star_l, foil_l = rep["a_star_label"], rep["a_foil_label"]
    fin = rep.get("final")
    settings = rep.get("settings", {})
    kappa = float(settings.get("margin", settings.get("confidence", 0)) or 0)

    def prob_of(q, action):
        for p in q.get("probs", []):
            if p["action"] == int(action):
                return float(p["p"])
        return float("nan")

    def qual(pct):
        if pct < 0.05:
            return "a little"
        if pct < 0.2:
            return "somewhat"
        if pct < 0.5:
            return "a lot"
        return "far, far"

    if not rep["success"]:
        simple.append(
            f"No possible goal would make the agent choose '{foil_l}' right now — that move is not part of "
            "any good strategy it knows from this situation. Knowing that is the answer.")
        if fin is None:
            lines.append(
                f"No feasible request was found that even moves the agent toward '{foil_l}' at this state — "
                "the search found no direction in the desired returns that raises its preference at all.")
            return lines, simple
        pf = prob_of(fin, rep["a_foil"])
        m = float(fin["target_margin"])
        if m <= -1e6:
            lines.append(
                f"No achievable request flips the agent from '{a_star_l}' to '{foil_l}' here — '{foil_l}' gets "
                "essentially zero preference everywhere in the achievable range; the search never found even a "
                "faint pull toward it.")
            m = float("-inf")
        else:
            lines.append(
                f"No achievable request flips the agent from '{a_star_l}' to '{foil_l}' here. Even at the most "
                f"favourable feasible request the search found, '{foil_l}' reaches only {pf:.1%} preference "
                f"(log-probability margin {m:.2f}, needed ≥ {kappa:g}).")
        if m > -0.5:
            lines.append(
                "It is very close, though — the boundary is probably crossable with a larger query budget, "
                "or from a neighbouring state (try stepping Back or forward once).")
            simple.append("It was very close though — one step earlier or later, it might be possible.")
        elif m > -3.0:
            lines.append(
                "The gap is real but not hopeless — a larger budget might close it, though more likely "
                f"'{foil_l}' is only weakly supported by the plans the agent learned from this state.")
        else:
            lines.append(
                f"The gap is structural: '{foil_l}' lies on none of the optimal plans the agent learned from "
                "this state, so no desired return inside the achievable range supports it. That impossibility "
                "is itself the method's answer.")
        ga = int(fin["greedy_action"])
        if ga not in (int(rep["a_star"]), int(rep["a_foil"])):
            lines.append(
                f"(Along the way the request did flip the action to '{fin['greedy_label']}' — but that answers "
                "a different question than the one you asked.)")
        return lines, simple

    delta = np.asarray(fin["delta"], np.float32)
    cfcmd = np.asarray(fin["command"], np.float32)
    scale = np.asarray(S["scale"], np.float32)
    low, high = [np.asarray(b, np.float32) for b in S["bounds"]]
    order = [int(i) for i in np.argsort(-np.abs(delta) / scale)]

    clauses, sclauses, demands, concessions = [], [], [], []
    for i in order:
        d = float(delta[i])
        pct = abs(d) / float(scale[i])
        if pct < 0.01:
            continue
        amt = f"{abs(d):.3g} ({pct * 100:.0f}% of its range)"
        edge = ""
        if np.isclose(cfcmd[i], low[i], atol=1e-5 + 1e-5 * abs(float(low[i]))) or \
           np.isclose(cfcmd[i], high[i], atol=1e-5 + 1e-5 * abs(float(high[i]))):
            edge = " — the very limit of what is achievable"
        if kinds[i] == "gain":
            if d > 0:
                clauses.append(f"wanted {amt} more {names[i]}{edge}")
                sclauses.append(f"cared {qual(pct)} more about {names[i]}")
                demands.append(names[i])
            else:
                clauses.append(f"been content with {amt} less {names[i]}{edge}")
                sclauses.append(f"cared {qual(pct)} less about {names[i]}")
                concessions.append(names[i])
        else:
            if d < 0:
                clauses.append(f"been willing to accept {amt} more {names[i]}{edge}")
                sclauses.append(f"been willing to spend {qual(pct)} more {names[i]}")
                concessions.append(names[i])
            else:
                clauses.append(f"insisted on {amt} less {names[i]}{edge}")
                sclauses.append(f"wanted to spend {qual(pct)} less {names[i]}")
                demands.append(names[i])

    dropped = 0
    if len(clauses) > 4:
        dropped = len(clauses) - 4
        clauses, sclauses = clauses[:4], sclauses[:4]
    if not clauses:
        lines.append(
            f"The agent was already nearly indifferent here: a negligible adjustment of the request (every "
            f"objective moved by under 1% of its range) is enough to flip its next action from '{a_star_l}' "
            f"to '{foil_l}'.")
        simple.append(
            f"The agent was already almost torn: the tiniest nudge of its goal makes it choose '{foil_l}' "
            f"instead of '{a_star_l}'.")
    else:
        body = clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + ", and " + clauses[-1]
        sbody = sclauses[0] if len(sclauses) == 1 else ", ".join(sclauses[:-1]) + ", and " + sclauses[-1]
        tail = f" (plus {dropped} smaller adjustment{'s' if dropped > 1 else ''})" if dropped else ""
        lines.append(
            f"The agent picked '{a_star_l}' because of what it was asked to deliver from this point on. Had it "
            f"instead {body}{tail}, its next action would have been '{foil_l}'.")
        simple.append(
            f"The agent chose '{a_star_l}' because of the goal it was given. If it had {sbody}, it would have "
            f"chosen '{foil_l}' instead.")
        u_dem = list(dict.fromkeys(demands))
        u_con = list(dict.fromkeys(concessions))
        if u_dem and u_con:
            lines.append(f"In short, this is a trade: {', '.join(u_con)} for {', '.join(u_dem)}.")
            simple.append(f"In short: it is a trade — {', '.join(u_con)} for {', '.join(u_dem)}.")

    pf, ps = prob_of(fin, rep["a_foil"]), prob_of(fin, rep["a_star"])
    lines.append(
        f"This is the smallest such change, so it flips the choice only just: at the counterfactual request the "
        f"preference sits at '{foil_l}' {pf:.1%} vs '{a_star_l}' {ps:.1%}"
        + (f" (with the required safety margin κ={kappa:g})" if kappa > 0 else "")
        + f" — any smaller adjustment keeps '{a_star_l}'.")
    simple.append(
        f"This is the smallest possible change of heart — with it, the agent is almost torn between the two "
        f"moves, but tips over to '{foil_l}'.")

    front = np.asarray(S["front"], np.float32)
    tot_orig = np.asarray(collected, np.float32) + np.asarray(rem_t, np.float32)
    tot_cf = np.asarray(collected, np.float32) + cfcmd
    d_orig = np.linalg.norm((front - tot_orig) / scale, axis=1)
    d_cf = np.linalg.norm((front - tot_cf) / scale, axis=1)
    io, ic = int(np.argmin(d_orig)), int(np.argmin(d_cf))
    if len(front) == 1:
        lines.append(
            f"Context: the learned menu here has a single achievable outcome {_fmt_vec(front[0])} — the "
            "counterfactual bends the request just far enough off it to flip the action; there is no "
            "alternative plan for it to point to.")
        simple.append(
            "There is only one sensible outcome in this world, so the wish bends the agent's goal away from "
            "it just enough to change the move.")
    elif io == ic:
        lines.append(
            f"Against the learned Pareto front: both requests still point at the same achievable outcome "
            f"{_fmt_vec(front[io])} (front #{io}) — the flip happens inside that plan's neighbourhood rather "
            "than by switching plans.")
    else:
        lines.append(
            f"Against the learned Pareto front: the original request pointed at the achievable outcome "
            f"{_fmt_vec(front[io])} (front #{io}); the counterfactual request points nearest to "
            f"{_fmt_vec(front[ic])} (front #{ic}) — the flip amounts to steering toward that alternative plan.")
        simple.append("In effect, the changed goal points the agent toward a different — but still sensible — plan.")
    if float(d_cf[ic]) > 0.15:
        lines.append(
            f"(The counterfactual request itself sits {float(d_cf[ic]):.2f} away from the nearest achievable "
            "outcome — it is a boundary probe rather than a plan; the continue-rollout step re-anchors it "
            "onto the menu.)")
    if S["variant"] == "RH" and hor_t is not None:
        lines.append(
            f"The demanded steps-to-go was held fixed at {float(hor_t):g} throughout — this explanation is "
            "purely about the desired returns.")
    return lines, simple


def run_cf_zoo(a_foil, overrides):
    """Mirrors interactive_pcn_zoo_cf.main() stage for stage."""
    obs_t, rem_t, hor_t = RT["obs"], RT["rem"], RT["hor"]
    model = cf_model(hor_t)
    p = policy_at(obs_t, rem_t, hor_t)
    a_star = int(p["action"])
    action_mask = p["action_mask"]
    if action_mask is None:
        action_mask = np.ones(S["n_actions"], dtype=bool)

    values = dict(APP_ZOO_DEFAULTS)
    overrides = overrides or {}
    for k, v in overrides.items():
        if k in values and v is not None and v != "":
            values[k] = type(values[k])(v)
    # random_directions tracks max_queries unless the user typed an explicit
    # value (matches the batch-eval CLI default): the seed stage should be
    # budget-limited, never direction-limited.
    if not str(overrides.get("random_directions") or "").strip():
        values["random_directions"] = values["max_queries"]
    settings = cf.ZooSettings(**values)

    env_bounds = S["bounds"]
    clip_bounds = env_bounds if settings.clip_command else None
    desired_eff = (rem_t + RT["collected"]).astype(np.float32)

    original = cf.evaluate_delta(
        model, obs_t, rem_t, np.zeros_like(rem_t, dtype=np.float32),
        a_star, a_foil, action_mask, settings, DEVICE, clip_bounds=clip_bounds,
    )

    search = cf.constrained_nearest_boundary_search(
        model, obs_t, rem_t, desired_eff, a_star, a_foil, action_mask,
        settings, DEVICE, env_bounds, front=S["front_commands"],
    )
    queries = int(search["queries"])
    seed_result = search["best_success"]
    final_result = seed_result
    stages = {"seed_success": bool(seed_result is not None and seed_result.success)}

    if seed_result is not None and seed_result.success:
        zoo_clip = search["bounds"] if settings.clip_command else None
        zoo = cf.zoo_coordinate_search(
            model, obs_t, rem_t, a_star, a_foil, action_mask, settings, DEVICE,
            clip_bounds=zoo_clip, initial_delta=seed_result.delta,
        )
        queries += int(zoo["queries"])
        final_result = cf.better_success_scaled(zoo["best_success"], seed_result, rem_t, S["scale"])
        stages["zoo_improved"] = final_result is not seed_result
    else:
        zoo = cf.zoo_coordinate_search(
            model, obs_t, rem_t, a_star, a_foil, action_mask, settings, DEVICE,
            clip_bounds=clip_bounds,
        )
        queries += int(zoo["queries"])
        final_result = zoo["best_success"]
        stages["zoo_improved"] = bool(final_result is not None and final_result.success)

    stages["binary_refined"] = False
    if final_result is not None and final_result.success:
        refined, _btrace, queries = cf.binary_refine_success(
            model, obs_t, rem_t, final_result.delta, a_star, a_foil, action_mask,
            settings, DEVICE, start_queries=queries,
            clip_bounds=search["bounds"] if seed_result is not None and settings.clip_command else clip_bounds,
            max_binary_queries=settings.line_search_steps,
        )
        if refined.success:
            stages["binary_refined"] = True
            final_result = refined
    else:
        final_result = search["best_failure"] or search["best_objective"] or zoo["best_objective"]

    CF.clear()
    CF.update(dict(
        method="zoo", a_star=a_star, a_foil=int(a_foil), t=len(RT["actions"]),
        result=final_result, success=bool(final_result is not None and final_result.success),
        settings=values,
    ))
    CONT.clear()
    CF["report"] = {
        "method": "zoo",
        "a_star": a_star, "a_star_label": S["action_labels"][a_star],
        "a_foil": int(a_foil), "a_foil_label": S["action_labels"][int(a_foil)],
        "t": len(RT["actions"]),
        "queries": queries,
        "stages": stages,
        "settings": values,
        "original": query_result_payload(original),
        "final": query_result_payload(final_result),
        "success": CF["success"],
    }
    CF["report"]["explanation"], CF["report"]["explanation_simple"] = explain_cf(
        CF["report"], rem_t, RT["collected"], hor_t)
    return CF["report"]


def run_cf_cw(a_foil, overrides):
    """Mirrors interactive_pcn_cw.main() stage for stage."""
    import interactive_pcn_cw as cw

    obs_t, rem_t, hor_t = RT["obs"], RT["rem"], RT["hor"]
    model = cf_model(hor_t)
    p = policy_at(obs_t, rem_t, hor_t)
    a_star = int(p["action"])
    action_mask = p["action_mask"]
    if action_mask is None:
        action_mask = np.ones(S["n_actions"], dtype=bool)

    values = dict(cw.DEFAULT_CW_SETTINGS, **APP_CW_OVERRIDES)
    for k, v in (overrides or {}).items():
        if k in values and v is not None and v != "":
            values[k] = type(values[k])(v)
    settings = cw.CwSettings(**values)
    eval_settings = cw.eval_settings_from_cw(settings)

    bounds = S["bounds"]
    clip_bounds = bounds if settings.clip_command else None

    original = cf.evaluate_delta(
        model, obs_t, rem_t, np.zeros_like(rem_t, dtype=np.float32),
        a_star, a_foil, action_mask, eval_settings, DEVICE, clip_bounds=clip_bounds,
    )
    command = cw.run_cw_attack(model, obs_t, rem_t, a_foil, action_mask, settings, DEVICE, bounds)
    final = cf.evaluate_delta(
        model, obs_t, rem_t,
        np.asarray(command, np.float32) - np.asarray(rem_t, np.float32),
        a_star, a_foil, action_mask, eval_settings, DEVICE, clip_bounds=clip_bounds,
    )

    CF.clear()
    CF.update(dict(
        method="cw", a_star=a_star, a_foil=int(a_foil), t=len(RT["actions"]),
        result=final, success=bool(final is not None and final.success),
        settings=values,
    ))
    CONT.clear()
    CF["report"] = {
        "method": "cw",
        "a_star": a_star, "a_star_label": S["action_labels"][a_star],
        "a_foil": int(a_foil), "a_foil_label": S["action_labels"][int(a_foil)],
        "t": len(RT["actions"]),
        "settings": values,
        "original": query_result_payload(original),
        "final": query_result_payload(final),
        "success": CF["success"],
    }
    CF["report"]["explanation"], CF["report"]["explanation_simple"] = explain_cf(
        CF["report"], rem_t, RT["collected"], hor_t)
    return CF["report"]


def original_outcome():
    """Full greedy episode under the original desired command (cached)."""
    key = (tuple(np.round(RT["desired"], 6)), RT["desired_horizon"])
    cache = S["original_outcomes"]
    if key not in cache:
        obs = env_reset()
        rem = RT["desired"].copy()
        hor = RT["desired_horizon"]
        total = np.zeros_like(rem)
        term = False
        for _ in range(ROLLOUT_CAP):
            a = policy_at(obs, rem, hor)["action"]
            obs, r, te, tr, _ = env_step(a)
            total = total + r
            rem = update_command(rem, r)
            if hor is not None:
                hor = max(float(hor) - 1.0, 1.0)
            if te or tr:
                term = bool(te)
                break
        cache[key] = {"total": total, "terminated": term}
    return cache[key]


def audition(issued_return, issued_horizon, a_foil, snap_once, collected0, capture=False):
    """Replay the prefix, issue the candidate command at s_t, roll to the end.
    Returns None if the first action does not flip to the foil."""
    obs, _rem, _hor, _col, term, trunc = replay(RT["actions"])
    if term or trunc:
        return None
    rem = np.asarray(issued_return, np.float32).copy()
    hor = issued_horizon
    suffix = np.zeros_like(rem)
    steps = []
    terminated = False
    truncated = False
    for step in range(ROLLOUT_CAP):
        p = policy_at(obs, rem, hor)
        a = int(p["action"])
        if step == 0 and a != int(a_foil):
            return {"rejected_first_action": a}
        obs, r, te, tr, _ = env_step(a)
        suffix = suffix + r
        naive = (rem - r).astype(np.float32)
        note = None
        if snap_once and step == 0:
            cands = np.clip(
                S["front_commands"] - (collected0 + suffix),
                S["bounds"][0], S["bounds"][1],
            ).astype(np.float32)
            j = int(np.argmin(np.linalg.norm((cands - naive) / S["scale"], axis=1)))
            rem = cands[j].copy()
            note = {"snap_from": jsonable(naive), "snap_to": jsonable(rem)}
        else:
            rem = naive
        if hor is not None:
            hor = max(float(hor) - 1.0, 1.0)
        if capture:
            steps.append({
                "step": step + 1, "action": a, "label": S["action_labels"][a],
                "p": float(p["probs"][a]), "reward": jsonable(r),
                "remaining_after": jsonable(rem),
                "collected_total": jsonable(collected0 + suffix),
                "frame": frame_b64(), "snap": note,
                "terminated": bool(te), "truncated": bool(tr),
            })
        if te or tr:
            terminated, truncated = bool(te), bool(tr)
            break
    total = (collected0 + suffix).astype(np.float32)
    landing_front = S["landing_front"]
    landing_distances = np.linalg.norm((landing_front - total) / S["scale"], axis=1)
    nearest = int(np.argmin(landing_distances))
    fd = float(landing_distances[nearest])
    return {
        "total": total, "fd": fd, "nearest_front_index": nearest,
        "nearest_front_point": landing_front[nearest].copy(),
        "terminated": terminated, "truncated": truncated,
        "n_steps": len(steps) if capture else None, "steps": steps if capture else None,
    }


def run_continue(tol_override=None):
    """Verified realization: audition witness residuals nearest the current
    remaining command, verify the foil action and landing-archive outcome, then
    fall back to R_cf and R_cf + snap-once."""
    if not CF or not CF.get("success"):
        raise ValueError("run a successful counterfactual first")
    a_foil = CF["a_foil"]
    Rcf = np.asarray(CF["result"].command, np.float32)
    t = len(RT["actions"])
    collected0 = RT["collected"].copy()
    rem0 = RT["rem"].copy()
    hor0 = RT["hor"]
    front = S["front"]
    front_commands = S["front_commands"]
    fhor = S["front_horizons"]
    tol = float(tol_override) if tol_override else float(S["landing_tol"])

    cands = np.clip(front_commands - collected0, S["bounds"][0], S["bounds"][1]).astype(np.float32)
    order = np.argsort(np.linalg.norm((cands - rem0) / S["scale"], axis=1))
    trials, seen = [], set()
    for j in order:
        key = tuple(np.round(cands[j], 5))
        if key in seen:
            continue
        seen.add(key)
        h = None if fhor is None else max(float(fhor[j]) - t, 1.0)
        trials.append(("front", cands[int(j)].copy(), h, int(j), False))
    trials.append(("rcf", Rcf.copy(), hor0, -1, False))
    trials.append(("rcf+snap", Rcf.copy(), hor0, -1, True))

    log, got, best = [], None, None
    for tier, cmd, h, j, snap in trials:
        res = audition(cmd, h, a_foil, snap, collected0)
        entry = {
            "tier": tier, "issued": jsonable(cmd),
            "issued_horizon": h,
            "target_front_index": j if j >= 0 else None,
            "target_front_point": jsonable(front[j]) if j >= 0 else None,
        }
        if res is None:
            entry["verdict"] = "prefix ended early"
        elif "rejected_first_action" in res:
            a = res["rejected_first_action"]
            entry["verdict"] = "REJECT (first action)"
            entry["first_action"] = {"action": a, "label": S["action_labels"][a]}
        else:
            entry.update({
                "achieved": jsonable(res["total"]), "fd": round(res["fd"], 5),
                "terminated": res["terminated"],
                "nearest_front_index": res["nearest_front_index"],
            })
            rec = (tier, cmd, h, j, snap, res)
            if best is None or res["fd"] < best[5]["fd"]:
                best = rec
            if res["fd"] < tol:
                entry["verdict"] = "ACCEPT"
                got = rec
            else:
                entry["verdict"] = "rolled out, missed front"
        log.append(entry)
        if got:
            break

    hit = got is not None
    chosen = got if got else best
    try:
        payload = _continue_payload(chosen, log, a_foil, Rcf, t, collected0, rem0, tol, hit)
    finally:
        obs, rem, hor, collected, term, trunc = replay(RT["actions"])
        RT.update(obs=obs, rem=rem, hor=hor, collected=collected, terminated=term, truncated=trunc)
    CONT.clear()
    CONT.update(payload)
    return payload


def _continue_payload(chosen, log, a_foil, Rcf, t, collected0, rem0, tol, hit):
    front = S["front"]
    payload = {
        "tol": tol, "hit": hit, "rollouts_used": len(log), "audition_log": log,
        "original_front_point": jsonable(
            RT.get("front_target") if RT.get("front_target") is not None else RT["desired"]),
        "original_issued_command": jsonable(RT["desired"]),
        "rcf_command": jsonable(Rcf),
        "foil": {"action": a_foil, "label": S["action_labels"][a_foil]},
        "t": t,
        "collected_before": jsonable(collected0),
        "remaining_before": jsonable(rem0),
    }
    if chosen is None:
        payload["outcome"] = "no candidate flipped the first action - nothing to realize"
        return jsonable(payload)

    tier, cmd, h, j, snap, _res = chosen
    final = audition(cmd, h, a_foil, snap, collected0, capture=True)
    total = final["total"]
    og = original_outcome()
    nearest = final["nearest_front_index"]
    target_identity = (
        tier == "front" and j >= 0
        and float(np.linalg.norm((front[j] - total) / S["scale"])) < tol
    )
    payload.update({
        "winner": {
            "tier": tier,
            "issued": jsonable(cmd),
            "issued_horizon": h,
            "target_front_index": j if j >= 0 else None,
            "target_front_point": jsonable(front[j]) if j >= 0 else None,
            "snap_once": snap,
        },
        "achieved_total": jsonable(total),
        "front_distance": round(final["fd"], 5),
        "on_front": final["fd"] < tol,
        "nearest_front_point": jsonable(final["nearest_front_point"]),
        "nearest_front_index": nearest,
        "terminated": final["terminated"],
        "truncated": final["truncated"],
        "target_identity": bool(target_identity),
        "original_outcome": {"total": jsonable(og["total"]), "terminated": og["terminated"]},
        "reverted_to_original": bool(np.allclose(total, og["total"], atol=1e-3)),
        "steps": final["steps"],
    })
    return jsonable(payload)


# ------------------------------------------------------------ env thumbnails

THUMBS_DIR = os.path.join(APP_DIR, "thumbs")


def env_thumbnail(env_name):
    """PNG snapshot of the env at its start state, generated once and cached
    on disk. Uses a throwaway env instance so the live session is untouched."""
    known = {m["env"] for m in list_models()}
    if env_name not in known:
        raise ValueError(f"unknown env '{env_name}'")
    os.makedirs(THUMBS_DIR, exist_ok=True)
    path = os.path.join(THUMBS_DIR, env_name + ".png")
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return fh.read()
    setup, _rm = build_env_setup(env_name)
    env = setup.env
    try:
        out = env.reset(seed=env_seed(env_name))
        if env_name == "walkroom2":
            base = env.unwrapped
            img = walkroom2_frame(env, [tuple(np.asarray(base.pos).tolist())])
        elif env_name == "walkroom3":
            base = env.unwrapped
            img = walkroom3_frame(env, [tuple(np.asarray(base.pos).tolist())])
        else:
            fr = env.render()
            if fr is None:
                img = Image.new("RGB", (360, 240), (23, 28, 38))
                ImageDraw.Draw(img).text((70, 110), "no picture for this world", fill=(140, 150, 170), font=_WFONT)
            else:
                img = Image.fromarray(np.asarray(fr).astype(np.uint8))
        if max(img.size) > 720:
            img.thumbnail((720, 720))
        img.save(path, "PNG")
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        try:
            env.close()
        except Exception:
            pass


# ----------------------------------------------------------- session report

def _fmt_vec(v):
    if v is None:
        return "-"
    return "(" + ", ".join(f"{float(x):.4f}".rstrip("0").rstrip(".") for x in v) + ")"


def build_report():
    from datetime import datetime

    L = []
    add = L.append
    add("=" * 78)
    add("CFZOO Rollout - session report")
    add(f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add("=" * 78)

    add("")
    add("--- MODEL ---")
    add(f"environment : {S.get('env_name', '?')}")
    add(f"variant     : {S.get('variant', '?')}"
        + (" (return+horizon conditioned)" if S.get("variant") == "RH" else " (return conditioned)"))
    if S.get("custom"):
        add(f"custom model: {S['custom']}")
    add(f"device      : {DEVICE}")
    add(f"actions     : " + ", ".join(f"{i}={l}" for i, l in enumerate(S.get("action_labels", []))))
    add(f"front source: {S.get('front_source', '?')}")
    add(f"front size  : {len(S.get('front', []))} displayed Pareto targets"
        + f"   landing archive: {len(S.get('landing_front', []))}")

    add("")
    add("--- ROLLOUT ---")
    if RT:
        if RT.get("front_target") is not None:
            add(f"achieved target : {_fmt_vec(RT['front_target'])}"
                + f"   front index: {RT.get('selected_front_index')}")
        add(f"desired command : {_fmt_vec(RT['desired'])}"
            + (f"   desired horizon: {RT['desired_horizon']}" if RT.get("desired_horizon") is not None else ""))
        add(f"steps taken     : {len(RT['actions'])}")
        add(f"remaining cmd   : {_fmt_vec(RT['rem'])}"
            + (f"   remaining horizon: {RT['hor']}" if RT.get("hor") is not None else ""))
        add(f"collected so far: {_fmt_vec(RT['collected'])}")
        add(f"status          : " + ("TERMINATED" if RT["terminated"] else "TRUNCATED" if RT["truncated"] else "running"))
        if RT["history"]:
            add("history (t | action | p | reward):")
            for h in RT["history"]:
                add(f"  t={h['t']:>4}  {h['label']:<16} p={h['p']:.3f}  reward={_fmt_vec(h['reward'])}")
    else:
        add("no rollout was started")

    add("")
    add("--- COUNTERFACTUAL ---")
    rep = CF.get("report")
    if rep:
        add(f"method   : {'CF-ZOO' if rep['method'] == 'zoo' else 'C&W'}")
        add(f"timestep : t={rep['t']}")
        add(f"flip     : {rep['a_star']}:{rep['a_star_label']}  ->  {rep['a_foil']}:{rep['a_foil_label']}")
        add(f"success  : {rep['success']}")
        if rep["method"] == "zoo":
            add(f"queries  : {rep.get('queries')}   stages: {rep.get('stages')}")
        add("settings : " + json.dumps(rep["settings"]))
        for tag in ("original", "final"):
            q = rep.get(tag)
            if not q:
                continue
            add(f"{tag}:")
            add(f"  command        : {_fmt_vec(q['command'])}")
            add(f"  delta          : {_fmt_vec(q['delta'])}")
            add(f"  distance       : raw L2 {q['l2_norm']:.5f} | scaled {q['scaled_distance']:.5f}")
            add(f"  foil margin    : {q['target_margin']:.5f}   hinge: {q['hinge']:.5f}")
            add(f"  greedy action  : {q['greedy_action']}:{q['greedy_label']}")
            add(f"  probabilities  : " + "  ".join(f"{p['label']}={p['p']:.3f}" for p in q["probs"]))
        if rep.get("explanation"):
            add("in plain words:")
            for s in rep["explanation"]:
                add(f"  - {s}")
    else:
        add("no counterfactual search was run")

    # The verified-realization results stay UI-only by design: reports cover
    # the rollout and the counterfactual search.
    add("")
    add("=" * 78)
    return "\n".join(L)


def save_report():
    from datetime import datetime

    reports = os.path.join(APP_DIR, "reports")
    os.makedirs(reports, exist_ok=True)
    text = build_report()
    name = f"report_{S.get('env_name', 'session')}_{S.get('variant', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path = os.path.join(reports, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return {"ok": True, "path": path, "filename": name, "content": text}


# ------------------------------------------------------- custom model upload

MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def sanitize_model_name(name):
    clean = "".join(c for c in str(name or "").strip() if c.isalnum() or c in "-_ ").strip()
    clean = clean.replace(" ", "-") or "custom-model"
    base, n = clean, 2
    while os.path.isdir(os.path.join(CUSTOM_DIR, clean)):
        clean = f"{base}-{n}"
        n += 1
    return clean


def non_dominated_mask(points):
    points = np.asarray(points, dtype=np.float32)
    keep = np.ones(len(points), dtype=bool)
    for i, c in enumerate(points):
        if keep[i]:
            keep[keep] = np.any(points[keep] > c, axis=1)
            keep[i] = True
    return keep


def front_from_h5(path, n_obj, variant):
    """Extract the logged Pareto front from a PCN run log.h5, exactly like
    eval_pcn.load_logged_commands (returns + horizons for RH, NaN filter,
    non-dominated, horizon = max(h−2, 1), sorted by objective 0)."""
    import h5py

    with h5py.File(path, "r") as log:
        horizons = None
        if "train/final_commands/returns" in log:
            rets = np.asarray(log["train/final_commands/returns"], dtype=np.float32)
            if variant == "RH":
                if "train/final_commands/horizons" not in log:
                    raise ValueError(
                        "this log has no train/final_commands/horizons data "
                        "- cannot build an RH front from it"
                    )
                horizons = np.asarray(
                    log["train/final_commands/horizons"], dtype=np.float32
                ).reshape(-1)
        elif "train/leaves/r/ndarray" in log:
            rets = np.asarray(log["train/leaves/r/ndarray"][-1], dtype=np.float32)
            if variant == "RH":
                if "train/leaves/h/ndarray" not in log:
                    raise ValueError("this log has no train/leaves/h horizons — cannot build an RH front from it")
                horizons = np.asarray(log["train/leaves/h/ndarray"][-1], dtype=np.float32).reshape(-1)
        elif "eval/return/desired/ndarray" in log:
            if variant == "RH":
                raise ValueError("log only has eval desired returns (no horizons) — cannot build an RH front")
            rets = np.asarray(log["eval/return/desired/ndarray"][-1], dtype=np.float32)
        else:
            raise ValueError("log.h5 contains neither train/leaves nor eval/return/desired data")
    if rets.ndim != 2:
        raise ValueError(f"logged returns have unexpected shape {rets.shape}")
    if rets.shape[1] != n_obj:
        raise ValueError(f"logged returns have {rets.shape[1]} objectives but this env has {n_obj}")
    if horizons is not None and len(horizons) != len(rets):
        raise ValueError(
            "logged return and horizon command counts do not match: "
            f"{len(rets)} returns versus {len(horizons)} horizons"
        )
    valid = np.isfinite(rets).all(axis=1)
    if horizons is not None:
        valid = valid & np.isfinite(horizons)
        horizons = horizons[valid]
    rets = rets[valid]
    if len(rets) == 0:
        raise ValueError("log front is empty after NaN filtering")
    keep = non_dominated_mask(rets)
    rets = rets[keep]
    if horizons is not None:
        horizons = np.maximum(horizons[keep] - 2.0, 1.0).astype(np.float32)
    order = np.argsort(rets[:, 0])
    rets = rets[order]
    if horizons is not None:
        return np.concatenate([rets, horizons[order][:, None]], axis=1)
    return rets


def parse_front_text(text, n_obj, variant):
    rows = []
    for ln, line in enumerate(str(text).splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            vals = [float(v) for v in line.replace(",", " ").replace(";", " ").split()]
        except ValueError:
            # tolerate a single CSV header row of column names
            if not rows and any(c.isalpha() for c in line):
                continue
            raise ValueError(f"front line {ln} is not numeric: {line!r}")
        rows.append(vals)
    if not rows:
        raise ValueError("front file contains no data rows")
    want = n_obj + (1 if variant == "RH" else 0)
    lens = {len(r) for r in rows}
    if lens == {n_obj} and variant == "RH":
        raise ValueError(f"RH fronts need {want} columns per row (last = desired horizon); got {n_obj}")
    if lens != {want}:
        raise ValueError(f"expected {want} columns per row for a {variant} front, got row lengths {sorted(lens)}")
    arr = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("front file contains non-finite values")
    return arr


def try_forward(model, obs, ret, hor=None):
    obs_b = torch.as_tensor(np.asarray([obs])).to(DEVICE)
    ret_b = torch.as_tensor(np.asarray([ret], dtype=np.float32)).to(DEVICE)
    with torch.no_grad():
        if hor is None:
            return model(obs_b, ret_b)
        hor_b = torch.as_tensor(np.asarray([[float(hor)]], dtype=np.float32)).to(DEVICE)
        return model(obs_b, ret_b, hor_b)


def upload_custom_model(body):
    """Validate and register a user-supplied PCN checkpoint. Returns a step-by-step
    report; nothing is persisted unless every hard check passes."""
    checks = []

    def ok(step, detail=""):
        checks.append({"step": step, "status": "ok", "detail": str(detail)})

    def warn(step, detail):
        checks.append({"step": step, "status": "warn", "detail": str(detail)})

    def fail(step, detail):
        checks.append({"step": step, "status": "fail", "detail": str(detail)})
        return {"ok": False, "checks": checks}

    env_name = str(body.get("env") or "")
    declared = str(body.get("variant") or "auto").upper()
    known_envs = {m["env"] for m in list_models()}
    if env_name not in known_envs:
        return fail("environment", f"unknown env '{env_name}'; pick one of the listed environments")
    if declared not in ("AUTO", "R", "RH"):
        return fail("variant", f"variant must be auto, R or RH (got {declared!r})")

    raw_b64 = body.get("model_b64") or ""
    if not raw_b64:
        return fail("file", "no model file received")
    try:
        blob = base64.b64decode(raw_b64)
    except Exception:
        return fail("file", "model payload is not valid base64")
    if len(blob) > MAX_UPLOAD_BYTES:
        return fail("file", f"model file is {len(blob)/1e6:.0f} MB (limit 512 MB)")
    ok("file", f"received {len(blob)/1e6:.1f} MB")

    os.makedirs(CUSTOM_DIR, exist_ok=True)
    tmp_path = os.path.join(CUSTOM_DIR, "_upload_tmp.pt")
    with open(tmp_path, "wb") as fh:
        fh.write(blob)

    env_built = None
    try:
        try:
            model = load_model_file(tmp_path)
        except ValueError as exc:
            return fail("load checkpoint", exc)
        ok("load checkpoint", type(model).__module__ + "." + type(model).__name__)

        try:
            setup, _rm = build_env_setup(env_name)
            env_built = setup.env
        except Exception as exc:
            return fail("build environment", exc)
        n_obj = int(np.asarray(setup.max_return).shape[0])
        n_actions = int(setup.n_actions)
        out = env_built.reset(seed=SEED)
        dummy_obs = out[0] if isinstance(out, tuple) else out
        dummy_ret = np.zeros(n_obj, dtype=np.float32)

        # variant detection: try both call signatures with the env's real observation
        errors = {}
        detected = None
        for var, hor in (("R", None), ("RH", 20.0)):
            try:
                out_t = try_forward(model, dummy_obs, dummy_ret, hor)
                detected = var if detected is None else detected
                if var == declared or (declared == "AUTO" and detected == var):
                    break
            except Exception as exc:
                errors[var] = f"{type(exc).__name__}: {exc}"
        if detected is None:
            return fail(
                "forward pass",
                "model rejects this env's observation/command in both R and RH form. "
                f"R attempt: {errors.get('R', '?')} | RH attempt: {errors.get('RH', '?')}",
            )
        if declared == "AUTO":
            variant = detected
            ok("variant", f"auto-detected {variant} ({'takes' if variant == 'RH' else 'no'} horizon input)")
        elif declared in errors:
            return fail("variant", f"declared {declared} but the model rejects that call signature: {errors[declared]}")
        else:
            variant = declared
            ok("variant", f"declared {variant}, call signature accepted")

        out_t = try_forward(model, dummy_obs, dummy_ret, 20.0 if variant == "RH" else None)
        try:
            arr = out_t.detach().cpu().numpy()
        except Exception as exc:
            return fail("output", f"model output is not a tensor: {exc}")
        if arr.ndim != 2 or arr.shape[0] != 1:
            return fail("output", f"expected output shape (1, n_actions), got {arr.shape}")
        if arr.shape[1] != n_actions:
            return fail("output", f"model outputs {arr.shape[1]} actions but {env_name} has {n_actions}")
        if not np.isfinite(arr).all():
            return fail("output", "model produced non-finite log-probs on a real observation")
        probs_sum = float(np.exp(arr[0]).sum())
        if abs(probs_sum - 1.0) > 0.05:
            warn("output", f"output does not look like LogSoftmax log-probs (exp-sum {probs_sum:.3f}); "
                           "argmax-based flow still works, but margins/probabilities may be misleading")
        else:
            ok("output", f"(1, {n_actions}) log-probs, exp-sum {probs_sum:.3f}")

        # Front: user file (txt / CSV / PCN log.h5) or the environment's sole
        # verified 1%-faithful command bank stored beside the bundled checkpoint.
        front_blob = b""
        if body.get("front_b64"):
            try:
                front_blob = base64.b64decode(body["front_b64"])
            except Exception:
                return fail("front file", "front payload is not valid base64")
        arr_front = None
        if front_blob.startswith(b"\x89HDF"):
            h5_tmp = os.path.join(CUSTOM_DIR, "_front_tmp.h5")
            with open(h5_tmp, "wb") as fh:
                fh.write(front_blob)
            try:
                arr_front = front_from_h5(h5_tmp, n_obj, variant)
            except Exception as exc:
                return fail("front file", f"log.h5: {exc}")
            finally:
                try:
                    os.remove(h5_tmp)
                except Exception:
                    pass
            front_source = f"extracted from your log.h5 ({len(arr_front)} non-dominated points)"
        elif front_blob.strip():
            try:
                arr_front = parse_front_text(front_blob.decode("utf-8", errors="replace"), n_obj, variant)
            except ValueError as exc:
                return fail("front file", exc)
            front_source = f"your front file ({len(arr_front)} points)"
        if arr_front is not None:
            pass
        else:
            builtin = verified_front_path(env_name, variant)
            if not os.path.isfile(builtin):
                return fail(
                    "front file",
                    f"no front supplied and no verified 1% {variant} front exists for {env_name}",
                )
            commands, _targets, hz = load_verified_front_file(builtin, variant, n_obj)
            arr_front = np.concatenate([commands, hz[:, None]], axis=1) if hz is not None else commands
            front_source = (
                f"built-in verified 1% {env_name} {variant} command bank "
                f"({len(commands)} points) — supply your own for best results"
            )
            warn("front file", "no front uploaded; using the environment's verified 1% command bank")
        ok("front", front_source)

        # smoke rollout: greedy-follow the first front command end to end
        fr_ret = arr_front[:, :n_obj]
        fr_hor = arr_front[:, n_obj] if variant == "RH" else None
        obs = dummy_obs
        rem = fr_ret[0].copy()
        hor = float(fr_hor[0]) if fr_hor is not None else None
        total = np.zeros(n_obj, dtype=np.float32)
        ended = "cap"
        try:
            for _ in range(600):
                out_t = try_forward(model, obs, rem, hor)
                lp = out_t.detach().cpu().numpy()[0].astype(np.float32)
                masked = apply_action_mask(lp, action_mask_for_env(env_built))
                a = int(np.argmax(masked))
                obs, r, te, tr, _ = cf.step_env(env_built, a)
                total = total + np.asarray(r, np.float32)
                rem = np.clip(rem - r, None, np.asarray(setup.max_return, np.float32)).astype(np.float32)
                if hor is not None:
                    hor = max(hor - 1.0, 1.0)
                if te or tr:
                    ended = "terminated" if te else "truncated"
                    break
        except Exception as exc:
            return fail("smoke rollout", f"rollout crashed at runtime: {type(exc).__name__}: {exc}")
        if ended == "cap":
            warn("smoke rollout", "episode did not end within 600 steps (policy may be degenerate); registered anyway")
        else:
            dist = float(np.linalg.norm((total - fr_ret[0]) / cf.command_scale(cf.command_bounds(setup))))
            ok("smoke rollout", f"followed front[0]={[round(float(x),3) for x in fr_ret[0]]} -> "
                                f"achieved {[round(float(x),3) for x in total]} ({ended}, scaled dist {dist:.4f})")

        # persist
        name = sanitize_model_name(body.get("name"))
        folder = os.path.join(CUSTOM_DIR, name)
        os.makedirs(folder, exist_ok=True)
        os.replace(tmp_path, os.path.join(folder, "model.pt"))
        header = ",".join(f"desired_return_{i}" for i in range(n_obj)) + (",desired_horizon" if variant == "RH" else "")
        with open(os.path.join(folder, "front.txt"), "w", encoding="utf-8") as fh:
            fh.write("# " + header + "\n")
            for row in arr_front:
                fh.write(",".join(repr(float(v)) for v in row) + "\n")
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "name": name, "env": env_name, "variant": variant,
                "n_obj": n_obj, "n_front": int(len(arr_front)),
                "front_source": front_source,
                "model_class": type(model).__module__ + "." + type(model).__name__,
            }, fh, indent=2)
        ok("registered", f"saved as custom model '{name}' — it now appears in the gallery")
        del model
        return {"ok": True, "checks": checks, "id": f"custom::{name}", "name": name,
                "env": env_name, "variant": variant}
    finally:
        if env_built is not None:
            try:
                env_built.close()
            except Exception:
                pass
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ---------------------------------------------------------------- HTTP layer

def model_info_payload(env_name, variant):
    info = dict(ENV_INFO.get(env_name, {}))
    info["variant_note"] = VARIANT_INFO[variant]
    front = S["front"]
    title = info.get("title", env_name)
    if S.get("custom"):
        title = f"{S['custom']} (custom) — {title}"
    return jsonable({
        "env": env_name, "variant": variant,
        "title": title,
        "info": info,
        "n_actions": S["n_actions"],
        "action_labels": S["action_labels"],
        "bounds_low": S["bounds"][0], "bounds_high": S["bounds"][1],
        "max_return": S["max_return"],
        "front": front,
        "front_commands": S["front_commands"],
        "front_horizons": S["front_horizons"],
        "front_fidelity": S["front_fidelity"],
        "front_source": S["front_source"],
        "front_verified": S["front_verified"],
        "landing_front_size": int(len(S["landing_front"])),
        "front_benchmark": VERIFIED_FRONT_BENCHMARKS.get((env_name, variant))
        if S["front_verified"] else None,
        "objectives": info.get("objectives", [f"objective {i}" for i in range(front.shape[1])]),
        "objective_names_short": OBJECTIVE_SHORT.get(env_name, [f"objective {i}" for i in range(front.shape[1])]),
        "render_available": S["render_mode"] == "rgb_array" or env_name.startswith("walkroom"),
        "landing_tol": S["landing_tol"],
        "zoo_defaults": APP_ZOO_DEFAULTS,
        "device": str(DEVICE),
    })


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(APP_DIR, "static", "index.html"), "rb") as fh:
                self._send(200, fh.read(), "text/html; charset=utf-8")
            return
        if self.path == "/api/models":
            with LOCK:
                self._send(200, {"models": list_models() + list_custom_models(), "device": str(DEVICE)})
            return
        if self.path == "/api/state":
            with LOCK:
                self._send(200, state_payload())
            return
        if self.path.startswith("/api/thumb/"):
            from urllib.parse import unquote
            env_name = unquote(self.path[len("/api/thumb/"):])
            try:
                with LOCK:
                    data = env_thumbnail(env_name)
                self._send(200, data, "image/png")
            except Exception as exc:
                self._send(404, {"error": str(exc)})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            with LOCK:
                self._route(body)
        except Exception:
            self._send(500, {"error": traceback.format_exc()})

    def _route(self, body):
        path = self.path
        if path == "/api/load":
            env_name, variant = load_model(
                body.get("env"), body.get("variant"), custom=body.get("custom"),
            )
            self._send(200, model_info_payload(env_name, variant))
        elif path == "/api/upload_model":
            self._send(200, upload_custom_model(body))
        elif path == "/api/delete_custom":
            name = str(body.get("name") or "")
            existing = {m["id"].split("::", 1)[1] for m in list_custom_models()}
            folder = os.path.join(CUSTOM_DIR, name)
            if name not in existing or not os.path.isdir(folder):
                self._send(404, {"error": f"unknown custom model '{name}'"})
            else:
                if S.get("custom") == name:
                    close_session()
                shutil.rmtree(folder, ignore_errors=True)
                self._send(200, {"ok": True, "deleted": name})
        elif path == "/api/start":
            if body.get("front_index") is not None:
                j = int(body["front_index"])
                if j < 0 or j >= len(S["front"]):
                    raise ValueError(f"front index {j} is outside 0..{len(S['front']) - 1}")
                desired = S["front_commands"][j]
                horizon = None if S["front_horizons"] is None else float(S["front_horizons"][j])
                front_index = j
            else:
                desired = np.asarray([float(v) for v in body["custom_return"]], np.float32)
                horizon = None
                front_index = None
                if S["front_horizons"] is not None:
                    horizon = float(body.get("custom_horizon") or float(np.max(S["front_horizons"])))
            start_rollout(desired, horizon, front_index=front_index)
            self._send(200, state_payload())
        elif path == "/api/step":
            do_step()
            self._send(200, state_payload())
        elif path == "/api/back":
            do_back()
            self._send(200, state_payload())
        elif path == "/api/cf":
            method = body.get("method", "zoo")
            a_foil = int(body["a_foil"])
            if method == "cw":
                self._send(200, run_cf_cw(a_foil, body.get("settings")))
            else:
                self._send(200, run_cf_zoo(a_foil, body.get("settings")))
        elif path == "/api/continue":
            self._send(200, run_continue(body.get("tol")))
        elif path == "/api/finish":
            if body.get("save"):
                result = save_report()
            else:
                result = {"ok": True}
            CF.clear()
            CONT.clear()
            self._send(200, result)
        else:
            self._send(404, {"error": "not found"})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8901
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"CFZOO_Rollout running on http://127.0.0.1:{port}  (device: {DEVICE})")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with LOCK:
            close_session()


if __name__ == "__main__":
    main()
