"""Goal-influence signal for a reward-only PCN (broad-generic same-state baseline).

WHAT IT COMPUTES
----------------
At a visited state s_t with residual return command R_t, the policy's action
distribution is p_goal(a) = pi(a | s_t, R_t). We compare it to a same-state,
command-NEUTRAL baseline p_base(s_t) -- what the same policy does at s_t when
averaged over a broad fan of generic return commands -- using KL divergence:

        I(s_t, R_t) = D_KL( p_goal || p_base ).

  * High I  -> the GOAL (the requested return) is what drives the action here.
  * Low  I  -> the action is driven by the STATE; the goal barely matters.

This separates "is the policy confident?" (entropy) from "is that choice caused
by the goal?" (this signal).

We use the broad-generic baseline, which the baseline study selected as the best
of the three candidates (zero / generic / sweep): it was the most faithful to an
exact dense-grid behavioral ground truth across DST (2-obj), three-tree and
minecart (3-obj), needs no tuning, and never degenerates.

Probe magnitudes are BOX-RELATIVE: each trade-off direction is scaled per objective
by fractions of that objective's feasible range. A second ablation compared this to
alpha*||R_t|| (scale-dependent) and Pareto-front quantile (front-geometry-dependent)
magnitude schemes; box-relative was the most consistent across all three envs and is
invariant to both the current command magnitude and the logged front's spread.

BLACK-BOX: needs only (s, R) -> pi(.|s,R). No weights, replay, env-branching, or
training data. Post-hoc explanation of an already-trained reward-only PCN.

INPUT  (in practice): a trained PCN run + a desired-return command (a Pareto-front
                      policy index, or several) to explain.
OUTPUT (in practice): a per-step influence profile I(s_0..s_{T-1}) along that
                      policy's trajectory, plus the scalar summary I(tau), saved
                      as CSV + a plot under goal_output_artifacts/. Optionally
                      validated against a dense scan.

Usage:
    python goal_influence/goal_influence.py --run dst --front 9   # explain one policy
    python goal_influence/goal_influence.py --run dst             # explain all front rows
    python goal_influence/goal_influence.py --run dst --validate  # + faithfulness check
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# Direct execution sets sys.path to this subfolder. Add the repository root so
# the existing shared PCN, artifact, plotting, and counterfactual modules remain
# importable. Package/module execution already has the root on sys.path.
if __package__ in (None, ""):
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py

import interactive_pcn_zoo_cf as cf
from artifacts import load_run_metadata, resolve_run_dir
from device_utils import device_description
from eval_pcn import load_logged_front, resolve_checkpoint_path
from goal_influence import goal_output_artifacts_dir
from paper_plot_utils import save_paper_figure
from pcn.env_setup import build_experiment_setup

EPS = 1e-8
# Probe magnitudes as fractions of each objective's feasible range (box-relative).
# Chosen over alpha*||R_t|| and Pareto-quantile schemes: most faithful+robust across
# DST/three-tree/minecart, invariant to both command magnitude and front geometry.
DEFAULT_FRACTIONS = tuple(float(np.round(f, 4)) for f in np.arange(0.02, 1.0 + 1e-9, 0.02))


# --------------------------------------------------------------------------- #
# Black-box policy query: fixed state, many commands -> action distributions   #
# --------------------------------------------------------------------------- #
def query_probs_batch(model, obs, commands, action_mask, device):
    commands = np.asarray(commands, dtype=np.float32).reshape(len(commands), -1)
    obs_batch = np.repeat(np.asarray([obs]), len(commands), axis=0)
    with torch.no_grad():
        log_probs = model(
            torch.as_tensor(obs_batch).to(device),
            torch.as_tensor(commands, dtype=torch.float32).to(device),
        ).detach().cpu().numpy()
    masked = log_probs.astype(np.float32).copy()
    if action_mask is not None:
        masked[:, ~np.asarray(action_mask, dtype=bool)] = -1e9
    shifted = masked - masked.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / probs.sum(axis=1, keepdims=True)
    greedy = masked.argmax(axis=1).astype(np.int64)
    return probs.astype(np.float32), greedy


def _valid_pair(p_goal, p_base, action_mask):
    valid = np.ones(len(p_goal), dtype=bool) if action_mask is None else np.asarray(action_mask, dtype=bool)
    pg = np.asarray(p_goal, np.float64)[valid] + EPS
    pb = np.asarray(p_base, np.float64)[valid] + EPS
    return pg / pg.sum(), pb / pb.sum()


def kl_divergence(p_goal, p_base, action_mask):
    """KL(p_goal||p_base) in nats. Range [0, inf), asymmetric, unbounded."""
    pg, pb = _valid_pair(p_goal, p_base, action_mask)
    return float(np.sum(pg * np.log(pg / pb)))


def tv_distance(p_goal, p_base, action_mask):
    """Total variation = 1/2 * sum|p_goal-p_base|. Range [0,1]: fraction of mass moved."""
    pg, pb = _valid_pair(p_goal, p_base, action_mask)
    return float(0.5 * np.sum(np.abs(pg - pb)))


def js_divergence(p_goal, p_base, action_mask):
    """Jensen-Shannon divergence in nats. Range [0, ln2], symmetric, bounded."""
    pg, pb = _valid_pair(p_goal, p_base, action_mask)
    m = 0.5 * (pg + pb)
    return float(0.5 * np.sum(pg * np.log(pg / m)) + 0.5 * np.sum(pb * np.log(pb / m)))


def all_scores(p_goal, p_base, action_mask):
    return {
        "kl": kl_divergence(p_goal, p_base, action_mask),
        "tv": tv_distance(p_goal, p_base, action_mask),
        "js": js_divergence(p_goal, p_base, action_mask),
    }


# --------------------------------------------------------------------------- #
# The method: broad-generic same-state baseline + goal-influence score         #
# --------------------------------------------------------------------------- #
def generic_directions(dim):
    """Trade-off directions in objective space: one-hot +/- per objective and
    balanced sign-corners. (For high dim, the 2^dim corners can be dropped.)"""
    dirs = []
    for j in range(dim):
        e = np.zeros(dim, dtype=np.float32); e[j] = 1.0; dirs.append(e.copy())
        e[j] = -1.0; dirs.append(e.copy())
    for signs in np.array(np.meshgrid(*[[1.0, -1.0]] * dim)).T.reshape(-1, dim):
        dirs.append(signs.astype(np.float32))
    return dirs


def generic_baseline(model, obs, command, action_mask, bounds, device, fractions=DEFAULT_FRACTIONS):
    """p_base = average action distribution over a broad fan of generic commands.

    Probe commands (BOX-RELATIVE): each trade-off direction is scaled per objective
    by a fraction f of that objective's feasible range (high-low), for f in
    `fractions`, plus the normalized current-target direction, all clipped to the
    valid command box. This is invariant to the current command's magnitude and to
    the logged front's geometry -- the most robust of the magnitude schemes tested.
    """
    command = np.asarray(command, dtype=np.float32)
    low, high = np.asarray(bounds[0], np.float32), np.asarray(bounds[1], np.float32)
    box_range = high - low
    dim = len(command)
    probe_dirs = generic_directions(dim)
    probe_dirs.append((command / (np.linalg.norm(command) + EPS)).astype(np.float32))
    commands = []
    for d in probe_dirs:
        unit = d / (np.linalg.norm(d) + EPS)
        for f in fractions:
            commands.append(np.clip(f * (box_range * unit), low, high).astype(np.float32))
    commands = np.unique(np.round(np.asarray(commands, np.float32), 5), axis=0)
    probs, greedy = query_probs_batch(model, obs, commands, action_mask, device)
    return probs.mean(axis=0), len(commands), greedy


def goal_influence_at_state(model, obs, command, action_mask, bounds, device, fractions=DEFAULT_FRACTIONS):
    """Return {kl,tv,js,flip} scores, the goal-conditioned dist, the baseline dist, and #queries.

    `flip` is a free behavioral companion computed from the same probe fan: the
    fraction of probe commands whose greedy action differs from the greedy action
    under the actual command. The averaged-distribution divergences can miss states
    where the argmax flips across goals while the mean distribution stays close
    (observed on DST cell 50); the flip fraction reads that directly.
    """
    p_goal, g_star = query_probs_batch(model, obs, [command], action_mask, device)
    p_goal = p_goal[0]
    p_base, n_base, probe_greedy = generic_baseline(model, obs, command, action_mask, bounds, device, fractions)
    scores = all_scores(p_goal, p_base, action_mask)
    scores["flip"] = float(np.mean(probe_greedy != int(g_star[0])))
    return scores, p_goal, p_base, n_base + 1


# --------------------------------------------------------------------------- #
# Explanation: per-step influence profile along a Pareto-front policy          #
# --------------------------------------------------------------------------- #
def explain_trajectory(env, model, front, front_index, device, bounds, fractions, max_timesteps):
    desired = front[front_index].astype(np.float32)
    obs, _ = cf.reset_env(env)
    remaining = desired.copy()
    rows = []
    for t in range(max_timesteps):
        pol = cf.greedy_action(model, env, obs, remaining, device)
        mask = pol["action_mask"]
        if mask is None:
            mask = np.ones(len(pol["probs"]), dtype=bool)
        scores, p_goal, p_base, q = goal_influence_at_state(model, obs, remaining, mask, bounds, device, fractions)
        rows.append({
            "front_index": int(front_index),
            "desired_return": [float(x) for x in desired],
            "timestep": int(t),
            "command_R_t": [float(x) for x in remaining],
            "greedy_action": int(pol["action"]),
            "greedy_label": cf.action_label(env, int(pol["action"])),
            "action_probs": [float(x) for x in p_goal],
            "baseline_probs": [float(x) for x in p_base],
            "I_kl": float(scores["kl"]),
            "I_tv": float(scores["tv"]),
            "I_js": float(scores["js"]),
            "I_flip": float(scores["flip"]),
            "queries": int(q),
            "obs": np.asarray(obs).copy(),
            "action_mask": np.asarray(mask, bool).copy(),
        })
        nobs, r, term, trunc, _ = cf.step_env(env, pol["action"])
        remaining = (remaining - r).astype(np.float32)
        obs = nobs
        if term or trunc:
            break
    return rows


# --------------------------------------------------------------------------- #
# Optional: faithfulness validation against an exact dense command-box scan    #
# --------------------------------------------------------------------------- #
MC_SAMPLES = 50_000
GT_CHUNK = 200_000


def grid_ground_truth(model, obs, command, action_mask, bounds, device, grid_n):
    """Behavioral ground truth over the command box. Exhaustive grid (grid_n per
    dimension) for command spaces up to 3-D; uniform Monte-Carlo samples (seeded)
    above that, where a dense lattice is intractable (e.g. 6-objective fruit-tree)."""
    low, high = np.asarray(bounds[0], np.float32), np.asarray(bounds[1], np.float32)
    scale = cf.command_scale(bounds)
    dim = len(command)
    if dim <= 3:
        axes = [np.linspace(low[j], high[j], grid_n, dtype=np.float32) for j in range(dim)]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim).astype(np.float32)
        method = "grid"
    else:
        rng = np.random.default_rng(0)
        mesh = (low + rng.random((MC_SAMPLES, dim)) * (high - low)).astype(np.float32)
        method = "mc"
    greedy = np.concatenate([
        query_probs_batch(model, obs, mesh[i:i + GT_CHUNK], action_mask, device)[1]
        for i in range(0, len(mesh), GT_CHUNK)])
    a_star = int(query_probs_batch(model, obs, [command], action_mask, device)[1][0])
    flips = greedy != a_star
    nearest = float("inf")
    if np.any(flips):
        nearest = float(np.min(np.linalg.norm((mesh[flips] - np.asarray(command, np.float32)) / scale, axis=1)))
    return {"flip_fraction": float(np.mean(flips)), "flip_exists": bool(np.any(flips)),
            "nearest_flip_scaled_l2": nearest, "gt_method": method}


def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float); ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) < EPS or np.std(y[m]) < EPS:
        return float("nan")
    return float(np.corrcoef(rankdata(x[m]), rankdata(y[m]))[0, 1])


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
# I/O                                                                          #
# --------------------------------------------------------------------------- #
def load_run(run, checkpoint, device):
    run_dir = resolve_run_dir(run)
    md = load_run_metadata(run_dir)
    setup = build_experiment_setup(md.get("env", "dst"), device=device,
                                   fruit_tree_depth=int(md.get("fruit_tree_depth", 6)), include_model=False)
    ckpt, _ = resolve_checkpoint_path(run_dir, checkpoint)
    model = torch.load(ckpt, map_location=device, weights_only=False).to(device)
    if hasattr(model, "scaling_factor"):
        model.scaling_factor = model.scaling_factor.to(device)
    model.eval()
    reward_dim = int(np.asarray(setup.max_return).shape[-1])
    with h5py.File(run_dir / "log.h5", "r") as log:
        front = load_logged_front(log, list(range(reward_dim))).astype(np.float32)
    return run_dir, setup, model, front, ckpt


def plot_profile(path, env, rows):
    ts = [r["timestep"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9.0, 4.6))
    ax1.plot(ts, [r["I_kl"] for r in rows], marker="o", color="#1f4e79", linewidth=2, label="KL (nats, unbounded)")
    ax1.set_xlabel("timestep"); ax1.set_ylabel("KL (nats)")
    for r in rows:
        ax1.annotate(r["greedy_label"], (r["timestep"], r["I_kl"]),
                     textcoords="offset points", xytext=(0, 6), fontsize=7.5, ha="center")
    ax2 = ax1.twinx()
    ax2.plot(ts, [r["I_tv"] for r in rows], marker="s", color="#2ca02c", linewidth=1.6, label="TV [0,1]")
    ax2.plot(ts, [r["I_js"] for r in rows], marker="^", color="#d62728", linewidth=1.6, label="JS (nats, <=ln2)")
    ax2.plot(ts, [r["I_flip"] for r in rows], marker="d", color="#9467bd", linewidth=1.4,
             linestyle="--", label="probe flip% [0,1]")
    ax2.set_ylabel("TV / JS / flip% (bounded)")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labs, loc="best", fontsize=8)
    ax1.set_title(f"Per-step goal influence (front {rows[0]['front_index']}, "
                  f"desired={np.round(rows[0]['desired_return'],2)})")
    ax1.grid(alpha=0.25); fig.tight_layout()
    save_paper_figure(fig, path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Goal-influence explainer for a reward-only PCN (broad-generic baseline).")
    ap.add_argument("--run", default="dst")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--front", default="all", help="Pareto-front policy index to explain, or 'all'")
    ap.add_argument("--max-fronts", type=int, default=None,
                    help="with --front all: evenly subsample the front to at most this many rows")
    ap.add_argument("--max-timesteps", type=int, default=60)
    ap.add_argument("--fraction-step", type=float, default=0.02,
                    help="box-relative probe step: fractions {step, 2*step, ..., 1.0} of each objective range")
    ap.add_argument("--label", default="goal_influence")
    ap.add_argument("--validate", action="store_true", help="also score faithfulness vs a dense command-box scan")
    ap.add_argument("--grid", type=int, default=100,
                    help="dense grid per dim for --validate (<=3-D; higher dims use MC samples)")
    ap.add_argument("--state-basis", action="store_true",
                    help="aggregate influence per state over the reachable commands across all explained fronts")
    args = ap.parse_args()

    device = args.device
    run_dir, setup, model, front, ckpt = load_run(args.run, args.checkpoint, device)
    bounds = cf.command_bounds(setup)
    env = setup.env
    step = float(args.fraction_step)
    fractions = tuple(float(np.round(f, 4)) for f in np.arange(step, 1.0 + 1e-9, step))

    out_dir = goal_output_artifacts_dir() / f"{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Reward-only PCN goal-influence explainer (broad-generic baseline)")
    print("=" * 72)
    print(f"Run {run_dir.name} | env {setup.env_name} | checkpoint {ckpt.name}")
    print(f"Device {device_description(device)} | command bounds {bounds}")

    front_indices = list(range(len(front))) if args.front == "all" else [int(args.front)]
    if args.max_fronts and len(front_indices) > args.max_fronts:
        front_indices = sorted({int(round(i)) for i in np.linspace(0, len(front) - 1, args.max_fronts)})
    all_rows = []
    try:
        for fi in front_indices:
            rows = explain_trajectory(env, model, front, fi, device, bounds, fractions, args.max_timesteps)
            all_rows.extend(rows)
            tau_kl = float(np.mean([r["I_kl"] for r in rows]))
            tau_tv = float(np.mean([r["I_tv"] for r in rows]))
            tau_js = float(np.mean([r["I_js"] for r in rows]))
            tau_fl = float(np.mean([r["I_flip"] for r in rows]))
            print(f"\nFront {fi}  desired={np.round(front[fi], 2)}  steps={len(rows)}  "
                  f"mean: KL={tau_kl:.3f} TV={tau_tv:.3f} JS={tau_js:.3f} flip%={tau_fl:.3f}")
            print(f"  {'t':>3} {'greedy':<8} {'KL':>8} {'TV':>7} {'JS':>7} {'flip%':>7}   action probs            R_t")
            for r in rows:
                print(f"  {r['timestep']:>3} {r['greedy_label']:<8} {r['I_kl']:>8.3f} {r['I_tv']:>7.3f} "
                      f"{r['I_js']:>7.3f} {r['I_flip']:>7.3f}   "
                      f"{np.round(r['action_probs'], 2)}   {np.round(r['command_R_t'], 2)}")
        # plot the longest trajectory as the representative figure
        by_front = {}
        for r in all_rows:
            by_front.setdefault(r["front_index"], []).append(r)
        rep = max(by_front.values(), key=len)
        plot_profile(out_dir / f"profile_front{rep[0]['front_index']}.png", env, rep)

        validation = None
        if args.validate:
            gts = [grid_ground_truth(model, r["obs"], np.asarray(r["command_R_t"], np.float32),
                                     r["action_mask"], bounds, device, args.grid) for r in all_rows]
            flip_frac = [g["flip_fraction"] for g in gts]
            flip_exists = [g["flip_exists"] for g in gts]
            nearest = [g["nearest_flip_scaled_l2"] for g in gts]
            nf = [(i, n) for i, n in enumerate(nearest) if np.isfinite(n)]
            validation = {"n_states": len(all_rows), "n_goal_active": int(sum(flip_exists)),
                          "gt_method": gts[0]["gt_method"], "scores": {}}
            for name in ("kl", "tv", "js", "flip"):
                I = [r[f"I_{name}"] for r in all_rows]
                validation["scores"][name] = {
                    "spearman_flip_fraction": spearman(I, flip_frac),
                    "auroc_detect_goal_active": auroc(I, flip_exists),
                    "spearman_nearest_flip": spearman([I[i] for i, _ in nf], [n for _, n in nf]),
                }
            print(f"\nFaithfulness vs behavioral ground truth ({validation['gt_method']}; "
                  f"{validation['n_states']} states, {validation['n_goal_active']} goal-active):")
            print(f"  {'score':<6}{'Spearman(flip%)':>17}{'AUROC':>9}{'Spearman(nearest)':>19}")
            for name in ("kl", "tv", "js", "flip"):
                s = validation["scores"][name]
                def f(x): return f"{x:.3f}" if np.isfinite(x) else "  n/a"
                print(f"  {name.upper():<6}{f(s['spearman_flip_fraction']):>17}{f(s['auroc_detect_goal_active']):>9}"
                      f"{f(s['spearman_nearest_flip']):>19}")
    finally:
        try:
            env.close()
        except Exception:
            pass

    if args.state_basis:
        from collections import defaultdict
        cells = defaultdict(list)
        for r in all_rows:
            key = tuple(np.asarray(r["obs"]).reshape(-1).tolist())
            cells[key].append(r)

        # Goal-influence per state = divergence of each reachable goal's action
        # distribution to the REACHABLE CENTROID (the agent's average behaviour over
        # the goals actually fed here). For KL this is the goal->action mutual
        # information under the reachable-command distribution. Referencing the
        # centroid (instead of the global box-fan baseline) removes the forced
        # final-dive spike and the single-visit inflation, so the value reflects how
        # much the goal actually CHANGES THE ACTION at this state.
        stats = {}
        for key, items in cells.items():
            probs = np.asarray([it["action_probs"] for it in items], dtype=np.float64)
            mask = items[0]["action_mask"]
            centroid = probs.mean(axis=0)
            counts = defaultdict(int)
            for it in items:
                counts[it["greedy_label"]] += 1
            stats[key] = {
                "n": len(items),
                "kl": np.array([kl_divergence(p, centroid, mask) for p in probs]),
                "tv": np.array([tv_distance(p, centroid, mask) for p in probs]),
                "js": np.array([js_divergence(p, centroid, mask) for p in probs]),
                "fronts": sorted({int(it["front_index"]) for it in items}),
                "actions": "/".join(f"{a}x{c}" for a, c in counts.items()),
                "R_t_seen": [it["command_R_t"] for it in items],
            }

        print("\n" + "=" * 104)
        print("State basis (c, fixed): goal->action divergence vs the REACHABLE CENTROID "
              "(how much the goal changes the action here)")
        print("=" * 104)
        print(f"  {'state':>6} {'#v':>3} {'actions':<16} "
              f"{'KL mean(min..max)':>22} {'TV mean(min..max)':>22} {'JS mean(min..max)':>22}  fronts")
        cell_str = lambda m: f"{m.mean():.3f}({m.min():.2f}..{m.max():.2f})"
        basis_rows = []
        for key in sorted(stats, key=lambda k: float(stats[k]["kl"].mean()), reverse=True):
            s = stats[key]
            state_str = str(key[0]) if len(key) == 1 else str(key)
            print(f"  {state_str:>6} {s['n']:>3} {s['actions']:<16} "
                  f"{cell_str(s['kl']):>22} {cell_str(s['tv']):>22} {cell_str(s['js']):>22}  {s['fronts']}")
            basis_rows.append({
                "state": state_str, "n_visits": s["n"], "distinct_actions": s["actions"],
                "fronts": json.dumps(s["fronts"]),
                "kl_mean": float(s["kl"].mean()), "kl_min": float(s["kl"].min()), "kl_max": float(s["kl"].max()),
                "tv_mean": float(s["tv"].mean()), "tv_min": float(s["tv"].min()), "tv_max": float(s["tv"].max()),
                "js_mean": float(s["js"].mean()), "js_min": float(s["js"].min()), "js_max": float(s["js"].max()),
                "R_t_seen": json.dumps(s["R_t_seen"]),
            })
        with (out_dir / "state_basis.csv").open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(basis_rows[0].keys())); w.writeheader(); w.writerows(basis_rows)
        print("\n  Influence is measured vs the reachable centroid (goal->action MI). "
              "Single-visit states -> 0 (one goal seen, no variation); forks -> high; always-same-action -> ~0.")

    serializable = []
    for r in all_rows:
        d = {k: v for k, v in r.items() if k not in ("obs", "action_mask")}
        for key in ("desired_return", "command_R_t", "action_probs", "baseline_probs"):
            d[key] = json.dumps(d[key])
        serializable.append(d)
    with (out_dir / "influence_profile.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(serializable[0].keys())); w.writeheader(); w.writerows(serializable)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as h:
        json.dump({"run": run_dir.name, "env": setup.env_name, "method": "broad_generic_goal_influence",
                   "fronts_explained": front_indices, "validation": validation}, h, indent=2)

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
