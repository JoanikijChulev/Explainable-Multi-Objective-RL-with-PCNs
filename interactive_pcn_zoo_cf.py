import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from artifacts import checkpoints_dir, load_run_metadata, resolve_run_dir
from device_utils import device_description, preferred_device
from eval_pcn import load_logged_front, resolve_checkpoint_path
from paper_plot_utils import save_paper_figure
from pcn.env_setup import build_experiment_setup, canonical_env_name
from pcn.pcn import action_mask_for_env, apply_action_mask




DEFAULT_ZOO_SETTINGS = {

    # Ray-search parameters. random_directions should be >= max_queries so the
    # ray-seed stage is budget-limited, never direction-limited.
    "random_directions": 9000,
    "line_search_steps": 16,

    # CW/ZOO loss parameters
    "margin": 0.05,
    "c": 1.0,

    # ZOO-ADAM finite-difference optimizer parameters
    "learning_rate": 0.01,
    "finite_diff_h": 1e-3,
    "max_queries": 9000,

    "seed": 0,
    "clip_command": True,
}


@dataclass
class ZooSettings:
    margin: float
    c: float
    learning_rate: float
    finite_diff_h: float
    max_queries: int
    random_directions: int
    line_search_steps: int
    seed: int
    clip_command: bool


@dataclass
class CFQueryResult:
    delta: np.ndarray
    command: np.ndarray
    masked_log_probs: np.ndarray
    probs: np.ndarray
    objective: float
    hinge: float
    target_margin: float
    success: bool
    greedy_action: int
    l2_norm: float


def ask_choice(prompt, choices, default=None):
    choices = [str(choice) for choice in choices]
    choice_map = {choice.lower(): choice for choice in choices}
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return str(default)
        selected = choice_map.get(raw.lower())
        if selected is not None:
            return selected
        print(f"Invalid choice. Choose one of: {', '.join(choices)}")


def ask_int(prompt, default=None, minimum=None, maximum=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            value = int(default)
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Invalid integer.")
                continue
        if minimum is not None and value < minimum:
            print(f"Value must be at least {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Value must be at most {maximum}.")
            continue
        return value


def parse_vector(text, expected_dim=None):
    pieces = [piece for piece in re.split(r"[\s,]+", text.strip()) if piece]
    if not pieces:
        raise ValueError("empty vector")
    values = np.asarray([float(piece) for piece in pieces], dtype=np.float32)
    if expected_dim is not None and len(values) != int(expected_dim):
        raise ValueError(f"expected {expected_dim} values, got {len(values)}")
    return values


def format_vector(vector, precision=4):
    return np.array2string(np.asarray(vector), precision=precision, floatmode="fixed")


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def load_run_and_model():
    device = preferred_device()

    while True:
        raw_run = input("Run name or run directory: ").strip()
        if not raw_run:
            print("Enter a run name under output_artifacts/ or a run directory path.")
            continue
        try:
            run_dir = resolve_run_dir(raw_run)
            break
        except FileNotFoundError as exc:
            print(exc)

    metadata = load_run_metadata(run_dir)
    env_name = metadata.get("env")
    if env_name is None:
        env_name = input("Run metadata has no env. Enter env name: ").strip()
    env_name = canonical_env_name(env_name)
    fruit_tree_depth = int(metadata.get("fruit_tree_depth", 6))

    setup = build_experiment_setup(
        env_name,
        device=device,
        fruit_tree_depth=fruit_tree_depth,
        include_model=False,
    )

    checkpoint_root = checkpoints_dir(run_dir)
    try:
        latest_checkpoint, checkpoints = resolve_checkpoint_path(run_dir, None)
    except (AssertionError, FileNotFoundError) as exc:
        raise FileNotFoundError(f"could not find checkpoints for {run_dir}: {exc}") from exc

    print(f"\nCheckpoint directory: {checkpoint_root}")
    print("Available checkpoints:")
    for index, checkpoint in enumerate(checkpoints):
        marker = " (latest)" if checkpoint == latest_checkpoint else ""
        print(f"  [{index}] {checkpoint.name}{marker}")

    while True:
        raw_checkpoint = input(f"Checkpoint index/name/path [default: {latest_checkpoint.name}]: ").strip()
        if not raw_checkpoint:
            checkpoint_path = latest_checkpoint
            break
        if raw_checkpoint.isdigit():
            index = int(raw_checkpoint)
            if 0 <= index < len(checkpoints):
                checkpoint_path = checkpoints[index]
                break
            print(f"Index out of range. Choose 0 to {len(checkpoints) - 1}.")
            continue
        try:
            checkpoint_path, _ = resolve_checkpoint_path(run_dir, raw_checkpoint)
            break
        except FileNotFoundError as exc:
            print(exc)

    model = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device)
    if hasattr(model, "scaling_factor"):
        model.scaling_factor = model.scaling_factor.to(device)
    model.eval()

    print(f"\nLoaded env: {setup.env_name}")
    print(f"Using device: {device_description(device)}")
    print(f"Using checkpoint: {checkpoint_path.name}")

    return {
        "run_dir": run_dir,
        "metadata": metadata,
        "setup": setup,
        "env": setup.env,
        "model": model,
        "checkpoint_path": checkpoint_path,
        "device": device,
    }


def query_log_probs(model, obs, command, device):
    obs_batch = np.asarray([obs])
    command_batch = np.asarray([command], dtype=np.float32)
    with torch.no_grad():
        log_probs = model(
            torch.as_tensor(obs_batch).to(device),
            torch.as_tensor(command_batch, dtype=torch.float32).to(device),
        )
    log_probs = log_probs.detach().cpu().numpy()
    if log_probs.ndim != 2 or log_probs.shape[0] != 1:
        raise ValueError(f"model returned unexpected log-prob shape: {log_probs.shape}")
    return log_probs[0].astype(np.float32)


def probabilities_from_log_probs(masked_log_probs):
    masked_log_probs = np.asarray(masked_log_probs, dtype=np.float32)
    shifted = masked_log_probs - np.max(masked_log_probs)
    probs = np.exp(shifted)
    total = np.sum(probs)
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(masked_log_probs, 1.0 / len(masked_log_probs), dtype=np.float32)
    return (probs / total).astype(np.float32)


def greedy_action(model, env, obs, command, device):
    raw_log_probs = query_log_probs(model, obs, command, device)
    action_mask = action_mask_for_env(env)
    masked_log_probs = apply_action_mask(raw_log_probs, action_mask)
    probs = probabilities_from_log_probs(masked_log_probs)
    return {
        "action": int(np.argmax(masked_log_probs)),
        "masked_log_probs": masked_log_probs,
        "probs": probs,
        "action_mask": action_mask,
    }


def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, np.asarray(reward, dtype=np.float32), bool(terminated), bool(truncated), info
    obs, reward, done, info = out
    return obs, np.asarray(reward, dtype=np.float32), bool(done), False, info


def base_env(env):
    return getattr(env, "unwrapped", env)


def env_chain(env):
    seen = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "env", None)

    root = getattr(env, "unwrapped", None)
    if root is not None and id(root) not in seen:
        yield root


def env_identity_text(env):
    tokens = []
    for current in env_chain(env):
        tokens.append(type(current).__name__.lower())
        spec = getattr(current, "spec", None)
        spec_id = getattr(spec, "id", None)
        if spec_id:
            tokens.append(str(spec_id).lower())
    return " ".join(tokens)


def explicit_action_labels(env):
    identity = env_identity_text(env)

    if "deepseatreasurelegacyactionorder" in identity:
        # These are the exposed PCN action ids after the local legacy action-order wrapper.
        return {0: "up", 1: "right", 2: "down", 3: "left"}
    if "deepseatreasure" in identity or "deep-sea-treasure" in identity or "deepsea-treasure" in identity:
        return {0: "up", 1: "down", 2: "left", 3: "right"}
    if "resourcegathering" in identity or "resource-gathering" in identity:
        return {0: "up", 1: "down", 2: "left", 3: "right"}
    if "breakablebottles" in identity or "breakable-bottles" in identity:
        return {0: "move left", 1: "move right", 2: "pick up bottle"}
    if "fruit-tree" in identity or "fruittree" in identity:
        return {0: "left", 1: "right"}
    if "four-room" in identity or "fourroom" in identity:
        return {0: "left", 1: "up", 2: "right", 3: "down"}
    if "minecart" in identity:
        return {0: "Mine", 1: "Left", 2: "Right", 3: "Accelerate", 4: "Brake", 5: "None"}
    return None


def action_label(env, action):
    action = int(action)
    explicit_labels = explicit_action_labels(env)
    if explicit_labels is not None and action in explicit_labels:
        return explicit_labels[action]

    root = base_env(env)
    labels = getattr(root, "ACTION_LABELS", None)
    if labels is not None and 0 <= action < len(labels):
        return str(labels[action])

    action_to_delta = getattr(root, "ACTION_TO_DELTA", None)
    if action_to_delta is not None and action in action_to_delta:
        delta = tuple(np.asarray(action_to_delta[action], dtype=np.int64).tolist())
        direction_labels = {
            (0, 1): "right",
            (-1, 0): "up",
            (0, -1): "left",
            (1, 0): "down",
        }
        return direction_labels.get(delta, str(delta))

    return f"action {action}"


def format_action(env, action):
    return f"{int(action)}:{action_label(env, action)}"


def format_action_list(env, actions):
    return [format_action(env, action) for action in actions]


def format_action_probs(env, probs):
    return [
        f"{format_action(env, action)}={float(prob):.3f}"
        for action, prob in enumerate(np.asarray(probs, dtype=np.float32))
    ]


def timestep_info_text(env, timestep, remaining_return, policy, last_transition=None):
    action_mask = policy.get("action_mask")
    if action_mask is None:
        action_mask = np.ones(len(policy["probs"]), dtype=bool)
    valid_actions = set(int(action) for action in np.flatnonzero(action_mask))
    greedy = int(policy["action"])

    lines = [
        f"Timestep: {int(timestep)}",
        f"Remaining command R_t: {format_vector(remaining_return, precision=2)}",
        f"Greedy action: {format_action(env, greedy)}",
    ]
    if last_transition is not None:
        lines.extend(
            [
                "",
                f"Previous action: {format_action(env, last_transition['action'])}",
                f"Previous reward: {format_vector(last_transition['reward'], precision=2)}",
            ]
        )

    lines.extend(["", "Action probabilities:"])
    for action, prob in enumerate(policy["probs"]):
        marker = "*" if action == greedy else " "
        valid = "" if action in valid_actions else " invalid"
        lines.append(f"{marker} {format_action(env, action):<18} p={float(prob):.3f}{valid}")
    return "\n".join(lines)


class TimestepInfoWindow:

    def __init__(self):
        self.root = None
        self.text = None
        self.unavailable = False

    def _ensure_window(self):
        if self.root is not None or self.unavailable:
            return self.root is not None

        try:
            import tkinter as tk
        except Exception as exc:
            print(f"Could not open timestep info window: {exc}")
            self.unavailable = True
            return False

        try:
            self.root = tk.Tk()
            self.root.title("PCN timestep info")
            self.root.geometry("560x360+40+40")
            self.root.minsize(460, 260)
            try:
                self.root.attributes("-topmost", True)
            except tk.TclError:
                pass
            self.text = tk.Text(
                self.root,
                wrap="word",
                font=("Consolas", 13),
                bg="#111111",
                fg="#f2f2f2",
                insertbackground="#f2f2f2",
                padx=12,
                pady=12,
            )
            self.text.pack(fill="both", expand=True)
            self.text.configure(state="disabled")
            self.root.protocol("WM_DELETE_WINDOW", self.close)
            return True
        except Exception as exc:
            print(f"Could not open timestep info window: {exc}")
            self.root = None
            self.text = None
            self.unavailable = True
            return False

    def update(self, env, timestep, remaining_return, policy, last_transition=None):
        if not self._ensure_window():
            return

        content = timestep_info_text(
            env,
            timestep,
            remaining_return,
            policy,
            last_transition=last_transition,
        )
        try:
            self.text.configure(state="normal")
            self.text.delete("1.0", "end")
            self.text.insert("1.0", content)
            self.text.configure(state="disabled")
            self.root.lift()
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self.close()

    def close(self):
        if self.root is None:
            return
        try:
            self.root.destroy()
        except Exception:
            pass
        self.root = None
        self.text = None


def render_env(env):
    render_fn = getattr(env, "render", None)
    if callable(render_fn):
        render_fn()


def set_render_mode_temporarily(env, render_mode):
    states = []
    seen = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "render_mode"):
            previous_render_mode = current.render_mode
            try:
                current.render_mode = render_mode
            except AttributeError:
                pass
            else:
                states.append((current, previous_render_mode))
        current = getattr(current, "env", None)

    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and id(unwrapped) not in seen and hasattr(unwrapped, "render_mode"):
        previous_render_mode = unwrapped.render_mode
        try:
            unwrapped.render_mode = render_mode
        except AttributeError:
            pass
        else:
            states.append((unwrapped, previous_render_mode))
    return states


def restore_render_modes(states):
    for target, render_mode in reversed(states):
        target.render_mode = render_mode


def rollout_to_timestep(env, model, desired_return, timestep, device, render=False):
    render_states = set_render_mode_temporarily(env, None) if render else []
    try:
        obs, info = reset_env(env)
        remaining_return = np.asarray(desired_return, dtype=np.float32).copy()
        trajectory = []

        for step_index in range(int(timestep)):
            policy = greedy_action(model, env, obs, remaining_return, device)
            next_obs, reward, terminated, truncated, info = step_env(env, policy["action"])
            trajectory.append(
                {
                    "step": step_index,
                    "obs": to_jsonable(np.asarray(obs)),
                    "remaining_return": to_jsonable(remaining_return),
                    "action": int(policy["action"]),
                    "reward": to_jsonable(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": to_jsonable(info),
                }
            )
            remaining_return = (remaining_return - reward).astype(np.float32)
            obs = next_obs
            if terminated or truncated:
                raise RuntimeError(
                    f"episode ended after step {step_index + 1}; cannot explain timestep {timestep}"
                )
    finally:
        restore_render_modes(render_states)

    if render:
        render_env(env)

    return {
        "obs": obs,
        "remaining_return": remaining_return,
        "info": info,
        "trajectory": trajectory,
    }


def timestep_selection_from_rollout(env, model, rollout, timestep, device):
    obs = rollout["obs"]
    remaining_return = rollout["remaining_return"]
    original_policy = greedy_action(model, env, obs, remaining_return, device)
    action_mask = original_policy["action_mask"]
    if action_mask is None:
        action_mask = np.ones(len(original_policy["masked_log_probs"]), dtype=bool)
    valid_actions = np.flatnonzero(action_mask)
    return {
        "timestep": int(timestep),
        "rollout": rollout,
        "original_policy": original_policy,
        "action_mask": action_mask,
        "valid_actions": valid_actions,
    }


def choose_timestep_manually(env, model, desired_return, device):
    while True:
        timestep = ask_int("\nTimestep to explain", default=0, minimum=0)
        try:
            rollout = rollout_to_timestep(env, model, desired_return, timestep, device)
            return timestep_selection_from_rollout(env, model, rollout, timestep, device)
        except RuntimeError as exc:
            print(exc)


def choose_timestep_with_render(context, desired_return):
    metadata = context["metadata"]
    setup = context["setup"]
    model = context["model"]
    device = context["device"]
    fruit_tree_depth = int(metadata.get("fruit_tree_depth", 6))

    try:
        render_setup = build_experiment_setup(
            setup.env_name,
            device=device,
            fruit_tree_depth=fruit_tree_depth,
            include_model=False,
            render_mode="human",
        )
    except Exception as exc:
        print(f"\nCould not open an environment render window: {exc}")
        return choose_timestep_manually(context["env"], model, desired_return, device)

    render_env_instance = render_setup.env
    info_window = TimestepInfoWindow()
    render_env_closed = False

    def close_render_env():
        nonlocal render_env_closed
        info_window.close()
        if not render_env_closed:
            render_env_instance.close()
            render_env_closed = True

    timestep = 0
    print("\nRendered timestep browser")
    print("  n/Enter: next policy step")
    print("  p: previous policy step")
    print("  s: select current timestep")
    print("  number: jump to timestep")
    print("  q: use manual CLI timestep prompt")

    try:
        while True:
            try:
                rollout = rollout_to_timestep(
                    render_env_instance,
                    model,
                    desired_return,
                    timestep,
                    device,
                    render=True,
                )
                selection = timestep_selection_from_rollout(
                    render_env_instance,
                    model,
                    rollout,
                    timestep,
                    device,
                )
            except RuntimeError as exc:
                print(exc)
                timestep = max(0, timestep - 1)
                continue
            except Exception as exc:
                print(f"\nRendering failed: {exc}")
                return choose_timestep_manually(context["env"], model, desired_return, device)

            info_window.update(
                render_env_instance,
                timestep,
                rollout["remaining_return"],
                selection["original_policy"],
                last_transition=rollout["trajectory"][-1] if rollout["trajectory"] else None,
            )
            print(
                f"\nTimestep {timestep} | remaining command {format_vector(rollout['remaining_return'])} "
                f"| greedy action {format_action(render_env_instance, selection['original_policy']['action'])} "
                f"| valid actions {format_action_list(render_env_instance, selection['valid_actions'])}"
            )
            print("Action probabilities: " + ", ".join(format_action_probs(render_env_instance, selection["original_policy"]["probs"])))
            raw = input("Step [n/p/s/q or timestep]: ").strip().lower()
            if raw in {"", "n", "next"}:
                timestep += 1
            elif raw in {"p", "prev", "previous", "back"}:
                timestep = max(0, timestep - 1)
            elif raw in {"s", "select", "choose"}:
                close_render_env()
                return selection
            elif raw in {"q", "quit", "manual"}:
                close_render_env()
                return choose_timestep_manually(context["env"], model, desired_return, device)
            else:
                try:
                    next_timestep = int(raw)
                except ValueError:
                    print("Choose n, p, s, q, or an integer timestep.")
                    continue
                if next_timestep < 0:
                    print("Timestep must be non-negative.")
                    continue
                timestep = next_timestep
    finally:
        close_render_env()


def command_bounds(setup):
    low = np.minimum(np.asarray(setup.ref_point, dtype=np.float32), np.asarray(setup.max_return, dtype=np.float32))
    high = np.maximum(np.asarray(setup.ref_point, dtype=np.float32), np.asarray(setup.max_return, dtype=np.float32))
    return low, high


def commands_from_logged_front(front, desired_return, remaining_return, bounds):
    if front is None or len(front) == 0:
        return None
    collected_reward = (
        np.asarray(desired_return, dtype=np.float32)
        - np.asarray(remaining_return, dtype=np.float32)
    )
    commands = np.asarray(front, dtype=np.float32) - collected_reward
    commands = commands[np.isfinite(commands).all(axis=1)]
    if len(commands) == 0:
        return None
    low, high = bounds
    commands = np.clip(commands, low, high).astype(np.float32)
    return unique_commands(commands)


def unique_commands(commands, decimals=6):
    commands = np.asarray(commands, dtype=np.float32)
    if commands.size == 0:
        return commands.reshape(0, 0)
    if commands.ndim == 1:
        commands = commands.reshape(1, -1)
    commands = commands[np.isfinite(commands).all(axis=1)]
    if len(commands) == 0:
        return commands
    return np.unique(np.round(commands, decimals=decimals), axis=0).astype(np.float32)


def command_scale(bounds):
    low, high = bounds
    scale = np.asarray(high, dtype=np.float32) - np.asarray(low, dtype=np.float32)
    scale[~np.isfinite(scale) | (np.abs(scale) < 1e-6)] = 1.0
    return scale


def scaled_command_distance(command, base_command, scale):
    delta = (np.asarray(command, dtype=np.float32) - np.asarray(base_command, dtype=np.float32)) / scale
    return float(np.linalg.norm(delta, ord=2))


def command_in_bounds(command, bounds, atol=1e-6):
    command = np.asarray(command, dtype=np.float32)
    low, high = bounds
    return bool(np.all(command >= np.asarray(low, dtype=np.float32) - atol) and np.all(command <= np.asarray(high, dtype=np.float32) + atol))


def better_success_scaled(candidate, incumbent, base_command, scale):
    if candidate is None or not candidate.success:
        return incumbent
    if incumbent is None:
        return candidate
    candidate_distance = scaled_command_distance(candidate.command, base_command, scale)
    incumbent_distance = scaled_command_distance(incumbent.command, base_command, scale)
    if candidate_distance < incumbent_distance - 1e-12:
        return candidate
    if abs(candidate_distance - incumbent_distance) <= 1e-12 and candidate.l2_norm < incumbent.l2_norm:
        return candidate
    return incumbent


def infer_directional_bounds(base_command, front_results, bounds):
    low, high = (np.asarray(bounds[0], dtype=np.float32).copy(), np.asarray(bounds[1], dtype=np.float32).copy())
    base_command = np.asarray(base_command, dtype=np.float32)
    anchor = None
    for result in front_results:
        if np.allclose(result.command, base_command, atol=1e-6):
            continue
        if anchor is None or result.target_margin > anchor.target_margin:
            anchor = result
    direction = np.zeros_like(base_command, dtype=np.float32)
    if anchor is None:
        return (low, high), direction

    diff = np.asarray(anchor.command, dtype=np.float32) - base_command
    for index, value in enumerate(diff):
        if value > 1e-6:
            low[index] = max(low[index], base_command[index])
            direction[index] = 1.0
        elif value < -1e-6:
            high[index] = min(high[index], base_command[index])
            direction[index] = -1.0
    low = np.minimum(low, high).astype(np.float32)
    return (low, high), direction


def direction_constraint_summary(direction, base_command):
    parts = []
    for index, value in enumerate(direction):
        if value > 0:
            parts.append(f"R[{index}] >= {base_command[index]:.4f}")
        elif value < 0:
            parts.append(f"R[{index}] <= {base_command[index]:.4f}")
    return ", ".join(parts) if parts else "box bounds only"


def normalize_direction(vector, eps=1e-12):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector, ord=2))
    if not np.isfinite(norm) or norm <= eps:
        return None
    return (vector / norm).astype(np.float32)


def unique_directions(directions, decimals=6):
    unique = []
    seen = set()
    for direction in directions:
        normalized = normalize_direction(direction)
        if normalized is None:
            continue
        key = tuple(np.round(normalized, decimals=decimals).tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def project_to_direction_cone(vector, direction, low_z, high_z):
    projected = np.asarray(vector, dtype=np.float64).copy()
    direction = np.asarray(direction, dtype=np.float32)
    low_z = np.asarray(low_z, dtype=np.float64)
    high_z = np.asarray(high_z, dtype=np.float64)

    for index, sign in enumerate(direction):
        if sign > 0:
            projected[index] = abs(projected[index])
        elif sign < 0:
            projected[index] = -abs(projected[index])

        if projected[index] > 0.0 and high_z[index] <= 1e-9:
            projected[index] = 0.0
        elif projected[index] < 0.0 and low_z[index] >= -1e-9:
            projected[index] = 0.0

    return projected


def generate_ray_directions(bounds, base_command, direction, front_results, settings, rng, scale):
    low, high = bounds
    base_command = np.asarray(base_command, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    low_z = (np.asarray(low, dtype=np.float32) - base_command) / scale
    high_z = (np.asarray(high, dtype=np.float32) - base_command) / scale
    dim = int(len(base_command))
    directions = []

    for index in range(dim):
        if direction[index] >= 0 and high_z[index] > 1e-9:
            vector = np.zeros(dim, dtype=np.float32)
            vector[index] = 1.0
            directions.append(vector)
        if direction[index] <= 0 and low_z[index] < -1e-9:
            vector = np.zeros(dim, dtype=np.float32)
            vector[index] = -1.0
            directions.append(vector)

    sorted_front = sorted(front_results, key=lambda result: result.target_margin, reverse=True)
    for result in sorted_front:
        front_direction = (np.asarray(result.command, dtype=np.float32) - base_command) / scale
        directions.append(project_to_direction_cone(front_direction, direction, low_z, high_z))

    for _ in range(max(0, int(settings.random_directions))):
        random_direction = rng.normal(size=dim)
        directions.append(project_to_direction_cone(random_direction, direction, low_z, high_z))

    return unique_directions(directions)


def ray_endpoint_from_scaled_direction(base_command, unit_z, bounds, scale):
    low, high = bounds
    base_command = np.asarray(base_command, dtype=np.float32)
    unit_z = np.asarray(unit_z, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    low_z = (np.asarray(low, dtype=np.float32) - base_command) / scale
    high_z = (np.asarray(high, dtype=np.float32) - base_command) / scale
    max_radius = np.inf

    for value, low_value, high_value in zip(unit_z, low_z, high_z):
        if abs(float(value)) <= 1e-12:
            continue
        boundary = high_value if value > 0 else low_value
        radius = float(boundary / value)
        if radius <= 1e-9 or not np.isfinite(radius):
            return None
        max_radius = min(max_radius, radius)

    if not np.isfinite(max_radius) or max_radius <= 1e-9:
        return None

    command = base_command + scale * unit_z * np.float32(max_radius)
    command = np.clip(command, low, high).astype(np.float32)
    return command, float(max_radius)


def refine_success_along_ray(
    model,
    obs,
    base_command,
    successful_delta,
    successful_result,
    a_star,
    a_foil,
    action_mask,
    settings,
    device,
    start_queries,
    clip_bounds,
    max_queries,
):
    lo = 0.0
    hi = 1.0
    queries = int(start_queries)
    trace = []
    best = successful_result
    target_delta = np.asarray(successful_delta, dtype=np.float32)
    max_queries = max(0, int(max_queries))

    for _ in range(max_queries):
        mid = (lo + hi) / 2.0
        candidate = evaluate_delta(
            model,
            obs,
            base_command,
            target_delta * np.float32(mid),
            a_star,
            a_foil,
            action_mask,
            settings,
            device,
            clip_bounds=clip_bounds,
        )
        queries += 1
        trace.append(trace_entry(queries, candidate))
        if candidate.success:
            best = candidate
            hi = mid
        else:
            lo = mid

    return best, trace, queries


def evaluate_delta(
    model,
    obs,
    base_command,
    delta,
    a_star,
    a_foil,
    action_mask,
    settings,
    device,
    clip_bounds=None,
):
    delta = np.asarray(delta, dtype=np.float32)
    command = np.asarray(base_command, dtype=np.float32) + delta
    if settings.clip_command and clip_bounds is not None:
        command = np.clip(command, clip_bounds[0], clip_bounds[1]).astype(np.float32)
        delta = (command - base_command).astype(np.float32)

    raw_log_probs = query_log_probs(model, obs, command, device)
    masked_log_probs = apply_action_mask(raw_log_probs, action_mask)
    probs = probabilities_from_log_probs(masked_log_probs)

    if action_mask is None:
        valid_actions = np.arange(len(masked_log_probs), dtype=np.int64)
    else:
        valid_actions = np.flatnonzero(action_mask).astype(np.int64)
    other_actions = valid_actions[valid_actions != int(a_foil)]
    if len(other_actions) == 0:
        raise ValueError("targeted ZOO loss requires at least one valid non-foil action")
    best_other_action = int(other_actions[np.argmax(masked_log_probs[other_actions])])

    target_margin = float(masked_log_probs[a_foil] - masked_log_probs[best_other_action])
    hinge = float(max(masked_log_probs[best_other_action] - masked_log_probs[a_foil] + settings.margin, 0.0))
    # Penalize perturbation size in normalized command space so each reward
    # dimension is measured relative to its feasible range.
    if clip_bounds is None:
        scale = np.ones_like(delta, dtype=np.float32)
    else:
        scale = command_scale(clip_bounds)
    scaled_delta = delta / scale
    l2_sq = float(np.dot(scaled_delta, scaled_delta))
    objective = float(l2_sq + settings.c * hinge)
    l2_norm = float(np.linalg.norm(delta, ord=2))

    return CFQueryResult(
        delta=delta.astype(np.float32),
        command=command.astype(np.float32),
        masked_log_probs=masked_log_probs,
        probs=probs,
        objective=objective,
        hinge=hinge,
        target_margin=target_margin,
        success=bool(target_margin >= settings.margin),
        greedy_action=int(np.argmax(masked_log_probs)),
        l2_norm=l2_norm,
    )


def trace_entry(
    query_count,
    result,
):
    return {
        "query": int(query_count),
        "objective": float(result.objective),
    }


def rebase_trace_queries(trace, query_offset):
    rebased = []
    for entry in trace:
        copied = dict(entry)
        copied["query"] = int(copied["query"]) + int(query_offset)
        rebased.append(copied)
    return rebased


def better_success(candidate, incumbent):
    if candidate is None or not candidate.success:
        return incumbent
    if incumbent is None:
        return candidate
    if candidate.l2_norm < incumbent.l2_norm - 1e-12:
        return candidate
    if abs(candidate.l2_norm - incumbent.l2_norm) <= 1e-12 and candidate.objective < incumbent.objective:
        return candidate
    return incumbent


def better_objective(candidate, incumbent):
    if candidate is None:
        return incumbent
    if incumbent is None or candidate.objective < incumbent.objective - 1e-12:
        return candidate
    if abs(candidate.objective - incumbent.objective) <= 1e-12 and candidate.l2_norm < incumbent.l2_norm:
        return candidate
    return incumbent


def better_failed_margin(candidate, incumbent):
    if candidate is None or candidate.success:
        return incumbent
    if incumbent is None:
        return candidate
    if candidate.target_margin > incumbent.target_margin + 1e-12:
        return candidate
    if abs(candidate.target_margin - incumbent.target_margin) <= 1e-12 and candidate.l2_norm < incumbent.l2_norm:
        return candidate
    return incumbent


ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8

def zoo_coordinate_search(
    model,
    obs,
    base_command,
    a_star,
    a_foil,
    action_mask,
    settings,
    device,
    clip_bounds=None,
    initial_delta=None,
):
    rng = np.random.default_rng(settings.seed)
    dim = int(len(base_command))
    base_command = np.asarray(base_command, dtype=np.float32)
    if clip_bounds is None:
        scale = np.ones(dim, dtype=np.float32)
    else:
        scale = command_scale(clip_bounds)
    queries = 0
    if initial_delta is None:
        current_delta = np.zeros(dim, dtype=np.float32)
    else:
        current_delta = np.asarray(initial_delta, dtype=np.float32).copy()
        if current_delta.shape != base_command.shape:
            raise ValueError("initial_delta must have the same shape as base_command")
    trace = []
    best_success = None
    best_objective = None

    current = evaluate_delta(
        model,
        obs,
        base_command,
        current_delta,
        a_star,
        a_foil,
        action_mask,
        settings,
        device,
        clip_bounds=clip_bounds,
    )
    queries += 1
    trace.append(trace_entry(queries, current))
    current_delta = current.delta.copy()
    current_z = (current_delta / scale).astype(np.float32)
    best_success = better_success_scaled(current, best_success, base_command, scale)
    best_objective = better_objective(current, best_objective)

    adam_m = np.zeros(dim, dtype=np.float64)
    adam_v = np.zeros(dim, dtype=np.float64)
    adam_t = np.zeros(dim, dtype=np.int64)

    while queries + 3 <= int(settings.max_queries):
        coordinate = int(rng.integers(dim))
        finite_diff_h = float(settings.finite_diff_h)

        plus_z = current_z.copy()
        plus_z[coordinate] += np.float32(finite_diff_h)
        plus_delta = plus_z * scale
        plus = evaluate_delta(
            model,
            obs,
            base_command,
            plus_delta,
            a_star,
            a_foil,
            action_mask,
            settings,
            device,
            clip_bounds=clip_bounds,
        )
        queries += 1
        trace.append(trace_entry(queries, plus))
        best_success = better_success_scaled(plus, best_success, base_command, scale)
        best_objective = better_objective(plus, best_objective)

        minus_z = current_z.copy()
        minus_z[coordinate] -= np.float32(finite_diff_h)
        minus_delta = minus_z * scale
        minus = evaluate_delta(
            model,
            obs,
            base_command,
            minus_delta,
            a_star,
            a_foil,
            action_mask,
            settings,
            device,
            clip_bounds=clip_bounds,
        )
        queries += 1
        trace.append(trace_entry(queries, minus))
        best_success = better_success_scaled(minus, best_success, base_command, scale)
        best_objective = better_objective(minus, best_objective)

        gradient = float((plus.objective - minus.objective) / (2.0 * finite_diff_h))
        adam_t[coordinate] += 1
        adam_m[coordinate] = (
            ADAM_BETA1 * adam_m[coordinate]
            + (1.0 - ADAM_BETA1) * gradient
        )
        adam_v[coordinate] = (
            ADAM_BETA2 * adam_v[coordinate]
            + (1.0 - ADAM_BETA2) * (gradient ** 2)
        )
        m_hat = adam_m[coordinate] / (1.0 - ADAM_BETA1 ** int(adam_t[coordinate]))
        v_hat = adam_v[coordinate] / (1.0 - ADAM_BETA2 ** int(adam_t[coordinate]))
        update = -float(settings.learning_rate) * m_hat / (np.sqrt(v_hat) + ADAM_EPSILON)

        candidate_z = current_z.copy()
        candidate_z[coordinate] += np.float32(update)
        candidate_delta = candidate_z * scale
        candidate = evaluate_delta(
            model,
            obs,
            base_command,
            candidate_delta,
            a_star,
            a_foil,
            action_mask,
            settings,
            device,
            clip_bounds=clip_bounds,
        )
        queries += 1
        trace.append(trace_entry(queries, candidate))
        current = candidate
        current_delta = candidate.delta.copy()
        current_z = (current_delta / scale).astype(np.float32)
        best_success = better_success_scaled(current, best_success, base_command, scale)
        best_objective = better_objective(current, best_objective)

    return {
        "current": current,
        "best_success": best_success,
        "best_objective": best_objective,
        "trace": trace,
        "queries": int(queries),
    }


def binary_refine_success(
    model,
    obs,
    base_command,
    successful_delta,
    a_star,
    a_foil,
    action_mask,
    settings,
    device,
    start_queries,
    clip_bounds=None,
    max_binary_queries=40,
):
    target_delta = np.asarray(successful_delta, dtype=np.float32)
    lo = 0.0
    hi = 1.0
    queries = int(start_queries)
    trace = []

    best = evaluate_delta(
        model,
        obs,
        base_command,
        target_delta,
        a_star,
        a_foil,
        action_mask,
        settings,
        device,
        clip_bounds=clip_bounds,
    )
    queries += 1
    trace.append(trace_entry(queries, best))
    if not best.success:
        return best, trace, queries

    for _ in range(int(max_binary_queries) - 1):
        mid = (lo + hi) / 2.0
        candidate = evaluate_delta(
            model,
            obs,
            base_command,
            target_delta * np.float32(mid),
            a_star,
            a_foil,
            action_mask,
            settings,
            device,
            clip_bounds=clip_bounds,
        )
        queries += 1
        trace.append(trace_entry(queries, candidate))
        if candidate.success:
            best = candidate
            hi = mid
        else:
            lo = mid

    return best, trace, queries


def constrained_nearest_boundary_search(
    model,
    obs,
    base_command,
    desired_return,
    a_star,
    a_foil,
    action_mask,
    settings,
    device,
    bounds,
    front=None,
):
    rng = np.random.default_rng(settings.seed)
    base_command = np.asarray(base_command, dtype=np.float32)
    bounds = (
        np.asarray(bounds[0], dtype=np.float32),
        np.asarray(bounds[1], dtype=np.float32),
    )
    scale = command_scale(bounds)
    candidate_budget = max(1, int(settings.max_queries) - int(settings.line_search_steps) - 1)
    queries = 0
    trace = []
    best_success = None
    best_objective = None
    best_failure = None
    front_results = []

    current = evaluate_delta(
        model,
        obs,
        base_command,
        np.zeros_like(base_command, dtype=np.float32),
        a_star,
        a_foil,
        action_mask,
        settings,
        device,
        clip_bounds=bounds,
    )
    queries += 1
    trace.append(trace_entry(queries, current))
    best_success = better_success_scaled(current, best_success, base_command, scale)
    best_objective = better_objective(current, best_objective)
    best_failure = better_failed_margin(current, best_failure)

    front_commands = commands_from_logged_front(front, desired_return, base_command, bounds)
    if front_commands is not None:
        for command in front_commands:
            if queries >= candidate_budget:
                break
            result = evaluate_delta(
                model,
                obs,
                base_command,
                np.asarray(command, dtype=np.float32) - base_command,
                a_star,
                a_foil,
                action_mask,
                settings,
                device,
                clip_bounds=bounds,
            )
            queries += 1
            trace.append(trace_entry(queries, result))
            front_results.append(result)
            best_success = better_success_scaled(result, best_success, base_command, scale)
            best_objective = better_objective(result, best_objective)
            best_failure = better_failed_margin(result, best_failure)

    constrained_bounds, direction = infer_directional_bounds(base_command, front_results, bounds)
    constrained_scale = command_scale(constrained_bounds)
    bounded_seed_candidates = [current, *front_results]
    best_success = None
    for result in bounded_seed_candidates:
        if command_in_bounds(result.command, constrained_bounds):
            best_success = better_success_scaled(result, best_success, base_command, constrained_scale)

    ray_directions = generate_ray_directions(
        constrained_bounds,
        base_command,
        direction,
        front_results,
        settings,
        rng,
        constrained_scale,
    )

    for unit_z in ray_directions:
        if queries >= candidate_budget:
            break
        endpoint = ray_endpoint_from_scaled_direction(
            base_command,
            unit_z,
            constrained_bounds,
            constrained_scale,
        )
        if endpoint is None:
            continue
        command, _ = endpoint
        result = evaluate_delta(
            model,
            obs,
            base_command,
            np.asarray(command, dtype=np.float32) - base_command,
            a_star,
            a_foil,
            action_mask,
            settings,
            device,
            clip_bounds=constrained_bounds,
        )
        queries += 1
        trace.append(trace_entry(queries, result))
        best_success = better_success_scaled(result, best_success, base_command, constrained_scale)
        best_objective = better_objective(result, best_objective)
        best_failure = better_failed_margin(result, best_failure)

        if result.success and queries < candidate_budget:
            remaining_budget = min(
                int(settings.line_search_steps),
                max(0, int(candidate_budget) - int(queries)),
            )
            refined, refine_trace, queries = refine_success_along_ray(
                model,
                obs,
                base_command,
                result.delta,
                result,
                a_star,
                a_foil,
                action_mask,
                settings,
                device,
                start_queries=queries,
                clip_bounds=constrained_bounds,
                max_queries=remaining_budget,
            )
            trace.extend(refine_trace)
            best_success = better_success_scaled(refined, best_success, base_command, constrained_scale)
            best_objective = better_objective(refined, best_objective)
            best_failure = better_failed_margin(refined, best_failure)

    return {
        "best_success": best_success,
        "best_objective": best_objective,
        "best_failure": best_failure,
        "trace": trace,
        "queries": int(queries),
        "bounds": constrained_bounds,
        "direction": direction,
    }


def observation_summary(obs):
    arr = np.asarray(obs)
    summary = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }
    if arr.size == 0:
        summary["value"] = []
    elif arr.size <= 20:
        summary["value"] = arr.tolist()
    else:
        flat = arr.astype(np.float32, copy=False).reshape(-1)
        summary.update(
            {
                "min": float(np.min(flat)),
                "max": float(np.max(flat)),
                "mean": float(np.mean(flat)),
                "first_values": flat[:10].tolist(),
            }
        )
    return summary


def print_vector_rows(rows):
    for index, row in enumerate(rows):
        print(f"  [{index:>3}] {format_vector(row)}")


def choose_desired_return(run_dir, setup):
    reward_dim = int(np.asarray(setup.max_return).shape[-1])
    print("\nCommand source:")
    print("  1. Logged Pareto front")
    print("  2. Manual desired return")
    source = ask_choice("Choose command source", ["1", "2"], default="1")

    if source == "1":
        log_path = Path(run_dir) / "log.h5"
        if not log_path.is_file():
            print(f"No log.h5 found at {log_path}; falling back to manual input.")
        else:
            try:
                with h5py.File(log_path, "r") as log:
                    front = load_logged_front(log, list(range(reward_dim)))
                if len(front) == 0:
                    print("Logged front is empty; falling back to manual input.")
                else:
                    print("\nLogged front rows:")
                    print_vector_rows(front)
                    index = ask_int("Policy/front index", minimum=0, maximum=len(front) - 1)
                    return front[index].astype(np.float32), {
                        "source": "logged_front",
                        "front_index": int(index),
                        "front": front.astype(np.float32),
                    }
            except (KeyError, OSError, ValueError) as exc:
                print(f"Could not load logged front: {exc}")
                print("Falling back to manual input.")

    while True:
        raw = input(f"Desired return vector ({reward_dim} comma-separated values): ").strip()
        try:
            return parse_vector(raw, expected_dim=reward_dim), {"source": "manual"}
        except ValueError as exc:
            print(exc)


def choose_zoo_settings():
    values = dict(DEFAULT_ZOO_SETTINGS)
    return ZooSettings(**values)


def print_selected_state(
    env,
    desired_return,
    timestep,
    obs,
    remaining_return,
    valid_actions,
    original_policy,
):
    print("\nSelected state")
    print(f"  selected desired return: {format_vector(desired_return)}")
    print(f"  timestep: {timestep}")
    print(f"  remaining command R_t: {format_vector(remaining_return)}")
    print(f"  observation summary: {json.dumps(observation_summary(obs), sort_keys=True)}")
    print(f"  valid actions: {format_action_list(env, valid_actions)}")
    print(f"  original greedy action: {format_action(env, original_policy['action'])}")
    print(f"  original masked log-probs: {format_vector(original_policy['masked_log_probs'])}")
    print(f"  original action probabilities: {', '.join(format_action_probs(env, original_policy['probs']))}")


def plot_action_probs_before_after(path, env, original_probs, counterfactual_probs):
    fig, ax = plt.subplots(figsize=(7, 4))
    indexes = np.arange(len(original_probs))
    width = 0.38
    ax.bar(indexes - width / 2, original_probs, width=width, label="Original")
    ax.bar(indexes + width / 2, counterfactual_probs, width=width, label="Counterfactual")
    ax.set_xticks(indexes)
    ax.set_xticklabels([format_action(env, action) for action in indexes], rotation=25, ha="right")
    ax.set_xlabel("Action")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, max(1.0, float(max(np.max(original_probs), np.max(counterfactual_probs))) * 1.1))
    ax.set_title("Action Probabilities Before and After")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_paper_figure(fig, path)
    plt.close(fig)


def plot_trace(path, trace, key, title, ylabel):
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [entry["query"] for entry in trace]
    ys = [entry[key] for entry in trace]
    ax.plot(xs, ys, linewidth=1.5)
    ax.set_xlabel("Model Queries")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_paper_figure(fig, path)
    plt.close(fig)


def plot_results(output_dir, env, original, counterfactual, trace):
    plot_action_probs_before_after(
        output_dir / "action_probs_before_after.png",
        env,
        original.probs,
        counterfactual.probs,
    )
    plot_trace(output_dir / "zoo_objective_trace.png", trace, "objective", "ZOO Objective Trace", "Objective")


def save_results(
    output_dir,
    env,
    env_name,
    timestep,
    original,
    counterfactual,
    a_star,
    a_foil,
    trace,
    queries,
):
    output_dir.mkdir(parents=True, exist_ok=False)

    summary = {
        "env": env_name,
        "timestep": int(timestep),
        "a_star": int(a_star),
        "a_star_label": action_label(env, a_star),
        "a_foil": int(a_foil),
        "a_foil_label": action_label(env, a_foil),
        "success": bool(counterfactual.success),
        "foil_is_greedy_action": bool(counterfactual.greedy_action == a_foil),
        "queries": int(queries),
        "l2_norm": float(counterfactual.l2_norm),
        "original": {
            "command": to_jsonable(original.command),
            "greedy_action": int(a_star),
            "greedy_action_label": action_label(env, a_star),
            "probs": to_jsonable(original.probs),
            "target_margin": float(original.target_margin),
        },
        "counterfactual": {
            "command": to_jsonable(counterfactual.command),
            "delta": to_jsonable(counterfactual.delta),
            "greedy_action": int(counterfactual.greedy_action),
            "greedy_action_label": action_label(env, counterfactual.greedy_action),
            "probs": to_jsonable(counterfactual.probs),
            "target_margin": float(counterfactual.target_margin),
        },
    }

    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")

    plot_results(output_dir, env, original, counterfactual, trace)


def print_final_result(env, final_result, a_foil):
    print("\nCounterfactual result")
    print(f"  counterfactual desired-return command: {format_vector(final_result.command)}")
    print(f"  desired-return command change delta_R: {format_vector(final_result.delta)}")
    print(f"  chosen action after change: {format_action(env, final_result.greedy_action)}")
    print(f"  foil action: {format_action(env, a_foil)}")
    print(f"  success: {final_result.success}")


def main():
    print("=" * 72)
    print("PCN ZOO Command-Space Counterfactual Explainer")
    print("=" * 72)

    context = load_run_and_model()
    run_dir = context["run_dir"]
    setup = context["setup"]
    env = context["env"]
    model = context["model"]
    device = context["device"]

    try:
        desired_return, command_source = choose_desired_return(run_dir, setup)

        selection = choose_timestep_with_render(context, desired_return)
        timestep = selection["timestep"]
        rollout = selection["rollout"]
        obs_t = rollout["obs"]
        remaining_return_t = rollout["remaining_return"]
        original_policy = selection["original_policy"]
        action_mask = selection["action_mask"]
        valid_actions = selection["valid_actions"]
        a_star = int(original_policy["action"])

        print_selected_state(
            env,
            desired_return,
            timestep,
            obs_t,
            remaining_return_t,
            valid_actions,
            original_policy,
        )

        foil_actions = [int(action) for action in valid_actions if int(action) != a_star]
        if not foil_actions:
            raise RuntimeError("No valid foil actions are available at this state.")

        print(f"\nOriginal greedy action: {format_action(env, a_star)}")
        while True:
            print("\nFoil action choices:")
            for action in foil_actions:
                print(f"  {format_action(env, action)}")
            a_foil = ask_int("Foil action id", minimum=0, maximum=len(original_policy["masked_log_probs"]) - 1)
            if a_foil not in set(foil_actions):
                print(f"Choose one of the listed foil actions: {format_action_list(env, foil_actions)}")
                continue
            break

        settings = choose_zoo_settings()
        env_bounds = command_bounds(setup)
        clip_bounds = env_bounds if settings.clip_command else None

        original_result = evaluate_delta(
            model,
            obs_t,
            remaining_return_t,
            np.zeros_like(remaining_return_t, dtype=np.float32),
            a_star,
            a_foil,
            action_mask,
            settings,
            device,
            clip_bounds=clip_bounds,
        )

        front = command_source.get("front") if isinstance(command_source, dict) else None
        print("\nRunning Reward Perturbation Alg...")
        search_result = constrained_nearest_boundary_search(
            model,
            obs_t,
            remaining_return_t,
            desired_return,
            a_star,
            a_foil,
            action_mask,
            settings,
            device,
            env_bounds,
            front=front,
        )
        trace = list(search_result["trace"])
        queries = int(search_result["queries"])
        seed_result = search_result["best_success"]
        final_result = seed_result

        if seed_result is not None and seed_result.success:
            zoo_clip_bounds = search_result["bounds"] if settings.clip_command else None
            zoo_result = zoo_coordinate_search(
                model,
                obs_t,
                remaining_return_t,
                a_star,
                a_foil,
                action_mask,
                settings,
                device,
                clip_bounds=zoo_clip_bounds,
                initial_delta=seed_result.delta,
            )
            trace.extend(rebase_trace_queries(zoo_result["trace"], queries))
            queries += int(zoo_result["queries"])
            scale = command_scale(search_result["bounds"])
            final_result = better_success_scaled(zoo_result["best_success"], seed_result, remaining_return_t, scale)
        else:
            zoo_result = zoo_coordinate_search(
                model,
                obs_t,
                remaining_return_t,
                a_star,
                a_foil,
                action_mask,
                settings,
                device,
                clip_bounds=clip_bounds,
            )
            trace.extend(rebase_trace_queries(zoo_result["trace"], queries))
            queries += int(zoo_result["queries"])
            final_result = zoo_result["best_success"]

        if final_result is not None and final_result.success:
            refined, binary_trace, queries = binary_refine_success(
                model,
                obs_t,
                remaining_return_t,
                final_result.delta,
                a_star,
                a_foil,
                action_mask,
                settings,
                device,
                start_queries=queries,
                clip_bounds=search_result["bounds"] if seed_result is not None and settings.clip_command else clip_bounds,
                max_binary_queries=settings.line_search_steps,
            )
            trace.extend(binary_trace)
            if refined.success:
                final_result = refined
        else:
            final_result = (
                search_result["best_failure"]
                or search_result["best_objective"]
                or zoo_result["best_objective"]
            )

        print_final_result(env, final_result, a_foil)

        output_dir = (
            Path(run_dir)
            / "counterfactuals"
            / f"zoo_cf_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        save_results(
            output_dir,
            env,
            setup.env_name,
            timestep,
            original_result,
            final_result,
            a_star,
            a_foil,
            trace,
            queries,
        )
        print(f"\nSaved counterfactual artifacts to: {output_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
