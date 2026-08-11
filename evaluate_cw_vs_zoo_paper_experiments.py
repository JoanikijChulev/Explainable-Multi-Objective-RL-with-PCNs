import argparse
import csv
import json
import time
import traceback
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import evaluate_pcn_zoo_counterfactuals as zoo_batch
import interactive_pcn_cw as cw
import interactive_pcn_zoo_cf as cf
from device_utils import preferred_device
from paper_plot_utils import apply_bar_hatches, save_paper_figure


DEFAULT_BENCHMARK_RUNS = ["bp", "c2", "dst", "Minecart", "reward_line"]


DEFAULT_LANDSCAPE_CASES = [
    {"run": "bp", "front": 2, "timestep": 1, "foil": 3, "dims": [0, 1], "name": "bp_front2_t1_down"},
    {"run": "c2", "front": 1, "timestep": 0, "foil": 1, "dims": [0, 1], "name": "c2_front1_t0_up"},
    {"run": "dst", "front": 0, "timestep": 0, "foil": 1, "dims": [0, 1], "name": "dst_front0_t0_right"},
    {"run": "Minecart", "front": 0, "timestep": 0, "foil": 5, "dims": [1, 2], "name": "minecart_front0_t0_none"},
    {"run": "reward_line", "front": 0, "timestep": 0, "foil": 1, "dims": [0, 1], "name": "reward_line_front0_t0_up"},
]


def bool_text(value):
    return "true" if bool(value) else "false"


def maybe_float(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return value


def json_vector(value):
    if value is None:
        return ""
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return json.dumps([float(x) for x in arr])


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_zoo_settings(args):
    values = dict(cf.DEFAULT_ZOO_SETTINGS)
    values.update(
        {
            "max_queries": int(args.zoo_max_queries),
            "random_directions": int(args.zoo_random_directions) if args.zoo_random_directions is not None else int(args.zoo_max_queries),
            "line_search_steps": int(args.zoo_line_search_steps),
            "learning_rate": float(args.zoo_learning_rate),
            "finite_diff_h": float(args.zoo_finite_diff_h),
            "margin": float(args.success_margin),
            "c": float(args.zoo_c),
            "seed": int(args.seed),
            "clip_command": not bool(args.no_clip_command),
        }
    )
    return cf.ZooSettings(**values)


def make_cw_settings(args):
    settings = cw.choose_cw_settings()
    return replace(
        settings,
        confidence=float(args.cw_confidence),
        learning_rate=float(args.cw_learning_rate),
        binary_search_steps=int(args.cw_binary_search_steps),
        max_iter=int(args.cw_max_iter),
        initial_const=float(args.cw_initial_const),
        max_halving=int(args.cw_max_halving),
        max_doubling=int(args.cw_max_doubling),
        batch_size=int(args.cw_batch_size),
        verbose=bool(args.cw_verbose),
        art_clip_relaxation=float(args.cw_art_clip_relaxation),
        clip_command=not bool(args.no_clip_command),
    )


def make_eval_settings(success_margin, clip_command=True):
    values = dict(cf.DEFAULT_ZOO_SETTINGS)
    values["margin"] = float(success_margin)
    values["clip_command"] = bool(clip_command)
    return cf.ZooSettings(**values)


def action_name(env, action):
    return f"{int(action)}:{cf.action_label(env, int(action))}"


def load_run_by_name(run_name, checkpoint, device):
    run_dir = zoo_batch.resolve_run_dir(run_name)
    run_context = zoo_batch.load_run(run_dir, checkpoint, device)
    run_context["device"] = device
    return run_context


def close_run(run_context):
    env = run_context.get("env")
    if env is not None and hasattr(env, "close"):
        env.close()


def rollout_single_case(run_context, front_index, timestep, foil_action):
    front = run_context["front"]
    if front_index < 0 or front_index >= len(front):
        raise ValueError(f"front index {front_index} is outside 0..{len(front) - 1}")

    env = run_context["env"]
    model = run_context["model"]
    device = run_context["device"]
    desired_return = np.asarray(front[int(front_index)], dtype=np.float32)
    obs, _ = cf.reset_env(env)
    remaining_command = desired_return.copy()

    for current_timestep in range(int(timestep) + 1):
        policy = cf.greedy_action(model, env, obs, remaining_command, device)
        action_mask = policy["action_mask"]
        if action_mask is None:
            action_mask = np.ones(len(policy["masked_log_probs"]), dtype=bool)
        valid_actions = np.flatnonzero(action_mask).astype(np.int64)
        original_action = int(policy["action"])

        if current_timestep == int(timestep):
            if int(foil_action) not in set(int(a) for a in valid_actions):
                raise ValueError(
                    f"foil action {foil_action} is invalid at timestep {timestep}; "
                    f"valid actions are {[int(a) for a in valid_actions]}"
                )
            if int(foil_action) == original_action:
                raise ValueError(
                    f"foil action {foil_action} equals the original greedy action {original_action}"
                )
            return {
                "front_index": int(front_index),
                "desired_return": desired_return.copy(),
                "timestep": int(current_timestep),
                "obs": np.asarray(obs).copy(),
                "remaining_command": np.asarray(remaining_command, dtype=np.float32).copy(),
                "policy": {
                    "action": original_action,
                    "masked_log_probs": np.asarray(policy["masked_log_probs"], dtype=np.float32).copy(),
                    "probs": np.asarray(policy["probs"], dtype=np.float32).copy(),
                },
                "action_mask": np.asarray(action_mask, dtype=bool).copy(),
                "valid_actions": valid_actions.copy(),
                "foil_action": int(foil_action),
            }

        next_obs, reward, terminated, truncated, _ = cf.step_env(env, original_action)
        if terminated or truncated:
            raise ValueError(f"episode ended before timestep {timestep}")
        remaining_command = (remaining_command - reward).astype(np.float32)
        obs = next_obs

    raise RuntimeError("unreachable rollout state")


def choose_landscape_cases(args):
    cases = DEFAULT_LANDSCAPE_CASES[: int(args.landscape_limit)]
    if args.landscape_case_json:
        with Path(args.landscape_case_json).open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, list):
            raise ValueError("--landscape-case-json must contain a list of case specs")
        cases = loaded[: int(args.landscape_limit)]
    return cases


def eval_command(run_context, case, command, settings):
    base = np.asarray(case["remaining_command"], dtype=np.float32)
    command = np.asarray(command, dtype=np.float32)
    bounds = cf.command_bounds(run_context["setup"])
    return cf.evaluate_delta(
        run_context["model"],
        case["obs"],
        base,
        command - base,
        int(case["policy"]["action"]),
        int(case["foil_action"]),
        case["action_mask"],
        settings,
        run_context["device"],
        clip_bounds=bounds,
    )


def run_ours(run_context, case, zoo_settings):
    try:
        return zoo_batch.run_counterfactual_case(run_context, case, zoo_settings), ""
    except Exception:
        return None, traceback.format_exc(limit=4)


def run_cw_case(run_context, case, cw_settings, eval_settings):
    model = run_context["model"]
    device = run_context["device"]
    setup = run_context["setup"]
    base = np.asarray(case["remaining_command"], dtype=np.float32)
    bounds = cf.command_bounds(setup)
    start = time.perf_counter()
    try:
        command = cw.run_cw_attack(
            model,
            case["obs"],
            base,
            int(case["foil_action"]),
            case["action_mask"],
            cw_settings,
            device,
            bounds,
        )
        result = cf.evaluate_delta(
            model,
            case["obs"],
            base,
            np.asarray(command, dtype=np.float32) - base,
            int(case["policy"]["action"]),
            int(case["foil_action"]),
            case["action_mask"],
            eval_settings,
            device,
            clip_bounds=bounds,
        )
        return {
            "final_result": result,
            "duration_sec": time.perf_counter() - start,
            "error": "",
        }
    except Exception:
        return {
            "final_result": None,
            "duration_sec": time.perf_counter() - start,
            "error": traceback.format_exc(limit=4),
        }


def scaled_distance_for_result(run_context, case, result):
    if result is None:
        return None
    bounds = cf.command_bounds(run_context["setup"])
    scale = cf.command_scale(bounds)
    return cf.scaled_command_distance(result.command, case["remaining_command"], scale)


def known_front_success(run_context, case, eval_settings):
    bounds = cf.command_bounds(run_context["setup"])
    front_commands = cf.commands_from_logged_front(
        run_context["front"],
        case["desired_return"],
        case["remaining_command"],
        bounds,
    )
    if front_commands is None:
        return None
    best = None
    scale = cf.command_scale(bounds)
    for command in front_commands:
        result = eval_command(run_context, case, command, eval_settings)
        best = cf.better_success_scaled(result, best, case["remaining_command"], scale)
    return best


def command_margin_gradient(run_context, case, command):
    model = run_context["model"]
    device = run_context["device"]
    command_tensor = torch.as_tensor(np.asarray(command), dtype=torch.float32, device=device)
    command_tensor = command_tensor.detach().clone().requires_grad_(True)
    obs_tensor = torch.as_tensor(np.asarray(case["obs"])).unsqueeze(0).to(device)
    out = model(obs_tensor, command_tensor.unsqueeze(0))[0]
    mask = torch.as_tensor(case["action_mask"], dtype=torch.bool, device=device)
    masked = out.clone()
    masked[~mask] = -1.0e9
    foil = int(case["foil_action"])
    other_indexes = [int(a) for a in np.flatnonzero(case["action_mask"]) if int(a) != foil]
    other_tensor = torch.as_tensor(other_indexes, dtype=torch.long, device=device)
    best_other = masked[other_tensor].max()
    target_margin = masked[foil] - best_other
    target_margin.backward()
    grad = command_tensor.grad.detach().cpu().numpy().astype(np.float32)
    return grad


def heatmap_arrays(run_context, case, dims, grid_size, eval_settings):
    bounds = cf.command_bounds(run_context["setup"])
    low, high = bounds
    base = np.asarray(case["remaining_command"], dtype=np.float32)
    dim_x, dim_y = int(dims[0]), int(dims[1])
    x_values = np.linspace(float(low[dim_x]), float(high[dim_x]), int(grid_size), dtype=np.float32)
    y_values = np.linspace(float(low[dim_y]), float(high[dim_y]), int(grid_size), dtype=np.float32)
    margins = np.zeros((len(y_values), len(x_values)), dtype=np.float32)
    foil_probs = np.zeros_like(margins)
    greedy = np.zeros_like(margins, dtype=np.int32)

    for yi, y_value in enumerate(y_values):
        for xi, x_value in enumerate(x_values):
            command = base.copy()
            command[dim_x] = x_value
            command[dim_y] = y_value
            result = eval_command(run_context, case, command, eval_settings)
            margins[yi, xi] = float(result.target_margin)
            foil_probs[yi, xi] = float(result.probs[int(case["foil_action"])])
            greedy[yi, xi] = int(result.greedy_action)
    return x_values, y_values, margins, foil_probs, greedy


def plot_landscape(
    output_path,
    run_context,
    case,
    dims,
    x_values,
    y_values,
    margins,
    success_margin,
    ours_result,
    cw_result,
    front_result,
    title,
    include_gradient=True,
):
    dim_x, dim_y = int(dims[0]), int(dims[1])
    base = np.asarray(case["remaining_command"], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    extent = [float(x_values[0]), float(x_values[-1]), float(y_values[0]), float(y_values[-1])]
    vmax = max(abs(float(np.nanmin(margins))), abs(float(np.nanmax(margins))), abs(float(success_margin)), 1e-3)
    image = ax.imshow(
        margins,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    fig.colorbar(image, ax=ax, label="target margin: logp(foil) - max_other_logp")
    ax.contour(
        x_values,
        y_values,
        margins,
        levels=[float(success_margin)],
        colors="black",
        linewidths=1.6,
    )
    ax.scatter(base[dim_x], base[dim_y], s=72, c="white", edgecolors="black", label="original", zorder=5)

    markers = [
        ("ours", ours_result, "#00A6D6", "*", 150),
        ("ART C&W", cw_result, "#D55E00", "X", 90),
        ("logged-front success", front_result, "#009E73", "D", 72),
    ]
    for label, result, color, marker, size in markers:
        if result is None:
            continue
        ax.scatter(
            result.command[dim_x],
            result.command[dim_y],
            s=size,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.75,
            label=label,
            zorder=6,
        )

    path_target = None
    if ours_result is not None and ours_result.success:
        path_target = ours_result.command
    elif front_result is not None and front_result.success:
        path_target = front_result.command
    elif cw_result is not None and cw_result.success:
        path_target = cw_result.command
    if path_target is not None:
        ax.plot(
            [base[dim_x], path_target[dim_x]],
            [base[dim_y], path_target[dim_y]],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="nearest-boundary movement",
            zorder=4,
        )

    if include_gradient:
        grad = command_margin_gradient(run_context, case, base)
        g_xy = np.asarray([grad[dim_x], grad[dim_y]], dtype=np.float32)
        g_norm = float(np.linalg.norm(g_xy))
        if np.isfinite(g_norm) and g_norm > 1e-9:
            x_span = max(float(x_values[-1] - x_values[0]), 1e-6)
            y_span = max(float(y_values[-1] - y_values[0]), 1e-6)
            scale = 0.14 * min(x_span, y_span) / g_norm
            ax.arrow(
                float(base[dim_x]),
                float(base[dim_y]),
                float(g_xy[0] * scale),
                float(g_xy[1] * scale),
                width=0.01 * min(x_span, y_span),
                facecolor="yellow",
                edgecolor="black",
                length_includes_head=True,
                label="local target-margin gradient",
                zorder=7,
            )

    ax.set_xlabel(f"command dimension {dim_x}")
    ax.set_ylabel(f"command dimension {dim_y}")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    save_paper_figure(fig, output_path)
    plt.close(fig)


def plot_line_profile(output_path, run_context, case, target_result, eval_settings, title):
    if target_result is None:
        return
    base = np.asarray(case["remaining_command"], dtype=np.float32)
    target_delta = np.asarray(target_result.command, dtype=np.float32) - base
    alphas = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    margins = []
    foil_probs = []
    original_probs = []
    for alpha in alphas:
        result = cf.evaluate_delta(
            run_context["model"],
            case["obs"],
            base,
            target_delta * alpha,
            int(case["policy"]["action"]),
            int(case["foil_action"]),
            case["action_mask"],
            eval_settings,
            run_context["device"],
            clip_bounds=cf.command_bounds(run_context["setup"]),
        )
        margins.append(float(result.target_margin))
        foil_probs.append(float(result.probs[int(case["foil_action"])]))
        original_probs.append(float(result.probs[int(case["policy"]["action"])]))

    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    ax1.plot(alphas, margins, color="#111111", linewidth=2.0, label="target margin")
    ax1.axhline(float(eval_settings.margin), color="#111111", linestyle="--", linewidth=1.0, label="success margin")
    ax1.set_xlabel("alpha along movement: R(alpha) = R0 + alpha * delta")
    ax1.set_ylabel("target margin")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(alphas, foil_probs, color="#0072B2", linewidth=1.8, label="foil probability")
    ax2.plot(alphas, original_probs, color="#D55E00", linewidth=1.8, label="original-action probability")
    ax2.set_ylabel("probability")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", fontsize=9)
    ax1.set_title(title)
    fig.tight_layout()
    save_paper_figure(fig, output_path)
    plt.close(fig)


def result_summary(prefix, result, run_context, case):
    if result is None:
        return {
            f"{prefix}_success": "false",
            f"{prefix}_scaled_l2": "",
            f"{prefix}_target_margin": "",
            f"{prefix}_prob_foil": "",
            f"{prefix}_greedy_action": "",
            f"{prefix}_greedy_label": "",
            f"{prefix}_command": "",
            f"{prefix}_delta": "",
        }
    return {
        f"{prefix}_success": bool_text(result.success),
        f"{prefix}_scaled_l2": maybe_float(scaled_distance_for_result(run_context, case, result)),
        f"{prefix}_target_margin": maybe_float(result.target_margin),
        f"{prefix}_prob_foil": maybe_float(result.probs[int(case["foil_action"])]),
        f"{prefix}_greedy_action": int(result.greedy_action),
        f"{prefix}_greedy_label": cf.action_label(run_context["env"], int(result.greedy_action)),
        f"{prefix}_command": json_vector(result.command),
        f"{prefix}_delta": json_vector(result.delta),
    }


def run_landscape_experiment(args, output_dir, zoo_settings, cw_settings, eval_settings, device):
    landscape_dir = Path(output_dir) / "experiment_1_landscapes"
    landscape_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for spec in choose_landscape_cases(args):
        run_context = None
        try:
            run_context = load_run_by_name(spec["run"], args.checkpoint, device)
            case = rollout_single_case(
                run_context,
                int(spec["front"]),
                int(spec["timestep"]),
                int(spec["foil"]),
            )
            dims = [int(spec["dims"][0]), int(spec["dims"][1])]
            if max(dims) >= int(run_context["reward_dim"]):
                raise ValueError(f"dims {dims} are outside reward dimension {run_context['reward_dim']}")

            ours_bundle, ours_error = run_ours(run_context, case, zoo_settings)
            ours_result = None if ours_bundle is None else ours_bundle["final_result"]
            cw_bundle = run_cw_case(run_context, case, cw_settings, eval_settings)
            cw_result = cw_bundle["final_result"]
            front_result = known_front_success(run_context, case, eval_settings)

            x_values, y_values, margins, foil_probs, greedy = heatmap_arrays(
                run_context,
                case,
                dims,
                args.heatmap_grid,
                eval_settings,
            )
            case_name = spec.get("name") or f"{spec['run']}_front{spec['front']}_t{spec['timestep']}_foil{spec['foil']}"
            title = (
                f"{case_name}: {run_context['env_name']} | "
                f"original {action_name(run_context['env'], case['policy']['action'])} -> "
                f"foil {action_name(run_context['env'], case['foil_action'])}"
            )
            plot_landscape(
                landscape_dir / f"{case_name}_margin_landscape.png",
                run_context,
                case,
                dims,
                x_values,
                y_values,
                margins,
                eval_settings.margin,
                ours_result,
                cw_result,
                front_result,
                title,
                include_gradient=True,
            )

            target_for_profile = ours_result if ours_result is not None and ours_result.success else front_result
            plot_line_profile(
                landscape_dir / f"{case_name}_movement_profile.png",
                run_context,
                case,
                target_for_profile,
                eval_settings,
                f"{case_name}: boundary crossing along selected movement",
            )

            np.savez_compressed(
                landscape_dir / f"{case_name}_landscape_arrays.npz",
                x_values=x_values,
                y_values=y_values,
                margins=margins,
                foil_probs=foil_probs,
                greedy=greedy,
                dims=np.asarray(dims, dtype=np.int64),
            )

            row = {
                "case_name": case_name,
                "run": run_context["run_name"],
                "env": run_context["env_name"],
                "front_index": int(case["front_index"]),
                "timestep": int(case["timestep"]),
                "foil_action": int(case["foil_action"]),
                "foil_label": cf.action_label(run_context["env"], int(case["foil_action"])),
                "original_action": int(case["policy"]["action"]),
                "original_label": cf.action_label(run_context["env"], int(case["policy"]["action"])),
                "dims": json.dumps(dims),
                "remaining_command": json_vector(case["remaining_command"]),
                "original_prob_foil": maybe_float(case["policy"]["probs"][int(case["foil_action"])]),
                "original_prob_greedy": maybe_float(case["policy"]["probs"][int(case["policy"]["action"])]),
                "ours_error": ours_error,
                "cw_error": cw_bundle["error"],
            }
            if ours_bundle is not None:
                row.update(
                    {
                        "ours_ray_success": bool_text(ours_bundle["seed_result"] is not None and ours_bundle["seed_result"].success),
                        "ours_zoo_success": bool_text(ours_bundle["selected_result"] is not None and ours_bundle["selected_result"].success),
                        "ours_zoo_improved_seed": bool_text(ours_bundle["zoo_improved_seed"]),
                        "ours_binary_improved_selected": bool_text(ours_bundle["binary_improved_selected"]),
                        "ours_total_queries": int(ours_bundle["total_queries"]),
                        "ours_duration_sec": maybe_float(ours_bundle["duration_sec"]),
                    }
                )
            row.update(result_summary("ours", ours_result, run_context, case))
            row.update(result_summary("cw", cw_result, run_context, case))
            row.update(result_summary("front", front_result, run_context, case))
            rows.append(row)
            print(f"Experiment 1 case {case_name}: ours={row['ours_success']} cw={row['cw_success']}")
        except Exception as exc:
            rows.append(
                {
                    "case_name": spec.get("name", ""),
                    "run": spec.get("run", ""),
                    "front_index": spec.get("front", ""),
                    "timestep": spec.get("timestep", ""),
                    "foil_action": spec.get("foil", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"Experiment 1 skipped {spec}: {type(exc).__name__}: {exc}")
        finally:
            if run_context is not None:
                close_run(run_context)

    write_csv(landscape_dir / "landscape_cases.csv", rows)
    return rows


def top_foil_cases(run_context, args):
    max_fronts = None if args.full_benchmark else args.max_fronts
    max_timesteps = None if args.full_benchmark else args.max_timesteps
    foil_top_k = None if args.full_benchmark else args.foil_top_k
    cases = []
    for case in zoo_batch.rollout_cases(
        run_context,
        max_fronts=max_fronts,
        max_timesteps=max_timesteps,
        foil_top_k=foil_top_k,
    ):
        cases.append(case)
        if not args.full_benchmark and args.max_benchmark_cases is not None:
            if len(cases) >= int(args.max_benchmark_cases):
                break
    return cases


def benchmark_long_row(case_id, method, run_context, case, result, success, duration_sec, extra):
    env = run_context["env"]
    policy = case["policy"]
    a_star = int(policy["action"])
    a_foil = int(case["foil_action"])
    probs = result.probs if result is not None else np.full_like(policy["probs"], np.nan)
    row = {
        "case_id": int(case_id),
        "method": method,
        "run": run_context["run_name"],
        "env": run_context["env_name"],
        "checkpoint": run_context["checkpoint_path"].name,
        "front_index": int(case["front_index"]),
        "timestep": int(case["timestep"]),
        "reward_dim": int(run_context["reward_dim"]),
        "foil_action": a_foil,
        "foil_label": cf.action_label(env, a_foil),
        "original_action": a_star,
        "original_label": cf.action_label(env, a_star),
        "valid_action_count": int(len(case["valid_actions"])),
        "desired_return": json_vector(case["desired_return"]),
        "remaining_command": json_vector(case["remaining_command"]),
        "original_prob_foil": maybe_float(policy["probs"][a_foil]),
        "original_prob_greedy": maybe_float(policy["probs"][a_star]),
        "success": bool_text(success),
        "duration_sec": maybe_float(duration_sec),
        "scaled_l2": maybe_float(scaled_distance_for_result(run_context, case, result)),
        "l2": maybe_float(None if result is None else result.l2_norm),
        "objective": maybe_float(None if result is None else result.objective),
        "hinge": maybe_float(None if result is None else result.hinge),
        "target_margin": maybe_float(None if result is None else result.target_margin),
        "prob_foil": maybe_float(probs[a_foil]),
        "prob_original_action": maybe_float(probs[a_star]),
        "greedy_action": "" if result is None else int(result.greedy_action),
        "greedy_label": "" if result is None else cf.action_label(env, int(result.greedy_action)),
        "command": json_vector(None if result is None else result.command),
        "delta": json_vector(None if result is None else result.delta),
    }
    row.update(extra)
    return row


def run_benchmark(args, output_dir, zoo_settings, cw_settings, eval_settings, device):
    benchmark_dir = Path(output_dir) / "experiment_2_same_case_benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = zoo_batch.discover_runs(args.runs)
    if args.max_runs is not None and not args.full_benchmark:
        run_dirs = run_dirs[: int(args.max_runs)]

    long_rows = []
    paired_rows = []
    case_id = 0

    for run_dir in run_dirs:
        run_context = None
        try:
            run_context = zoo_batch.load_run(run_dir, args.checkpoint, device)
            run_context["device"] = device
            cases = top_foil_cases(run_context, args)
            print(f"Experiment 2 run {run_context['run_name']}: {len(cases)} cases")

            for case in cases:
                if not args.full_benchmark and args.max_benchmark_cases is not None:
                    if case_id >= int(args.max_benchmark_cases):
                        break

                ours_bundle, ours_error = run_ours(run_context, case, zoo_settings)
                ours_result = None if ours_bundle is None else ours_bundle["final_result"]
                ours_success = bool(ours_result is not None and ours_result.success)

                cw_bundle = run_cw_case(run_context, case, cw_settings, eval_settings)
                cw_result = cw_bundle["final_result"]
                cw_success = bool(cw_result is not None and cw_result.success)

                long_rows.append(
                    benchmark_long_row(
                        case_id,
                        "ray_zoo",
                        run_context,
                        case,
                        ours_result,
                        ours_success,
                        "" if ours_bundle is None else ours_bundle["duration_sec"],
                        {
                            "ray_queries": "" if ours_bundle is None else int(ours_bundle["ray_queries"]),
                            "zoo_queries": "" if ours_bundle is None else int(ours_bundle["zoo_queries"]),
                            "binary_queries": "" if ours_bundle is None else int(ours_bundle["binary_queries"]),
                            "total_queries": "" if ours_bundle is None else int(ours_bundle["total_queries"]),
                            "ray_success": "" if ours_bundle is None else bool_text(ours_bundle["seed_result"] is not None and ours_bundle["seed_result"].success),
                            "zoo_success": "" if ours_bundle is None else bool_text(ours_bundle["selected_result"] is not None and ours_bundle["selected_result"].success),
                            "zoo_improved_seed": "" if ours_bundle is None else bool_text(ours_bundle["zoo_improved_seed"]),
                            "binary_improved_selected": "" if ours_bundle is None else bool_text(ours_bundle["binary_improved_selected"]),
                            "failure_reason": ours_error if ours_error else ("" if ours_bundle is None else ours_bundle["failure_reason"]),
                        },
                    )
                )
                long_rows.append(
                    benchmark_long_row(
                        case_id,
                        "art_cw",
                        run_context,
                        case,
                        cw_result,
                        cw_success,
                        cw_bundle["duration_sec"],
                        {
                            "cw_binary_search_steps": int(cw_settings.binary_search_steps),
                            "cw_max_iter": int(cw_settings.max_iter),
                            "cw_nominal_steps": int(cw_settings.binary_search_steps) * int(cw_settings.max_iter),
                            "failure_reason": cw_bundle["error"],
                        },
                    )
                )

                paired = {
                    "case_id": int(case_id),
                    "run": run_context["run_name"],
                    "env": run_context["env_name"],
                    "front_index": int(case["front_index"]),
                    "timestep": int(case["timestep"]),
                    "original_action": int(case["policy"]["action"]),
                    "original_label": cf.action_label(run_context["env"], int(case["policy"]["action"])),
                    "foil_action": int(case["foil_action"]),
                    "foil_label": cf.action_label(run_context["env"], int(case["foil_action"])),
                    "remaining_command": json_vector(case["remaining_command"]),
                    "original_prob_foil": maybe_float(case["policy"]["probs"][int(case["foil_action"])]),
                    "original_prob_greedy": maybe_float(case["policy"]["probs"][int(case["policy"]["action"])]),
                    "ours_success": bool_text(ours_success),
                    "cw_success": bool_text(cw_success),
                    "ours_scaled_l2": maybe_float(scaled_distance_for_result(run_context, case, ours_result)),
                    "cw_scaled_l2": maybe_float(scaled_distance_for_result(run_context, case, cw_result)),
                    "ours_target_margin": maybe_float(None if ours_result is None else ours_result.target_margin),
                    "cw_target_margin": maybe_float(None if cw_result is None else cw_result.target_margin),
                    "ours_total_queries": "" if ours_bundle is None else int(ours_bundle["total_queries"]),
                    "ours_ray_success": "" if ours_bundle is None else bool_text(ours_bundle["seed_result"] is not None and ours_bundle["seed_result"].success),
                    "ours_zoo_success": "" if ours_bundle is None else bool_text(ours_bundle["selected_result"] is not None and ours_bundle["selected_result"].success),
                    "ours_zoo_improved_seed": "" if ours_bundle is None else bool_text(ours_bundle["zoo_improved_seed"]),
                    "ours_binary_improved_selected": "" if ours_bundle is None else bool_text(ours_bundle["binary_improved_selected"]),
                    "ours_duration_sec": "" if ours_bundle is None else maybe_float(ours_bundle["duration_sec"]),
                    "cw_duration_sec": maybe_float(cw_bundle["duration_sec"]),
                    "ours_command": json_vector(None if ours_result is None else ours_result.command),
                    "cw_command": json_vector(None if cw_result is None else cw_result.command),
                    "ours_error": ours_error,
                    "cw_error": cw_bundle["error"],
                }
                paired_rows.append(paired)
                case_id += 1

            if not args.full_benchmark and args.max_benchmark_cases is not None:
                if case_id >= int(args.max_benchmark_cases):
                    break
        except Exception as exc:
            print(f"Experiment 2 skipped run {run_dir}: {type(exc).__name__}: {exc}")
        finally:
            if run_context is not None:
                close_run(run_context)

    write_csv(benchmark_dir / "benchmark_cases_long.csv", long_rows)
    write_csv(benchmark_dir / "benchmark_cases_paired.csv", paired_rows)
    summaries = save_benchmark_summaries_and_plots(benchmark_dir, long_rows, paired_rows)
    return long_rows, paired_rows, summaries


def group_success_summary(rows, group_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    summary = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        successes = [row for row in items if row.get("success") == "true"]
        distances = [float(row["scaled_l2"]) for row in successes if row.get("scaled_l2") != ""]
        durations = [float(row["duration_sec"]) for row in items if row.get("duration_sec") != ""]
        out = {group_keys[index]: key[index] for index in range(len(group_keys))}
        out.update(
            {
                "cases": len(items),
                "successes": len(successes),
                "success_rate": len(successes) / len(items) if items else 0.0,
                "median_scaled_l2_success": float(np.median(distances)) if distances else "",
                "mean_scaled_l2_success": float(np.mean(distances)) if distances else "",
                "median_duration_sec": float(np.median(durations)) if durations else "",
                "mean_duration_sec": float(np.mean(durations)) if durations else "",
            }
        )
        summary.append(out)
    return summary


def numeric_from_rows(rows, key, success_only=False):
    values = []
    for row in rows:
        if success_only and row.get("success") != "true":
            continue
        raw = row.get(key, "")
        if raw == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def save_grouped_success_plot(path, summary_by_env_method):
    envs = sorted({row["env"] for row in summary_by_env_method})
    methods = ["ray_zoo", "art_cw"]
    method_labels = {"ray_zoo": "Ray+ZOO", "art_cw": "ART C&W"}
    lookup = {(row["env"], row["method"]): float(row["success_rate"]) for row in summary_by_env_method}
    x = np.arange(len(envs))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8.5, 0.65 * len(envs)), 5.0))
    for hatch, (offset, method) in zip(("", "///"), [(-width / 2, "ray_zoo"), (width / 2, "art_cw")]):
        values = [lookup.get((env, method), 0.0) for env in envs]
        bars = ax.bar(x + offset, values, width=width, label=method_labels[method], edgecolor="#111111", linewidth=0.6)
        apply_bar_hatches(bars, hatches=(hatch,))
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("Same-case counterfactual success by environment")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_paper_figure(fig, path)
    plt.close(fig)


def save_method_bar(path, summary_by_method, metric, title, ylabel):
    labels = [row["method"] for row in summary_by_method]
    values = [float(row[metric]) if row[metric] != "" else 0.0 for row in summary_by_method]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    bars = ax.bar(
        labels,
        values,
        color=["#0072B2" if label == "ray_zoo" else "#D55E00" for label in labels],
        edgecolor="#111111",
        linewidth=0.6,
    )
    apply_bar_hatches(bars)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if "rate" in metric:
        ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_paper_figure(fig, path)
    plt.close(fig)


def save_distance_boxplot(path, long_rows):
    methods = ["ray_zoo", "art_cw"]
    labels = ["Ray+ZOO", "ART C&W"]
    data = [numeric_from_rows([row for row in long_rows if row["method"] == method], "scaled_l2", success_only=True) for method in methods]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_title("Scaled L2 distance for successful counterfactuals")
    ax.set_ylabel("scaled L2 distance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_paper_figure(fig, path)
    plt.close(fig)


def save_runtime_boxplot(path, long_rows):
    methods = ["ray_zoo", "art_cw"]
    labels = ["Ray+ZOO", "ART C&W"]
    data = [numeric_from_rows([row for row in long_rows if row["method"] == method], "duration_sec") for method in methods]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_title("Runtime per same-case attack")
    ax.set_ylabel("seconds")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_paper_figure(fig, path)
    plt.close(fig)


def save_stage_plot(path, paired_rows):
    labels = ["Ray seed", "ZOO success", "ZOO improved", "Binary improved", "Final ours", "Final C&W"]
    values = [
        sum(1 for row in paired_rows if row.get("ours_ray_success") == "true"),
        sum(1 for row in paired_rows if row.get("ours_zoo_success") == "true"),
        sum(1 for row in paired_rows if row.get("ours_zoo_improved_seed") == "true"),
        sum(1 for row in paired_rows if row.get("ours_binary_improved_selected") == "true"),
        sum(1 for row in paired_rows if row.get("ours_success") == "true"),
        sum(1 for row in paired_rows if row.get("cw_success") == "true"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        values,
        color=["#56B4E9", "#0072B2", "#009E73", "#F0E442", "#004C7F", "#D55E00"],
        edgecolor="#111111",
        linewidth=0.6,
    )
    apply_bar_hatches(bars)
    ax.set_title("Pipeline outcomes on benchmark cases")
    ax.set_ylabel("number of cases")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_paper_figure(fig, path)
    plt.close(fig)


def save_pair_outcome_plot(path, paired_rows):
    counts = {
        "both success": 0,
        "ours only": 0,
        "C&W only": 0,
        "both fail": 0,
    }
    for row in paired_rows:
        ours = row.get("ours_success") == "true"
        art = row.get("cw_success") == "true"
        if ours and art:
            counts["both success"] += 1
        elif ours:
            counts["ours only"] += 1
        elif art:
            counts["C&W only"] += 1
        else:
            counts["both fail"] += 1
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(
        counts.keys(),
        counts.values(),
        color=["#009E73", "#0072B2", "#D55E00", "#777777"],
        edgecolor="#111111",
        linewidth=0.6,
    )
    apply_bar_hatches(bars)
    ax.set_title("Paired same-case outcomes")
    ax.set_ylabel("number of cases")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_paper_figure(fig, path)
    plt.close(fig)


def save_benchmark_summaries_and_plots(benchmark_dir, long_rows, paired_rows):
    summary_by_method = group_success_summary(long_rows, ["method"])
    summary_by_env_method = group_success_summary(long_rows, ["env", "method"])
    summary_by_run_method = group_success_summary(long_rows, ["run", "method"])

    write_csv(Path(benchmark_dir) / "summary_by_method.csv", summary_by_method)
    write_csv(Path(benchmark_dir) / "summary_by_env_method.csv", summary_by_env_method)
    write_csv(Path(benchmark_dir) / "summary_by_run_method.csv", summary_by_run_method)

    plots_dir = Path(benchmark_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if long_rows:
        save_method_bar(
            plots_dir / "success_rate_by_method.png",
            summary_by_method,
            "success_rate",
            "Overall same-case success rate",
            "success rate",
        )
        save_grouped_success_plot(plots_dir / "success_rate_by_env_method.png", summary_by_env_method)
        save_method_bar(
            plots_dir / "median_runtime_by_method.png",
            summary_by_method,
            "median_duration_sec",
            "Median runtime by method",
            "seconds",
        )
        save_distance_boxplot(plots_dir / "scaled_l2_success_boxplot.png", long_rows)
        save_runtime_boxplot(plots_dir / "runtime_boxplot.png", long_rows)
    if paired_rows:
        save_stage_plot(plots_dir / "ours_stage_counts.png", paired_rows)
        save_pair_outcome_plot(plots_dir / "paired_outcomes.png", paired_rows)

    summary = {
        "cases": len(paired_rows),
        "long_rows": len(long_rows),
        "summary_by_method": summary_by_method,
        "summary_by_env_method": summary_by_env_method,
        "summary_by_run_method": summary_by_run_method,
    }
    with (Path(benchmark_dir) / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def save_config(output_dir, args, zoo_settings, cw_settings, eval_settings):
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "zoo_settings": zoo_settings.__dict__,
        "cw_settings": cw_settings.__dict__,
        "eval_settings": eval_settings.__dict__,
    }
    with (Path(output_dir) / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper experiments for PCN command-space counterfactuals: Ray+ZOO vs ART C&W."
    )
    parser.add_argument("--label", default="cw_vs_zoo", help="Name prefix for the artifact directory.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint name/path/index; default uses latest.")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=DEFAULT_BENCHMARK_RUNS,
        help="Run names/directories for the benchmark.",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Cap benchmark runs in bounded mode.")
    parser.add_argument("--skip-experiment1", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true", help="Use all logged fronts, timesteps, and foils.")
    parser.add_argument("--max-fronts", type=int, default=4, help="Bounded benchmark: sampled logged fronts per run.")
    parser.add_argument("--max-timesteps", type=int, default=4, help="Bounded benchmark: policy timesteps per front.")
    parser.add_argument("--foil-top-k", type=int, default=1, help="Bounded benchmark: most likely non-greedy foils.")
    parser.add_argument("--max-benchmark-cases", type=int, default=60, help="Global bounded benchmark case cap.")
    parser.add_argument("--landscape-limit", type=int, default=5, help="Number of landscape cases to plot.")
    parser.add_argument("--landscape-case-json", default=None, help="Optional JSON list overriding landscape cases.")
    parser.add_argument("--heatmap-grid", type=int, default=61, help="Grid resolution per heatmap dimension.")
    parser.add_argument("--success-margin", type=float, default=cf.DEFAULT_ZOO_SETTINGS["margin"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-clip-command", action="store_true", help="Allow commands outside env reward bounds.")

    parser.add_argument("--zoo-max-queries", type=int, default=6000)
    parser.add_argument(
        "--zoo-random-directions",
        type=int,
        default=None,
        help="Random ray directions for the seed stage. Default: match --zoo-max-queries so the ray budget is never direction-limited.",
    )
    parser.add_argument("--zoo-line-search-steps", type=int, default=16)
    parser.add_argument("--zoo-learning-rate", type=float, default=cf.DEFAULT_ZOO_SETTINGS["learning_rate"])
    parser.add_argument("--zoo-finite-diff-h", type=float, default=cf.DEFAULT_ZOO_SETTINGS["finite_diff_h"])
    parser.add_argument("--zoo-c", type=float, default=cf.DEFAULT_ZOO_SETTINGS["c"])

    parser.add_argument("--cw-confidence", type=float, default=0.05)
    parser.add_argument("--cw-learning-rate", type=float, default=0.001)
    parser.add_argument("--cw-binary-search-steps", type=int, default=3)
    parser.add_argument("--cw-max-iter", type=int, default=250)
    parser.add_argument("--cw-initial-const", type=float, default=1.0)
    parser.add_argument("--cw-max-halving", type=int, default=5)
    parser.add_argument("--cw-max-doubling", type=int, default=5)
    parser.add_argument("--cw-batch-size", type=int, default=1)
    parser.add_argument("--cw-art-clip-relaxation", type=float, default=1e-2)
    parser.add_argument("--cw-verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = preferred_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output_artifacts") / "cw_vs_zoo_paper_experiments" / f"{args.label}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    zoo_settings = make_zoo_settings(args)
    cw_settings = make_cw_settings(args)
    eval_settings = make_eval_settings(args.success_margin, clip_command=not args.no_clip_command)
    save_config(output_dir, args, zoo_settings, cw_settings, eval_settings)

    print("=" * 72)
    print("PCN command-space counterfactual paper experiments")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Success margin: {eval_settings.margin}")
    print()

    if not args.skip_experiment1:
        print("Experiment 1: landscape heatmaps and boundary movement plots")
        run_landscape_experiment(args, output_dir, zoo_settings, cw_settings, eval_settings, device)
        print()

    if not args.skip_benchmark:
        print("Experiment 2: same-case Ray+ZOO vs ART C&W benchmark")
        long_rows, paired_rows, summary = run_benchmark(args, output_dir, zoo_settings, cw_settings, eval_settings, device)
        print(f"Benchmark cases: {len(paired_rows)}")
        for row in summary.get("summary_by_method", []):
            print(
                f"  {row['method']}: success {row['successes']}/{row['cases']} "
                f"({float(row['success_rate']):.3f})"
            )

    print(f"\nSaved artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
