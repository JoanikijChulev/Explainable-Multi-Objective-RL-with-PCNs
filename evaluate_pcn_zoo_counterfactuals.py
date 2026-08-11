from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import interactive_pcn_zoo_cf as cf
from artifacts import load_run_metadata, output_artifacts_dir, resolve_run_dir
from device_utils import device_description, preferred_device
from eval_pcn import load_logged_front, resolve_checkpoint_path
from paper_plot_utils import apply_bar_hatches, save_paper_figure
from pcn.env_setup import build_experiment_setup, canonical_env_name


RESULT_FIELDS = [
    "case_id",
    "run",
    "env",
    "checkpoint",
    "front_index",
    "timestep",
    "reward_dim",
    "foil_action",
    "foil_label",
    "original_action",
    "original_label",
    "valid_action_count",
    "desired_return",
    "remaining_command",
    "original_prob_foil",
    "original_prob_greedy",
    "ray_success",
    "zoo_success",
    "final_success",
    "zoo_improved_seed",
    "binary_improved_selected",
    "ray_queries",
    "zoo_queries",
    "binary_queries",
    "total_queries",
    "seed_scaled_l2",
    "selected_scaled_l2",
    "final_scaled_l2",
    "final_l2",
    "final_objective",
    "final_hinge",
    "final_target_margin",
    "final_prob_foil",
    "final_prob_original_action",
    "final_greedy_action",
    "final_greedy_label",
    "final_command",
    "final_delta",
    "failure_reason",
    "duration_sec",
]


def json_vector(value):
    if value is None:
        return ""
    return json.dumps(np.asarray(value, dtype=np.float32).tolist())


def maybe_float(value):
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(float(value)):
            return ""
    return float(value)


def bool_text(value):
    return "true" if bool(value) else "false"


def infer_env_name(run_dir, metadata):
    env_name = metadata.get("env")
    if env_name:
        return canonical_env_name(env_name)

    text = str(run_dir).lower()
    aliases = [
        ("dst", "dst"),
        ("deepsea", "dst"),
        ("collect_two", "collect_two"),
        ("collect-two", "collect_two"),
        ("c2", "collect_two"),
        ("branch-path", "branch-path"),
        ("branch_path", "branch-path"),
        ("bp", "branch-path"),
        ("reward-line", "reward-line"),
        ("reward_line", "reward-line"),
        ("three-tree", "three-tree"),
        ("three_tree", "three-tree"),
        ("minecart", "minecart"),
        ("mc", "minecart"),
        ("fruit-tree", "fruit-tree-v0"),
        ("fruit_tree", "fruit-tree-v0"),
        ("ft", "fruit-tree-v0"),
        ("four-room", "four-room-v0"),
        ("four_room", "four-room-v0"),
        ("resource-gathering", "resource-gathering-v0"),
        ("resource_gathering", "resource-gathering-v0"),
        ("rsg", "resource-gathering-v0"),
        ("breakable-bottles", "breakable-bottles-v0"),
        ("breakable_bottles", "breakable-bottles-v0"),
        ("bb", "breakable-bottles-v0"),
    ]
    for token, canonical in aliases:
        if re.search(rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)", text):
            return canonical_env_name(canonical)
    raise ValueError(f"could not infer environment for {run_dir}")


def discover_runs(requested_runs):
    if requested_runs:
        return [resolve_run_dir(run) for run in requested_runs]

    runs = []
    for candidate in sorted(output_artifacts_dir().iterdir()):
        if not candidate.is_dir():
            continue
        has_log = (candidate / "log.h5").is_file()
        has_checkpoint_dir = (candidate / "checkpoints").is_dir()
        has_root_checkpoint = bool(list(candidate.glob("model_*.pt")))
        if has_log and (has_checkpoint_dir or has_root_checkpoint):
            runs.append(candidate.resolve())
    return runs


def load_run(run_dir, checkpoint_arg, device):
    metadata = load_run_metadata(run_dir)
    env_name = infer_env_name(run_dir, metadata)
    fruit_tree_depth = int(metadata.get("fruit_tree_depth", 6))
    setup = build_experiment_setup(
        env_name,
        device=device,
        fruit_tree_depth=fruit_tree_depth,
        include_model=False,
    )
    checkpoint_path, _ = resolve_checkpoint_path(run_dir, checkpoint_arg)
    model = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device)
    if hasattr(model, "scaling_factor"):
        model.scaling_factor = model.scaling_factor.to(device)
    model.eval()

    reward_dim = int(np.asarray(setup.max_return).shape[-1])
    with h5py.File(Path(run_dir) / "log.h5", "r") as log:
        front = load_logged_front(log, list(range(reward_dim))).astype(np.float32)
    if len(front) == 0:
        raise ValueError("logged Pareto front is empty")

    return {
        "run_dir": Path(run_dir),
        "run_name": Path(run_dir).name,
        "metadata": metadata,
        "setup": setup,
        "env": setup.env,
        "env_name": setup.env_name,
        "model": model,
        "checkpoint_path": checkpoint_path,
        "front": front,
        "reward_dim": reward_dim,
    }


def selected_front_indices(front, max_fronts):
    if max_fronts is None or max_fronts <= 0 or len(front) <= max_fronts:
        return list(range(len(front)))
    indexes = np.linspace(0, len(front) - 1, int(max_fronts))
    return sorted({int(round(index)) for index in indexes})


def rollout_cases(run_context, max_fronts=None, max_timesteps=None, foil_top_k=None):
    env = run_context["env"]
    model = run_context["model"]
    device = run_context["device"]
    front = run_context["front"]

    for front_index in selected_front_indices(front, max_fronts):
        desired_return = np.asarray(front[front_index], dtype=np.float32)
        obs, _ = cf.reset_env(env)
        remaining_command = desired_return.copy()
        timestep = 0

        while True:
            policy = cf.greedy_action(model, env, obs, remaining_command, device)
            action_mask = policy["action_mask"]
            if action_mask is None:
                action_mask = np.ones(len(policy["masked_log_probs"]), dtype=bool)
            valid_actions = np.flatnonzero(action_mask).astype(np.int64)
            original_action = int(policy["action"])
            foil_actions = [int(action) for action in valid_actions if int(action) != original_action]
            foil_actions.sort(key=lambda action: float(policy["probs"][action]), reverse=True)
            if foil_top_k is not None and foil_top_k > 0:
                foil_actions = foil_actions[: int(foil_top_k)]

            for foil_action in foil_actions:
                yield {
                    "front_index": int(front_index),
                    "desired_return": desired_return.copy(),
                    "timestep": int(timestep),
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

            if max_timesteps is not None and timestep + 1 >= int(max_timesteps):
                break

            next_obs, reward, terminated, truncated, _ = cf.step_env(env, original_action)
            if terminated or truncated:
                break
            remaining_command = (remaining_command - reward).astype(np.float32)
            obs = next_obs
            timestep += 1


def scaled_distance(result, base_command, bounds):
    if result is None:
        return None
    return cf.scaled_command_distance(result.command, base_command, cf.command_scale(bounds))


def run_counterfactual_case(run_context, case, settings):
    model = run_context["model"]
    device = run_context["device"]
    setup = run_context["setup"]
    base_command = case["remaining_command"]
    obs = case["obs"]
    action_mask = case["action_mask"]
    a_star = int(case["policy"]["action"])
    a_foil = int(case["foil_action"])
    env_bounds = cf.command_bounds(setup)
    clip_bounds = env_bounds if settings.clip_command else None
    start = time.perf_counter()

    search_result = cf.constrained_nearest_boundary_search(
        model,
        obs,
        base_command,
        case["desired_return"],
        a_star,
        a_foil,
        action_mask,
        settings,
        device,
        env_bounds,
        front=run_context["front"],
    )
    ray_queries = int(search_result["queries"])
    seed_result = search_result["best_success"]
    seed_success = bool(seed_result is not None and seed_result.success)

    zoo_queries = 0
    binary_queries = 0
    zoo_result = None
    selected_result = seed_result
    selected_bounds = search_result["bounds"]
    selected_before_binary = None

    if seed_success:
        zoo_result = cf.zoo_coordinate_search(
            model,
            obs,
            base_command,
            a_star,
            a_foil,
            action_mask,
            settings,
            device,
            clip_bounds=selected_bounds if settings.clip_command else None,
            initial_delta=seed_result.delta,
        )
        zoo_queries = int(zoo_result["queries"])
        selected_result = cf.better_success_scaled(
            zoo_result["best_success"],
            seed_result,
            base_command,
            cf.command_scale(selected_bounds),
        )
    else:
        zoo_result = cf.zoo_coordinate_search(
            model,
            obs,
            base_command,
            a_star,
            a_foil,
            action_mask,
            settings,
            device,
            clip_bounds=clip_bounds,
        )
        zoo_queries = int(zoo_result["queries"])
        selected_result = zoo_result["best_success"]
        selected_bounds = env_bounds

    selected_before_binary = selected_result
    queries_before_binary = ray_queries + zoo_queries
    final_result = selected_result

    if selected_result is not None and selected_result.success:
        refined, _, cumulative_queries = cf.binary_refine_success(
            model,
            obs,
            base_command,
            selected_result.delta,
            a_star,
            a_foil,
            action_mask,
            settings,
            device,
            start_queries=queries_before_binary,
            clip_bounds=selected_bounds if settings.clip_command else None,
            max_binary_queries=settings.line_search_steps,
        )
        binary_queries = int(cumulative_queries) - int(queries_before_binary)
        if refined.success:
            final_result = refined

    if final_result is None:
        final_result = (
            search_result["best_failure"]
            or search_result["best_objective"]
            or (zoo_result or {}).get("best_objective")
        )

    total_queries = ray_queries + zoo_queries + binary_queries
    duration_sec = time.perf_counter() - start
    seed_dist = scaled_distance(seed_result, base_command, selected_bounds)
    selected_dist = scaled_distance(selected_before_binary, base_command, selected_bounds)
    final_dist = scaled_distance(final_result, base_command, selected_bounds)
    zoo_improved = (
        seed_dist is not None
        and selected_dist is not None
        and selected_dist < seed_dist - 1e-8
    )
    binary_improved = (
        selected_dist is not None
        and final_dist is not None
        and final_dist < selected_dist - 1e-8
    )

    if final_result is not None and final_result.success:
        failure_reason = ""
    elif not seed_success and not (selected_before_binary is not None and selected_before_binary.success):
        failure_reason = "no_successful_seed_or_zoo_point"
    elif selected_before_binary is not None and selected_before_binary.success:
        failure_reason = "binary_refinement_lost_success"
    else:
        failure_reason = "unknown_failure"

    return {
        "search_result": search_result,
        "seed_result": seed_result,
        "selected_result": selected_before_binary,
        "final_result": final_result,
        "ray_queries": ray_queries,
        "zoo_queries": zoo_queries,
        "binary_queries": binary_queries,
        "total_queries": total_queries,
        "seed_scaled_l2": seed_dist,
        "selected_scaled_l2": selected_dist,
        "final_scaled_l2": final_dist,
        "zoo_improved_seed": zoo_improved,
        "binary_improved_selected": binary_improved,
        "failure_reason": failure_reason,
        "duration_sec": duration_sec,
    }


def case_to_row(case_id, run_context, case, result_bundle):
    env = run_context["env"]
    final_result = result_bundle["final_result"]
    seed_result = result_bundle["seed_result"]
    selected_result = result_bundle["selected_result"]
    policy = case["policy"]
    original_probs = policy["probs"]
    a_star = int(policy["action"])
    a_foil = int(case["foil_action"])

    final_success = bool(final_result is not None and final_result.success)
    final_probs = final_result.probs if final_result is not None else np.full_like(original_probs, np.nan)

    return {
        "case_id": int(case_id),
        "run": run_context["run_name"],
        "env": run_context["env_name"],
        "checkpoint": run_context["checkpoint_path"].name,
        "front_index": int(case["front_index"]),
        "timestep": int(case["timestep"]),
        "reward_dim": int(run_context["reward_dim"]),
        "foil_action": int(a_foil),
        "foil_label": cf.action_label(env, a_foil),
        "original_action": int(a_star),
        "original_label": cf.action_label(env, a_star),
        "valid_action_count": int(len(case["valid_actions"])),
        "desired_return": json_vector(case["desired_return"]),
        "remaining_command": json_vector(case["remaining_command"]),
        "original_prob_foil": maybe_float(original_probs[a_foil]),
        "original_prob_greedy": maybe_float(original_probs[a_star]),
        "ray_success": bool_text(seed_result is not None and seed_result.success),
        "zoo_success": bool_text(selected_result is not None and selected_result.success),
        "final_success": bool_text(final_success),
        "zoo_improved_seed": bool_text(result_bundle["zoo_improved_seed"]),
        "binary_improved_selected": bool_text(result_bundle["binary_improved_selected"]),
        "ray_queries": int(result_bundle["ray_queries"]),
        "zoo_queries": int(result_bundle["zoo_queries"]),
        "binary_queries": int(result_bundle["binary_queries"]),
        "total_queries": int(result_bundle["total_queries"]),
        "seed_scaled_l2": maybe_float(result_bundle["seed_scaled_l2"]),
        "selected_scaled_l2": maybe_float(result_bundle["selected_scaled_l2"]),
        "final_scaled_l2": maybe_float(result_bundle["final_scaled_l2"]),
        "final_l2": maybe_float(None if final_result is None else final_result.l2_norm),
        "final_objective": maybe_float(None if final_result is None else final_result.objective),
        "final_hinge": maybe_float(None if final_result is None else final_result.hinge),
        "final_target_margin": maybe_float(None if final_result is None else final_result.target_margin),
        "final_prob_foil": maybe_float(final_probs[a_foil]),
        "final_prob_original_action": maybe_float(final_probs[a_star]),
        "final_greedy_action": "" if final_result is None else int(final_result.greedy_action),
        "final_greedy_label": "" if final_result is None else cf.action_label(env, final_result.greedy_action),
        "final_command": json_vector(None if final_result is None else final_result.command),
        "final_delta": json_vector(None if final_result is None else final_result.delta),
        "failure_reason": result_bundle["failure_reason"],
        "duration_sec": maybe_float(result_bundle["duration_sec"]),
    }


def summarize_group(rows, group_key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)

    summary = []
    for key, items in sorted(groups.items(), key=lambda item: str(item[0])):
        successes = [row for row in items if row["final_success"] == "true"]
        ray_successes = [row for row in items if row["ray_success"] == "true"]
        zoo_improvements = [row for row in items if row["zoo_improved_seed"] == "true"]
        binary_improvements = [row for row in items if row["binary_improved_selected"] == "true"]
        query_values = [float(row["total_queries"]) for row in successes if row["total_queries"] != ""]
        distance_values = [float(row["final_scaled_l2"]) for row in successes if row["final_scaled_l2"] != ""]
        summary.append({
            group_key: key,
            "cases": len(items),
            "final_successes": len(successes),
            "success_rate": len(successes) / len(items) if items else 0.0,
            "ray_successes": len(ray_successes),
            "ray_success_rate": len(ray_successes) / len(items) if items else 0.0,
            "zoo_improvements": len(zoo_improvements),
            "binary_improvements": len(binary_improvements),
            "median_queries_success": float(np.median(query_values)) if query_values else "",
            "mean_queries_success": float(np.mean(query_values)) if query_values else "",
            "median_scaled_l2_success": float(np.median(distance_values)) if distance_values else "",
            "mean_scaled_l2_success": float(np.mean(distance_values)) if distance_values else "",
        })
    return summary


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_bool(row, key):
    return row.get(key) == "true"


def save_bar(path, labels, values, title, ylabel, color="#4C78A8"):
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 4.8))
    bars = ax.bar(np.arange(len(labels)), values, color=color, edgecolor="#111111", linewidth=0.6)
    apply_bar_hatches(bars)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0.0, max(1.0, max(values, default=1.0) * 1.15))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_paper_figure(fig, path)
    plt.close(fig)


def save_plots(output_dir, rows, summary_by_env, summary_by_run):
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    env_labels = [row["env"] for row in summary_by_env]
    env_success = [float(row["success_rate"]) for row in summary_by_env]
    save_bar(
        plots_dir / "success_rate_by_env.png",
        env_labels,
        env_success,
        "Counterfactual Success Rate by Environment",
        "Success Rate",
        color="#2F7D62",
    )

    run_labels = [row["run"] for row in summary_by_run]
    run_success = [float(row["success_rate"]) for row in summary_by_run]
    save_bar(
        plots_dir / "success_rate_by_run.png",
        run_labels,
        run_success,
        "Counterfactual Success Rate by Run",
        "Success Rate",
        color="#4C78A8",
    )

    stage_counts = [
        sum(1 for row in rows if as_bool(row, "ray_success")),
        sum(1 for row in rows if as_bool(row, "zoo_improved_seed")),
        sum(1 for row in rows if as_bool(row, "binary_improved_selected")),
        sum(1 for row in rows if as_bool(row, "final_success")),
    ]
    save_bar(
        plots_dir / "stage_counts.png",
        ["Ray Seed", "ZOO Improved", "Binary Improved", "Final Success"],
        stage_counts,
        "Counterfactual Pipeline Outcomes",
        "Number of Cases",
        color="#756BB1",
    )

    success_rows = [row for row in rows if as_bool(row, "final_success")]
    by_env_dist = defaultdict(list)
    for row in success_rows:
        value = row.get("final_scaled_l2", "")
        if value != "":
            by_env_dist[row["env"]].append(float(value))
    if by_env_dist:
        labels = sorted(by_env_dist)
        data = [by_env_dist[label] for label in labels]
        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 5.0))
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title("Scaled Distance of Successful Counterfactuals by Environment")
        ax.set_ylabel("Scaled L2 Distance")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        save_paper_figure(fig, plots_dir / "scaled_distance_by_env.png")
        plt.close(fig)

    by_timestep = defaultdict(list)
    for row in rows:
        by_timestep[int(row["timestep"])].append(row)
    if by_timestep:
        xs = sorted(by_timestep)
        ys = [
            sum(1 for row in by_timestep[timestep] if as_bool(row, "final_success")) / len(by_timestep[timestep])
            for timestep in xs
        ]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(xs, ys, marker="o", color="#4C78A8", linewidth=2)
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Success Rate by Timestep")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Success Rate")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_paper_figure(fig, plots_dir / "success_rate_by_timestep.png")
        plt.close(fig)


def evaluation_output_dir(label):
    root = output_artifacts_dir() / "pcn_zoo_cf_evaluations"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{label}_{timestamp}" if label else f"eval_{timestamp}"
    path = root / name
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch evaluate PCN command-space ZOO counterfactuals over logged Pareto fronts."
    )
    parser.add_argument("--runs", nargs="*", default=None, help="Run names/paths. Default: all runs under output_artifacts.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint name/index/path. Default: latest per run.")
    parser.add_argument("--label", default="eval", help="Output directory label.")
    parser.add_argument("--max-fronts", type=int, default=None, help="Optional evenly-spaced subset of front rows per run.")
    parser.add_argument("--max-timesteps", type=int, default=None, help="Optional max timesteps per front rollout.")
    parser.add_argument("--foil-top-k", type=int, default=None, help="Optional top-k non-greedy foils by original probability.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional global cap for smoke tests.")
    parser.add_argument("--max-queries", type=int, default=cf.DEFAULT_ZOO_SETTINGS["max_queries"])
    parser.add_argument(
        "--random-directions",
        type=int,
        default=None,
        help="Random ray directions for the seed stage. Default: match --max-queries so the ray budget is never direction-limited.",
    )
    parser.add_argument("--line-search-steps", type=int, default=cf.DEFAULT_ZOO_SETTINGS["line_search_steps"])
    parser.add_argument("--learning-rate", type=float, default=cf.DEFAULT_ZOO_SETTINGS["learning_rate"])
    parser.add_argument("--finite-diff-h", type=float, default=cf.DEFAULT_ZOO_SETTINGS["finite_diff_h"])
    parser.add_argument("--margin", type=float, default=cf.DEFAULT_ZOO_SETTINGS["margin"])
    parser.add_argument("--c", type=float, default=cf.DEFAULT_ZOO_SETTINGS["c"])
    parser.add_argument("--seed", type=int, default=cf.DEFAULT_ZOO_SETTINGS["seed"])
    parser.add_argument("--no-clip-command", action="store_true", help="Disable command clipping to environment bounds.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = preferred_device()
    settings = replace(
        cf.ZooSettings(**cf.DEFAULT_ZOO_SETTINGS),
        max_queries=int(args.max_queries),
        random_directions=int(args.random_directions) if args.random_directions is not None else int(args.max_queries),
        line_search_steps=int(args.line_search_steps),
        learning_rate=float(args.learning_rate),
        finite_diff_h=float(args.finite_diff_h),
        margin=float(args.margin),
        c=float(args.c),
        seed=int(args.seed),
        clip_command=not args.no_clip_command,
    )

    output_dir = evaluation_output_dir(args.label)
    config = {
        "device": device_description(device),
        "settings": cf.to_jsonable(settings.__dict__),
        "runs": args.runs if args.runs else "all",
        "checkpoint": args.checkpoint,
        "max_fronts": args.max_fronts,
        "max_timesteps": args.max_timesteps,
        "foil_top_k": args.foil_top_k,
        "max_cases": args.max_cases,
        "output_dir": str(output_dir),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    run_dirs = discover_runs(args.runs)
    print(f"Output: {output_dir}")
    print(f"Device: {device_description(device)}")
    print(f"Runs: {len(run_dirs)}")

    rows = []
    load_errors = []
    case_id = 0
    stop = False

    for run_dir in run_dirs:
        if stop:
            break
        try:
            run_context = load_run(run_dir, args.checkpoint, device)
            run_context["device"] = device
        except Exception as exc:
            load_errors.append({"run": str(run_dir), "error": str(exc)})
            print(f"Skipping {run_dir}: {exc}")
            continue

        setup = run_context["setup"]
        run_case_count = 0
        run_success_count = 0
        print(
            f"\nRun {run_context['run_name']} | env={run_context['env_name']} "
            f"| fronts={len(run_context['front'])} | checkpoint={run_context['checkpoint_path'].name}",
            flush=True,
        )

        try:
            for case in rollout_cases(
                run_context,
                max_fronts=args.max_fronts,
                max_timesteps=args.max_timesteps,
                foil_top_k=args.foil_top_k,
            ):
                if args.max_cases is not None and case_id >= int(args.max_cases):
                    stop = True
                    break
                case_id += 1
                run_case_count += 1
                try:
                    result_bundle = run_counterfactual_case(run_context, case, settings)
                    row = case_to_row(case_id, run_context, case, result_bundle)
                    if row["final_success"] == "true":
                        run_success_count += 1
                except Exception as exc:
                    row = {
                        field: "" for field in RESULT_FIELDS
                    }
                    row.update({
                        "case_id": int(case_id),
                        "run": run_context["run_name"],
                        "env": run_context["env_name"],
                        "checkpoint": run_context["checkpoint_path"].name,
                        "front_index": int(case["front_index"]),
                        "timestep": int(case["timestep"]),
                        "reward_dim": int(run_context["reward_dim"]),
                        "foil_action": int(case["foil_action"]),
                        "foil_label": cf.action_label(run_context["env"], case["foil_action"]),
                        "original_action": int(case["policy"]["action"]),
                        "original_label": cf.action_label(run_context["env"], case["policy"]["action"]),
                        "desired_return": json_vector(case["desired_return"]),
                        "remaining_command": json_vector(case["remaining_command"]),
                        "final_success": "false",
                        "failure_reason": f"exception: {exc}",
                    })
                rows.append(row)
                if case_id % 25 == 0:
                    total_success = sum(1 for item in rows if item["final_success"] == "true")
                    print(f"  cases={case_id} successes={total_success}", flush=True)
        finally:
            try:
                setup.env.close()
            except Exception:
                pass

        print(f"  run cases={run_case_count} successes={run_success_count}", flush=True)

    write_csv(output_dir / "results_cases.csv", rows, RESULT_FIELDS)
    summary_by_env = summarize_group(rows, "env")
    summary_by_run = summarize_group(rows, "run")
    summary_by_front = summarize_group(rows, "front_index")
    write_csv(output_dir / "summary_by_env.csv", summary_by_env, list(summary_by_env[0].keys()) if summary_by_env else ["env"])
    write_csv(output_dir / "summary_by_run.csv", summary_by_run, list(summary_by_run[0].keys()) if summary_by_run else ["run"])
    write_csv(output_dir / "summary_by_front_index.csv", summary_by_front, list(summary_by_front[0].keys()) if summary_by_front else ["front_index"])

    failures = [row for row in rows if row["final_success"] != "true"]
    write_csv(output_dir / "failures.csv", failures, RESULT_FIELDS)
    with (output_dir / "load_errors.json").open("w", encoding="utf-8") as handle:
        json.dump(load_errors, handle, indent=2, sort_keys=True)
        handle.write("\n")

    total_cases = len(rows)
    total_successes = sum(1 for row in rows if row["final_success"] == "true")
    summary = {
        "total_cases": total_cases,
        "total_successes": total_successes,
        "success_rate": total_successes / total_cases if total_cases else 0.0,
        "ray_successes": sum(1 for row in rows if row["ray_success"] == "true"),
        "zoo_improvements": sum(1 for row in rows if row["zoo_improved_seed"] == "true"),
        "binary_improvements": sum(1 for row in rows if row["binary_improved_selected"] == "true"),
        "load_errors": load_errors,
        "output_dir": str(output_dir),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    save_plots(output_dir, rows, summary_by_env, summary_by_run)

    print("\nEvaluation complete")
    print(f"  cases: {total_cases}")
    print(f"  successes: {total_successes}")
    print(f"  success rate: {summary['success_rate']:.3f}")
    print(f"  ray successes: {summary['ray_successes']}")
    print(f"  ZOO improvements: {summary['zoo_improvements']}")
    print(f"  binary improvements: {summary['binary_improvements']}")
    print(f"  saved to: {output_dir}")


if __name__ == "__main__":
    main()
