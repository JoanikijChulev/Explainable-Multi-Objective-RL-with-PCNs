import argparse
import csv
import json
import math
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import evaluate_cw_vs_zoo_paper_experiments as paper
import evaluate_pcn_zoo_counterfactuals as zoo_batch
import interactive_pcn_zoo_cf as cf
from device_utils import preferred_device
from paper_plot_utils import save_paper_figure


DEFAULT_RUNS = ["bp", "c2", "dst", "Minecart", "reward_line", "bb", "rsg2", "ft"]
METHOD_COLORS = {
    "original": "#FFFFFF",
    "ours": "#00A6D6",
    "cw": "#E24A33",
    "front": "#009E73",
    "boundary": "#B00020",
}


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


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def dim_pairs(reward_dim):
    return [(i, j) for i in range(int(reward_dim)) for j in range(i + 1, int(reward_dim))]


def command_bounds_for_dims(run_context, dims):
    low, high = cf.command_bounds(run_context["setup"])
    dim_x, dim_y = int(dims[0]), int(dims[1])
    return (
        float(low[dim_x]),
        float(high[dim_x]),
        float(low[dim_y]),
        float(high[dim_y]),
    )


def finite_span(lo, hi):
    span = float(hi) - float(lo)
    return span if np.isfinite(span) and abs(span) > 1e-9 else 1.0


def heatmap_arrays_window(run_context, case, dims, x_values, y_values, eval_settings):
    base = np.asarray(case["remaining_command"], dtype=np.float32)
    dim_x, dim_y = int(dims[0]), int(dims[1])
    margins = np.zeros((len(y_values), len(x_values)), dtype=np.float32)
    foil_probs = np.zeros_like(margins)
    greedy = np.zeros_like(margins, dtype=np.int32)

    for yi, y_value in enumerate(y_values):
        for xi, x_value in enumerate(x_values):
            command = base.copy()
            command[dim_x] = np.float32(x_value)
            command[dim_y] = np.float32(y_value)
            result = paper.eval_command(run_context, case, command, eval_settings)
            margins[yi, xi] = float(result.target_margin)
            foil_probs[yi, xi] = float(result.probs[int(case["foil_action"])])
            greedy[yi, xi] = int(result.greedy_action)
    return margins, foil_probs, greedy


def boundary_cell_centers(x_values, y_values, margins, threshold):
    signed = np.asarray(margins, dtype=np.float32) - float(threshold)
    centers = []
    for yi in range(signed.shape[0] - 1):
        for xi in range(signed.shape[1] - 1):
            cell = signed[yi: yi + 2, xi: xi + 2]
            if float(np.nanmin(cell)) <= 0.0 <= float(np.nanmax(cell)):
                centers.append(
                    (
                        float((x_values[xi] + x_values[xi + 1]) / 2.0),
                        float((y_values[yi] + y_values[yi + 1]) / 2.0),
                    )
                )
    return np.asarray(centers, dtype=np.float32).reshape(-1, 2)


def landscape_screen_score(x_values, y_values, margins, threshold):
    signed = np.asarray(margins, dtype=np.float32) - float(threshold)
    min_value = float(np.nanmin(signed))
    max_value = float(np.nanmax(signed))
    has_boundary = min_value <= 0.0 <= max_value
    success_fraction = float(np.mean(signed >= 0.0))
    balance = 1.0 - min(abs(success_fraction - 0.5) * 2.0, 1.0)
    contrast = math.tanh((max_value - min_value) / 3.0)

    boundary_points = boundary_cell_centers(x_values, y_values, margins, threshold)
    centrality = 0.0
    if len(boundary_points) > 0:
        x_span = finite_span(x_values[0], x_values[-1])
        y_span = finite_span(y_values[0], y_values[-1])
        normalized = np.column_stack(
            (
                (boundary_points[:, 0] - float(x_values[0])) / x_span,
                (boundary_points[:, 1] - float(y_values[0])) / y_span,
            )
        )
        distances = np.linalg.norm(normalized - np.asarray([0.5, 0.5]), axis=1)
        centrality = 1.0 - min(float(np.min(distances)) / math.sqrt(0.5), 1.0)

    score = 3.0 * centrality + 1.5 * balance + 0.75 * contrast
    if has_boundary:
        score += 2.0
    return {
        "score": float(score),
        "has_boundary": bool(has_boundary),
        "centrality": float(centrality),
        "balance": float(balance),
        "contrast": float(contrast),
        "success_fraction": float(success_fraction),
        "boundary_count": int(len(boundary_points)),
        "min_signed_margin": float(min_value),
        "max_signed_margin": float(max_value),
    }


def projected_distance(run_context, case, dims, command_a, command_b):
    low, high = cf.command_bounds(run_context["setup"])
    dim_x, dim_y = int(dims[0]), int(dims[1])
    scale = np.asarray([high[dim_x] - low[dim_x], high[dim_y] - low[dim_y]], dtype=np.float32)
    scale[~np.isfinite(scale) | (np.abs(scale) < 1e-9)] = 1.0
    point_a = np.asarray(command_a, dtype=np.float32)[[dim_x, dim_y]]
    point_b = np.asarray(command_b, dtype=np.float32)[[dim_x, dim_y]]
    return float(np.linalg.norm((point_a - point_b) / scale, ord=2))


def projected_point(run_context, dims, command):
    dim_x, dim_y = int(dims[0]), int(dims[1])
    command = np.asarray(command, dtype=np.float32)
    return float(command[dim_x]), float(command[dim_y])


def showcase_score(run_context, case, dims, screen, ours_bundle, cw_bundle, front_result):
    base = np.asarray(case["remaining_command"], dtype=np.float32)
    ours_result = None if ours_bundle is None else ours_bundle["final_result"]
    cw_result = None if cw_bundle is None else cw_bundle["final_result"]

    score = float(screen["score"])
    if ours_result is not None and ours_result.success:
        score += 5.0
        score += min(projected_distance(run_context, case, dims, base, ours_result.command) * 5.0, 2.0)
    else:
        score -= 6.0

    if cw_result is not None:
        if not cw_result.success:
            score += 2.5
        if ours_result is not None:
            score += min(projected_distance(run_context, case, dims, cw_result.command, ours_result.command) * 4.0, 2.0)
        score += min(projected_distance(run_context, case, dims, base, cw_result.command) * 2.0, 1.0)
    else:
        score -= 1.0

    if front_result is not None and front_result.success:
        score += 0.75

    if screen["success_fraction"] < 0.12 or screen["success_fraction"] > 0.88:
        score -= 1.0
    return float(score)


def screen_run_candidates(run_context, args, eval_settings):
    cases = []
    for case in zoo_batch.rollout_cases(
        run_context,
        max_fronts=args.max_fronts,
        max_timesteps=args.max_timesteps,
        foil_top_k=args.foil_top_k,
    ):
        cases.append(case)
        if len(cases) >= int(args.max_candidates_per_run):
            break

    scored = []
    for case in cases:
        original = paper.eval_command(
            run_context,
            case,
            case["remaining_command"],
            eval_settings,
        )
        if original.target_margin >= eval_settings.margin:
            continue

        for dims in dim_pairs(run_context["reward_dim"]):
            x_low, x_high, y_low, y_high = command_bounds_for_dims(run_context, dims)
            x_values = np.linspace(x_low, x_high, int(args.score_grid), dtype=np.float32)
            y_values = np.linspace(y_low, y_high, int(args.score_grid), dtype=np.float32)
            margins, _, _ = heatmap_arrays_window(run_context, case, dims, x_values, y_values, eval_settings)
            screen = landscape_screen_score(x_values, y_values, margins, eval_settings.margin)
            if screen["has_boundary"] or screen["score"] >= float(args.min_screen_score):
                scored.append(
                    {
                        "case": case,
                        "dims": dims,
                        "screen": screen,
                        "original_result": original,
                    }
                )

    scored.sort(key=lambda item: item["screen"]["score"], reverse=True)
    return scored[: int(args.shortlist)]


def display_window(run_context, case, dims, x_values, y_values, margins, threshold, points, padding_fraction):
    x_low, x_high, y_low, y_high = command_bounds_for_dims(run_context, dims)
    dim_x, dim_y = int(dims[0]), int(dims[1])
    full_x_span = finite_span(x_low, x_high)
    full_y_span = finite_span(y_low, y_high)
    boundary_points = boundary_cell_centers(x_values, y_values, margins, threshold)

    finite_points = []
    for point in points:
        if point is None:
            continue
        point = np.asarray(point, dtype=np.float32)
        if point.shape[0] <= max(dim_x, dim_y):
            continue
        finite_points.append((float(point[dim_x]), float(point[dim_y])))

    if boundary_points.size > 0 and finite_points:
        point_center = np.mean(np.asarray(finite_points, dtype=np.float32), axis=0)
        normalized_boundary = boundary_points.copy()
        normalized_boundary[:, 0] = (normalized_boundary[:, 0] - x_low) / full_x_span
        normalized_boundary[:, 1] = (normalized_boundary[:, 1] - y_low) / full_y_span
        normalized_center = np.asarray([(point_center[0] - x_low) / full_x_span, (point_center[1] - y_low) / full_y_span])
        boundary = boundary_points[int(np.argmin(np.linalg.norm(normalized_boundary - normalized_center, axis=1)))]
        center_x, center_y = float(boundary[0]), float(boundary[1])
    elif finite_points:
        center_x, center_y = np.mean(np.asarray(finite_points, dtype=np.float32), axis=0).tolist()
    else:
        center_x, center_y = (x_low + x_high) / 2.0, (y_low + y_high) / 2.0

    if finite_points:
        points_arr = np.asarray(finite_points, dtype=np.float32)
        half_x = max(float(np.max(np.abs(points_arr[:, 0] - center_x))), 0.18 * full_x_span)
        half_y = max(float(np.max(np.abs(points_arr[:, 1] - center_y))), 0.18 * full_y_span)
    else:
        half_x, half_y = 0.35 * full_x_span, 0.35 * full_y_span

    half_x += float(padding_fraction) * full_x_span
    half_y += float(padding_fraction) * full_y_span
    half_x = min(max(half_x, 0.15 * full_x_span), 0.5 * full_x_span)
    half_y = min(max(half_y, 0.15 * full_y_span), 0.5 * full_y_span)

    x0, x1 = center_x - half_x, center_x + half_x
    y0, y1 = center_y - half_y, center_y + half_y
    if x0 < x_low:
        x1 += x_low - x0
        x0 = x_low
    if x1 > x_high:
        x0 -= x1 - x_high
        x1 = x_high
    if y0 < y_low:
        y1 += y_low - y0
        y0 = y_low
    if y1 > y_high:
        y0 -= y1 - y_high
        y1 = y_high
    x0, x1 = max(x_low, x0), min(x_high, x1)
    y0, y1 = max(y_low, y0), min(y_high, y1)
    return float(x0), float(x1), float(y0), float(y1)


def draw_landscape_axis(
    ax,
    run_context,
    case,
    dims,
    x_values,
    y_values,
    margins,
    threshold,
    ours_result,
    cw_result,
    front_result,
    title,
    show_legend=True,
):
    dim_x, dim_y = int(dims[0]), int(dims[1])
    vmax = max(
        abs(float(np.nanmin(margins))),
        abs(float(np.nanmax(margins))),
        abs(float(threshold)),
        0.25,
    )
    image = ax.imshow(
        margins,
        origin="lower",
        extent=[float(x_values[0]), float(x_values[-1]), float(y_values[0]), float(y_values[-1])],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="bicubic",
    )
    max_margin = float(np.nanmax(margins))
    if np.isfinite(max_margin) and max_margin > float(threshold):
        ax.contourf(
            x_values,
            y_values,
            margins,
            levels=[float(threshold), max_margin + 1e-6],
            colors=[METHOD_COLORS["boundary"]],
            alpha=0.12,
        )
    try:
        ax.contour(
            x_values,
            y_values,
            margins,
            levels=[float(threshold)],
            colors=METHOD_COLORS["boundary"],
            linewidths=2.8,
        )
    except ValueError:
        pass

    base = np.asarray(case["remaining_command"], dtype=np.float32)
    ax.scatter(
        base[dim_x],
        base[dim_y],
        s=95,
        marker="o",
        c=METHOD_COLORS["original"],
        edgecolors="#111111",
        linewidths=1.0,
        label="Original",
        zorder=7,
        clip_on=False,
    )

    if cw_result is not None:
        ax.scatter(
            cw_result.command[dim_x],
            cw_result.command[dim_y],
            s=105,
            marker="X",
            c=METHOD_COLORS["cw"],
            edgecolors="#111111",
            linewidths=0.8,
            label="C&W",
            zorder=8,
            clip_on=False,
        )

    if ours_result is not None:
        ax.scatter(
            ours_result.command[dim_x],
            ours_result.command[dim_y],
            s=175,
            marker="*",
            c=METHOD_COLORS["ours"],
            edgecolors="#111111",
            linewidths=0.8,
            label="Ours",
            zorder=9,
            clip_on=False,
        )
        if ours_result.success:
            ax.scatter(
                ours_result.command[dim_x],
                ours_result.command[dim_y],
                s=260,
                marker="*",
                facecolors="none",
                edgecolors=METHOD_COLORS["boundary"],
                linewidths=2.0,
                zorder=9.5,
                clip_on=False,
            )

    if front_result is not None and front_result.success:
        ax.scatter(
            front_result.command[dim_x],
            front_result.command[dim_y],
            s=70,
            marker="D",
            c=METHOD_COLORS["front"],
            edgecolors="#111111",
            linewidths=0.7,
            label="Front",
            zorder=6,
            clip_on=False,
        )

    draw_gradient_arrow(
        ax,
        run_context,
        case,
        dims,
        None if cw_result is None else cw_result.command,
        x_values,
        y_values,
        label="C&W grad.",
        color=METHOD_COLORS["cw"],
        zorder=10,
    )
    draw_gradient_arrow(
        ax,
        run_context,
        case,
        dims,
        None if ours_result is None else ours_result.command,
        x_values,
        y_values,
        label="Ours grad.",
        color="#FFD447",
        zorder=11,
    )

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel(f"Command $R_{{{dim_x}}}$")
    ax.set_ylabel(f"Command $R_{{{dim_y}}}$")
    ax.grid(color="white", alpha=0.16, linewidth=0.7)
    x_span = finite_span(x_values[0], x_values[-1])
    y_span = finite_span(y_values[0], y_values[-1])
    ax.set_xlim(float(x_values[0]) - 0.025 * x_span, float(x_values[-1]) + 0.025 * x_span)
    ax.set_ylim(float(y_values[0]) - 0.025 * y_span, float(y_values[-1]) + 0.025 * y_span)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        boundary_handle = Line2D(
            [0],
            [0],
            color=METHOD_COLORS["boundary"],
            linewidth=2.8,
            label="Decision boundary",
        )
        ax.legend(
            [boundary_handle, *handles],
            ["Decision boundary", *labels],
            loc="best",
            fontsize=9,
            frameon=True,
            framealpha=0.9,
        )
    return image


def draw_gradient_arrow(ax, run_context, case, dims, command, x_values, y_values, label, color, zorder):
    if command is None:
        return
    dim_x, dim_y = int(dims[0]), int(dims[1])
    command = np.asarray(command, dtype=np.float32)
    grad = paper.command_margin_gradient(run_context, case, command)
    grad_xy = np.asarray([grad[dim_x], grad[dim_y]], dtype=np.float32)
    grad_norm = float(np.linalg.norm(grad_xy, ord=2))
    if not np.isfinite(grad_norm) or grad_norm <= 1e-9:
        return
    x_span = finite_span(x_values[0], x_values[-1])
    y_span = finite_span(y_values[0], y_values[-1])
    arrow_scale = 0.12 * min(x_span, y_span) / grad_norm
    ax.arrow(
        float(command[dim_x]),
        float(command[dim_y]),
        float(grad_xy[0] * arrow_scale),
        float(grad_xy[1] * arrow_scale),
        width=0.007 * min(x_span, y_span),
        facecolor=color,
        edgecolor="#111111",
        linewidth=0.55,
        length_includes_head=True,
        label=label,
        zorder=zorder,
        clip_on=False,
    )


def plot_individual_showcase(output_dir, selected, final_grid, eval_settings):
    run_context = selected["run_context"]
    case = selected["case"]
    dims = selected["dims"]
    ours_result = selected["ours_result"]
    cw_result = selected["cw_result"]
    front_result = selected["front_result"]
    window = selected["window"]
    x_values = np.linspace(window[0], window[1], int(final_grid), dtype=np.float32)
    y_values = np.linspace(window[2], window[3], int(final_grid), dtype=np.float32)
    margins, foil_probs, greedy = heatmap_arrays_window(run_context, case, dims, x_values, y_values, eval_settings)
    selected["final_arrays"] = {
        "x_values": x_values,
        "y_values": y_values,
        "margins": margins,
        "foil_probs": foil_probs,
        "greedy": greedy,
    }

    name = selected["name"]
    env_title = (
        f"{run_context['run_name']}: "
        f"{cf.action_label(run_context['env'], int(case['policy']['action']))} -> "
        f"{cf.action_label(run_context['env'], int(case['foil_action']))} "
        f"(front {case['front_index']}, t={case['timestep']})"
    )
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    image = draw_landscape_axis(
        ax,
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
        env_title,
        show_legend=True,
    )
    cbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.025)
    cbar.set_label(r"Target margin $m(R)$")
    save_paper_figure(fig, Path(output_dir) / f"{name}_showcase.png")
    plt.close(fig)

    np.savez_compressed(
        Path(output_dir) / f"{name}_arrays.npz",
        x_values=x_values,
        y_values=y_values,
        margins=margins,
        foil_probs=foil_probs,
        greedy=greedy,
        dims=np.asarray(dims, dtype=np.int64),
    )


def plot_combined_grid(output_dir, selections, eval_settings):
    if not selections:
        return
    rows = 2 if len(selections) > 3 else 1
    cols = int(math.ceil(len(selections) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows))
    axes = np.asarray(axes).reshape(-1)
    image = None
    for ax, selected in zip(axes, selections):
        arrays = selected["final_arrays"]
        run_context = selected["run_context"]
        case = selected["case"]
        title = (
            f"{run_context['run_name']}: "
            f"{cf.action_label(run_context['env'], int(case['policy']['action']))} -> "
            f"{cf.action_label(run_context['env'], int(case['foil_action']))}"
        )
        image = draw_landscape_axis(
            ax,
            run_context,
            case,
            selected["dims"],
            arrays["x_values"],
            arrays["y_values"],
            arrays["margins"],
            eval_settings.margin,
            selected["ours_result"],
            selected["cw_result"],
            selected["front_result"],
            title,
            show_legend=False,
        )
    for ax in axes[len(selections):]:
        ax.axis("off")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.82, pad=0.018)
        cbar.set_label(r"Target margin $m(R)$")
    handles, labels = axes[0].get_legend_handles_labels()
    boundary_handle = Line2D(
        [0],
        [0],
        color=METHOD_COLORS["boundary"],
        linewidth=2.8,
        label="Decision boundary",
    )
    if handles:
        fig.legend(
            [boundary_handle, *handles],
            ["Decision boundary", *labels],
            loc="lower center",
            ncol=min(len(labels) + 1, 5),
            frameon=True,
            fontsize=9,
        )
    else:
        fig.legend([boundary_handle], ["Decision boundary"], loc="lower center", frameon=True, fontsize=9)
    fig.suptitle("Command-Space Decision Landscapes", fontsize=10, y=0.98)
    save_paper_figure(fig, Path(output_dir) / "showcase_landscapes_grid.png")
    plt.close(fig)


def row_for_selection(selected):
    run_context = selected["run_context"]
    case = selected["case"]
    ours_result = selected["ours_result"]
    cw_result = selected["cw_result"]
    front_result = selected["front_result"]
    return {
        "run": run_context["run_name"],
        "env": run_context["env_name"],
        "checkpoint": run_context["checkpoint_path"].name,
        "front_index": int(case["front_index"]),
        "timestep": int(case["timestep"]),
        "foil_action": int(case["foil_action"]),
        "foil_label": cf.action_label(run_context["env"], int(case["foil_action"])),
        "original_action": int(case["policy"]["action"]),
        "original_label": cf.action_label(run_context["env"], int(case["policy"]["action"])),
        "dims": json.dumps([int(x) for x in selected["dims"]]),
        "window": json.dumps([float(x) for x in selected["window"]]),
        "desired_return": json_vector(case["desired_return"]),
        "remaining_command": json_vector(case["remaining_command"]),
        "observation_summary": json.dumps(cf.observation_summary(case["obs"])),
        "score": maybe_float(selected["score"]),
        "screen_score": maybe_float(selected["screen"]["score"]),
        "screen_centrality": maybe_float(selected["screen"]["centrality"]),
        "screen_balance": maybe_float(selected["screen"]["balance"]),
        "screen_success_fraction": maybe_float(selected["screen"]["success_fraction"]),
        "ours_success": "true" if ours_result is not None and ours_result.success else "false",
        "ours_scaled_l2": maybe_float(paper.scaled_distance_for_result(run_context, case, ours_result)),
        "ours_target_margin": maybe_float(None if ours_result is None else ours_result.target_margin),
        "ours_command": json_vector(None if ours_result is None else ours_result.command),
        "ours_total_queries": "" if selected["ours_bundle"] is None else int(selected["ours_bundle"]["total_queries"]),
        "ours_ray_success": "" if selected["ours_bundle"] is None else ("true" if selected["ours_bundle"]["seed_result"] is not None and selected["ours_bundle"]["seed_result"].success else "false"),
        "ours_zoo_success": "" if selected["ours_bundle"] is None else ("true" if selected["ours_bundle"]["selected_result"] is not None and selected["ours_bundle"]["selected_result"].success else "false"),
        "ours_zoo_improved_seed": "" if selected["ours_bundle"] is None else ("true" if selected["ours_bundle"]["zoo_improved_seed"] else "false"),
        "ours_binary_improved_selected": "" if selected["ours_bundle"] is None else ("true" if selected["ours_bundle"]["binary_improved_selected"] else "false"),
        "cw_success": "true" if cw_result is not None and cw_result.success else "false",
        "cw_scaled_l2": maybe_float(paper.scaled_distance_for_result(run_context, case, cw_result)),
        "cw_target_margin": maybe_float(None if cw_result is None else cw_result.target_margin),
        "cw_command": json_vector(None if cw_result is None else cw_result.command),
        "front_success": "true" if front_result is not None and front_result.success else "false",
        "front_command": json_vector(None if front_result is None else front_result.command),
    }


def select_showcase_for_run(run_name, args, zoo_settings, cw_settings, eval_settings, device):
    run_context = paper.load_run_by_name(run_name, args.checkpoint, device)
    shortlisted = screen_run_candidates(run_context, args, eval_settings)
    if not shortlisted:
        raise ValueError(f"no landscape candidates found for {run_name}")

    attack_cache = {}
    best = None
    evaluated = []
    for index, item in enumerate(shortlisted, start=1):
        case = item["case"]
        key = (int(case["front_index"]), int(case["timestep"]), int(case["foil_action"]))
        if key not in attack_cache:
            ours_bundle, ours_error = paper.run_ours(run_context, case, zoo_settings)
            cw_bundle = paper.run_cw_case(run_context, case, cw_settings, eval_settings)
            front_result = paper.known_front_success(run_context, case, eval_settings)
            attack_cache[key] = (ours_bundle, ours_error, cw_bundle, front_result)
        else:
            ours_bundle, ours_error, cw_bundle, front_result = attack_cache[key]

        score = showcase_score(
            run_context,
            case,
            item["dims"],
            item["screen"],
            ours_bundle,
            cw_bundle,
            front_result,
        )
        candidate = {
            "run_context": run_context,
            "case": case,
            "dims": item["dims"],
            "screen": item["screen"],
            "score": score,
            "ours_bundle": ours_bundle,
            "ours_error": ours_error,
            "ours_result": None if ours_bundle is None else ours_bundle["final_result"],
            "cw_bundle": cw_bundle,
            "cw_result": cw_bundle["final_result"],
            "front_result": front_result,
            "name": f"{run_context['run_name']}_front{case['front_index']}_t{case['timestep']}_foil{case['foil_action']}",
        }
        evaluated.append(candidate)
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    successful = [
        candidate
        for candidate in evaluated
        if candidate["ours_result"] is not None and candidate["ours_result"].success
    ]
    if successful:
        best = max(successful, key=lambda candidate: candidate["score"])
    elif evaluated:
        best = max(evaluated, key=lambda candidate: candidate["score"])

    if best is None:
        raise ValueError(f"could not select showcase candidate for {run_name}")

    x_low, x_high, y_low, y_high = command_bounds_for_dims(run_context, best["dims"])
    coarse_x = np.linspace(x_low, x_high, int(args.score_grid), dtype=np.float32)
    coarse_y = np.linspace(y_low, y_high, int(args.score_grid), dtype=np.float32)
    coarse_margins, _, _ = heatmap_arrays_window(
        run_context,
        best["case"],
        best["dims"],
        coarse_x,
        coarse_y,
        eval_settings,
    )
    points = [
        best["case"]["remaining_command"],
        None if best["ours_result"] is None else best["ours_result"].command,
        None if best["cw_result"] is None else best["cw_result"].command,
        None if best["front_result"] is None else best["front_result"].command,
    ]
    best["window"] = display_window(
        run_context,
        best["case"],
        best["dims"],
        coarse_x,
        coarse_y,
        coarse_margins,
        eval_settings.margin,
        points,
        args.window_padding_fraction,
    )
    return best


def close_selections(selections):
    seen = set()
    for selected in selections:
        run_context = selected.get("run_context")
        if run_context is None:
            continue
        key = id(run_context)
        if key in seen:
            continue
        seen.add(key)
        paper.close_run(run_context)


def save_config(output_dir, args, zoo_settings, cw_settings, eval_settings):
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "This script curates real cases by visual-selection criteria. "
            "It does not alter model outputs, but it is intentionally selection-biased."
        ),
        "args": vars(args),
        "zoo_settings": asdict(zoo_settings),
        "cw_settings": asdict(cw_settings),
        "eval_settings": asdict(eval_settings),
    }
    with (Path(output_dir) / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate curated paper showcase landscapes for PCN command-space counterfactuals."
    )
    parser.add_argument("--label", default="showcase_landscapes")
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-fronts", type=int, default=5)
    parser.add_argument("--max-timesteps", type=int, default=4)
    parser.add_argument("--foil-top-k", type=int, default=2)
    parser.add_argument("--max-candidates-per-run", type=int, default=36)
    parser.add_argument("--shortlist", type=int, default=4)
    parser.add_argument("--score-grid", type=int, default=17)
    parser.add_argument("--final-grid", type=int, default=121)
    parser.add_argument("--min-screen-score", type=float, default=2.0)
    parser.add_argument("--window-padding-fraction", type=float, default=0.06)
    parser.add_argument("--success-margin", type=float, default=cf.DEFAULT_ZOO_SETTINGS["margin"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-clip-command", action="store_true")

    parser.add_argument("--zoo-max-queries", type=int, default=6000)
    parser.add_argument("--zoo-random-directions", type=int, default=1000)
    parser.add_argument("--zoo-line-search-steps", type=int, default=16)
    parser.add_argument("--zoo-learning-rate", type=float, default=cf.DEFAULT_ZOO_SETTINGS["learning_rate"])
    parser.add_argument("--zoo-finite-diff-h", type=float, default=cf.DEFAULT_ZOO_SETTINGS["finite_diff_h"])
    parser.add_argument("--zoo-c", type=float, default=cf.DEFAULT_ZOO_SETTINGS["c"])

    parser.add_argument("--cw-confidence", type=float, default=0.0)
    parser.add_argument("--cw-learning-rate", type=float, default=0.001)
    parser.add_argument("--cw-binary-search-steps", type=int, default=2)
    parser.add_argument("--cw-max-iter", type=int, default=150)
    parser.add_argument("--cw-initial-const", type=float, default=1.0)
    parser.add_argument("--cw-max-halving", type=int, default=5)
    parser.add_argument("--cw-max-doubling", type=int, default=5)
    parser.add_argument("--cw-batch-size", type=int, default=1)
    parser.add_argument("--cw-art-clip-relaxation", type=float, default=1e-2)
    parser.add_argument("--cw-verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output_artifacts") / "showcase_landscapes" / f"{args.label}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    device = preferred_device()
    zoo_settings = paper.make_zoo_settings(args)
    cw_settings = paper.make_cw_settings(args)
    eval_settings = paper.make_eval_settings(args.success_margin, clip_command=not args.no_clip_command)
    save_config(output_dir, args, zoo_settings, cw_settings, eval_settings)

    print("=" * 72)
    print("Curated PCN command-space showcase landscapes")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Output: {output_dir.resolve()}")
    print("Selection is intentionally biased toward visually clear real cases.")
    print()

    selections = []
    rows = []
    start = time.perf_counter()
    try:
        for run_name in args.runs:
            print(f"Selecting showcase case for {run_name}...")
            try:
                selected = select_showcase_for_run(
                    run_name,
                    args,
                    zoo_settings,
                    cw_settings,
                    eval_settings,
                    device,
                )
                plot_individual_showcase(output_dir, selected, args.final_grid, eval_settings)
                selections.append(selected)
                row = row_for_selection(selected)
                rows.append(row)
                print(
                    f"  selected front={row['front_index']} timestep={row['timestep']} "
                    f"foil={row['foil_action']} dims={row['dims']} "
                    f"score={float(row['score']):.3f} ours={row['ours_success']} cw={row['cw_success']}"
                )
                print(
                    f"  CF state: desired_return={row['desired_return']} "
                    f"remaining_command={row['remaining_command']} "
                    f"observation={row['observation_summary']}"
                )
            except Exception as exc:
                rows.append(
                    {
                        "run": run_name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=5),
                    }
                )
                print(f"  skipped {run_name}: {type(exc).__name__}: {exc}")

        if selections:
            plot_combined_grid(output_dir, selections, eval_settings)

        write_csv(output_dir / "showcase_selected_cases.csv", rows)
        with (output_dir / "showcase_selected_cases.json").open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
    finally:
        close_selections(selections)

    print(f"\nCompleted in {time.perf_counter() - start:.1f}s")
    print(f"Saved showcase artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
