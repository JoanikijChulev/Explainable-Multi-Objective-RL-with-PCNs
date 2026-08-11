"""Aggregate the cross-env goal-influence evaluation into summary CSVs and plots.

Reads the per-env outputs of goal_influence/evaluate_goal_influence_envs.py and writes:
  cross_env_summary.csv   one row per env x score with faithfulness + basis stats
  states_all.csv          pooled per-state scores + ground truth for all envs
  plots/                  faithfulness bars, AUROC bars, flip calibration scatter,
                          state-basis fork separation

The input and output root is goal_influence/goal_output_artifacts/.

Usage:
    python goal_influence/aggregate_goal_influence_results.py --out-label comprehensive
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from goal_influence import goal_output_artifacts_dir
from paper_plot_utils import apply_bar_hatches, save_paper_figure

SCORES = ("kl", "tv", "js", "flip")
SCORE_LABELS = {"kl": "KL", "tv": "TV", "js": "JS", "flip": "probe flip%"}
# categorical slots in fixed order (validated palette, light mode)
SCORE_COLORS = {"kl": "#2a78d6", "tv": "#1baf7a", "js": "#eda100", "flip": "#008300"}
OBJ_COLORS = {2: "#2a78d6", 3: "#1baf7a", 4: "#eda100", 6: "#008300"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(ax, envs, values_by_score, ylabel, title):
    x = np.arange(len(envs))
    width = 0.19
    for i, s in enumerate(SCORES):
        vals = [v if np.isfinite(v) else 0.0 for v in values_by_score[s]]
        bars = ax.bar(x + (i - 1.5) * width, vals, width * 0.92,
                      label=SCORE_LABELS[s], color=SCORE_COLORS[s],
                      edgecolor=SURFACE, linewidth=1.0)
        apply_bar_hatches(bars)
        for j, v in enumerate(values_by_score[s]):
            if not np.isfinite(v):
                ax.text(x[j] + (i - 1.5) * width, 0.02, "n/a", rotation=90,
                        ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=20, ha="right", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, fontsize=11)
    ax.legend(fontsize=8, framealpha=0.9, ncol=4, loc="lower right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-label", default="comprehensive")
    args = ap.parse_args()

    out_dir = goal_output_artifacts_dir() / args.out_label
    per_env = out_dir / "per_env"
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    summaries = []
    for p in sorted(per_env.glob("*_summary.json")):
        with p.open(encoding="utf-8") as h:
            s = json.load(h)
        if "failed" not in s:
            summaries.append(s)
    # order: by objective count, then by name (stable, readable axis)
    summaries.sort(key=lambda s: (s["n_objectives"], s["env"]))
    envs = [s["env"] for s in summaries]

    # ---- cross_env_summary.csv (one row per env x score) ----
    rows = []
    for s in summaries:
        for sc in SCORES:
            f = s["faithfulness"][sc]
            rows.append({
                "env": s["env"], "n_objectives": s["n_objectives"], "n_actions": s["n_actions"],
                "n_states": s["n_states"], "n_unique_states": s["n_unique_states"],
                "obs_repeat_ratio": s["obs_repeat_ratio"], "gt_method": s["gt_method"],
                "frac_goal_active": s["frac_goal_active"], "probes_per_state": s["probes_per_state"],
                "score": sc,
                "spearman_flip_fraction": f["spearman_flip_fraction"],
                "auroc_detect_goal_active": f["auroc_detect_goal_active"],
                "spearman_nearest_flip": f["spearman_nearest_flip"],
                "mean_I": f["mean_I"], "median_I": f["median_I"],
                "basis_n_multi": s["state_basis"]["n_multi_visit"],
                "basis_n_forks": s["state_basis"]["n_forks"],
                "basis_mi_forks": s["state_basis"]["mean_mi_forks"],
                "basis_mi_nonfork": s["state_basis"]["mean_mi_nonfork_multi"],
                "elapsed_s": s["elapsed_s"],
            })
    with (out_dir / "cross_env_summary.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- pooled states_all.csv ----
    pooled = []
    for name in envs:
        with (per_env / f"{name}_states.csv").open(encoding="utf-8") as h:
            pooled.extend(list(csv.DictReader(h)))
    with (out_dir / "states_all.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(pooled[0].keys())); w.writeheader(); w.writerows(pooled)

    # ---- Fig 1: Spearman(I, flip-fraction) ----
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    vals = {sc: [s["faithfulness"][sc]["spearman_flip_fraction"] for s in summaries] for sc in SCORES}
    grouped_bars(ax, envs, vals, "Spearman(I, flip-fraction)",
                 "Faithfulness: does the score rank states by real command influence?  (higher = better)")
    ax.set_ylim(-0.1, 1.05)
    fig.tight_layout(); save_paper_figure(fig, plots / "faithfulness_spearman.png"); plt.close(fig)

    # ---- Fig 2: AUROC (only defined where both classes exist) ----
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    vals = {sc: [s["faithfulness"][sc]["auroc_detect_goal_active"] for s in summaries] for sc in SCORES}
    grouped_bars(ax, envs, vals, "AUROC (detect goal-active states)",
                 "Detection: separating goal-active from goal-inert states  (0.5 = chance)")
    ax.axhline(0.5, color=MUTED, linewidth=1.0, linestyle="--")
    ax.set_ylim(0, 1.05)
    fig.tight_layout(); save_paper_figure(fig, plots / "detection_auroc.png"); plt.close(fig)

    # ---- Fig 3: probe flip% calibration scatter ----
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    obj_of = {s["env"]: s["n_objectives"] for s in summaries}
    seen = set()
    for r in pooled:
        d = obj_of[r["env"]]
        lbl = f"{d}-objective" if d not in seen else None
        seen.add(d)
        ax.scatter(float(r["I_flip"]), float(r["gt_flip_fraction"]), s=22,
                   color=OBJ_COLORS[d], alpha=0.55, edgecolors=SURFACE, linewidths=0.5, label=lbl)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--")
    ax.text(0.83, 0.88, "y = x", color=MUTED, fontsize=8)
    ax.set_xlabel("probe flip%  (free, ~100-3800 fan probes)", color=INK2)
    ax.set_ylabel("ground-truth flip-fraction  (exhaustive / MC scan)", color=INK2)
    ax.set_title("The free probe flip% tracks the exhaustive command-box scan", color=INK, fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    order = np.argsort([int(l.split("-")[0]) for l in labels])
    ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=8, loc="lower right")
    fig.tight_layout(); save_paper_figure(fig, plots / "flip_calibration.png"); plt.close(fig)

    # ---- Fig 4: state-basis fork separation ----
    with_multi = [s for s in summaries if s["state_basis"]["n_multi_visit"] > 0]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    x = np.arange(len(with_multi))
    width = 0.36
    fk = [s["state_basis"]["mean_mi_forks"] for s in with_multi]
    nf = [s["state_basis"]["mean_mi_nonfork_multi"] for s in with_multi]
    b1 = ax.bar(x - width / 2, fk, width * 0.92, label="fork states (action varies with goal)",
                color="#2a78d6", edgecolor=SURFACE, linewidth=1.0)
    b2 = ax.bar(x + width / 2, nf, width * 0.92, label="non-fork states (same action for all goals)",
                color="#1baf7a", edgecolor=SURFACE, linewidth=1.0)
    apply_bar_hatches(b1); apply_bar_hatches(b2)
    for xi, s in zip(x, with_multi):
        ax.text(xi, max(s["state_basis"]["mean_mi_forks"], 0.02) + 0.02,
                f"{s['state_basis']['n_forks']}/{s['state_basis']['n_multi_visit']}",
                ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([s["env"] for s in with_multi], rotation=20, ha="right", color=INK2)
    ax.set_ylabel("state-basis MI to reachable centroid (nats)", color=INK2)
    ax.set_title("State basis separates goal-forks from state-forced cells  (label = forks / multi-visit cells)",
                 color=INK, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); save_paper_figure(fig, plots / "state_basis_forks.png"); plt.close(fig)

    print(f"Aggregated {len(summaries)} envs -> {out_dir}")
    print("  cross_env_summary.csv, states_all.csv, plots/*.png")


if __name__ == "__main__":
    main()
