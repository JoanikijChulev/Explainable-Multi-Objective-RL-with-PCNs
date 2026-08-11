"""Comprehensive cross-environment evaluation of the goal-influence method.

For every trained run it:
  1. rolls the greedy policy for a (subsampled) set of Pareto-front commands,
  2. scores every visited state with the four influence signals
     (KL / TV / JS to the broad-generic baseline, plus the free probe flip%),
  3. computes the behavioral ground truth over the command box
     (exhaustive grid when tractable, Monte-Carlo otherwise),
  4. scores faithfulness (Spearman vs flip-fraction, AUROC vs goal-active,
     Spearman vs nearest-flip distance) for each signal,
  5. aggregates the reachable-centroid state basis (goal->action MI per state).

Each env writes <out>/per_env/<env>_states.csv, <env>_state_basis.csv and
<env>_summary.json so the evaluation can be run env-by-env and aggregated later.
The output root is goal_influence/goal_output_artifacts/.

Usage:
    python goal_influence/evaluate_goal_influence_envs.py --envs dst bp reward_line --out-label comprehensive
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

import numpy as np

from goal_influence import goal_influence as gi, goal_output_artifacts_dir
import interactive_pcn_zoo_cf as cf

# Per-run evaluation settings. Ground truth: exhaustive grid at `grid` points per
# dimension for <=3 objectives (100 -> up to 1e6 points/state); >3 objectives use
# 50k Monte-Carlo samples automatically (see goal_influence.grid_ground_truth).
ENV_SETTINGS = {
    "dst":         dict(run="dst",         grid=100, max_fronts=None, max_ts=25),
    "bp":          dict(run="bp",          grid=100, max_fronts=12,   max_ts=25),
    "reward_line": dict(run="reward_line", grid=100, max_fronts=12,   max_ts=25),
    "three_tree":  dict(run="three_tree",  grid=100, max_fronts=None, max_ts=25),
    "fourroom2":   dict(run="fourroom2",   grid=100, max_fronts=12,   max_ts=40),
    "bb":          dict(run="bb",          grid=100, max_fronts=12,   max_ts=40),
    "rsg2":        dict(run="rsg2",        grid=100, max_fronts=12,   max_ts=40),
    "c2":          dict(run="c2",          grid=100, max_fronts=12,   max_ts=40),
    "MC3":         dict(run="MC3",         grid=100, max_fronts=10,   max_ts=20),
    "ft":          dict(run="ft",          grid=100, max_fronts=16,   max_ts=10),
}
SCORES = ("kl", "tv", "js", "flip")


def evaluate_env(name, settings, device, out_dir):
    t0 = time.time()
    run_dir, setup, model, front, ckpt = gi.load_run(settings["run"], None, device)
    bounds = cf.command_bounds(setup)
    env = setup.env
    dim = int(np.asarray(setup.max_return).shape[-1])

    front_indices = list(range(len(front)))
    mf = settings["max_fronts"]
    if mf and len(front_indices) > mf:
        front_indices = sorted({int(round(i)) for i in np.linspace(0, len(front) - 1, mf)})

    all_rows, gts = [], []
    try:
        for fi in front_indices:
            rows = gi.explain_trajectory(env, model, front, fi, device, bounds,
                                         gi.DEFAULT_FRACTIONS, settings["max_ts"])
            for r in rows:
                gt = gi.grid_ground_truth(model, r["obs"], np.asarray(r["command_R_t"], np.float32),
                                          r["action_mask"], bounds, device, settings["grid"])
                r.update({f"gt_{k}": v for k, v in gt.items()})
            all_rows.extend(rows)
    finally:
        try:
            env.close()
        except Exception:
            pass

    # faithfulness per score
    flip_frac = [r["gt_flip_fraction"] for r in all_rows]
    flip_exists = [r["gt_flip_exists"] for r in all_rows]
    nearest = [r["gt_nearest_flip_scaled_l2"] for r in all_rows]
    nf = [(i, n) for i, n in enumerate(nearest) if np.isfinite(n)]
    faith = {}
    for s in SCORES:
        I = [r[f"I_{s}"] for r in all_rows]
        faith[s] = {
            "spearman_flip_fraction": gi.spearman(I, flip_frac),
            "auroc_detect_goal_active": gi.auroc(I, flip_exists),
            "spearman_nearest_flip": gi.spearman([I[i] for i, _ in nf], [n for _, n in nf]),
            "mean_I": float(np.mean(I)), "median_I": float(np.median(I)),
        }

    # reachable-centroid state basis
    cells = defaultdict(list)
    for r in all_rows:
        cells[tuple(np.asarray(r["obs"]).reshape(-1).tolist())].append(r)
    basis_rows = []
    for key, items in cells.items():
        probs = np.asarray([it["action_probs"] for it in items], dtype=np.float64)
        mask = items[0]["action_mask"]
        centroid = probs.mean(axis=0)
        mi = float(np.mean([gi.kl_divergence(p, centroid, mask) for p in probs]))
        tvc = float(np.mean([gi.tv_distance(p, centroid, mask) for p in probs]))
        jsc = float(np.mean([gi.js_divergence(p, centroid, mask) for p in probs]))
        acts = sorted({it["greedy_label"] for it in items})
        basis_rows.append({"state": str(key if len(key) > 1 else key[0]), "n_visits": len(items),
                           "n_distinct_actions": len(acts), "actions": "/".join(acts),
                           "mi_kl": mi, "mi_tv": tvc, "mi_js": jsc,
                           "fronts": json.dumps(sorted({int(it["front_index"]) for it in items}))})
    multi = [b for b in basis_rows if b["n_visits"] > 1]
    forks = [b for b in multi if b["n_distinct_actions"] > 1]

    summary = {
        "env": name, "run": settings["run"], "env_name": setup.env_name, "checkpoint": ckpt.name,
        "n_objectives": dim, "n_actions": int(setup.n_actions),
        "n_fronts_total": int(len(front)), "n_fronts_used": len(front_indices),
        "n_states": len(all_rows), "n_unique_states": len(cells),
        "obs_repeat_ratio": round(1.0 - len(cells) / max(len(all_rows), 1), 3),
        "probes_per_state": int(np.mean([r["queries"] for r in all_rows])),
        "gt_method": all_rows[0]["gt_method"] if "gt_method" in all_rows[0] else all_rows[0]["gt_gt_method"],
        "n_goal_active": int(sum(flip_exists)),
        "frac_goal_active": round(float(np.mean(flip_exists)), 3),
        "faithfulness": faith,
        "state_basis": {
            "n_cells": len(basis_rows), "n_multi_visit": len(multi), "n_forks": len(forks),
            "frac_forks_of_multi": round(len(forks) / max(len(multi), 1), 3),
            "mean_mi_forks": float(np.mean([b["mi_kl"] for b in forks])) if forks else 0.0,
            "mean_mi_nonfork_multi": float(np.mean([b["mi_kl"] for b in multi if b["n_distinct_actions"] == 1])) if any(
                b["n_distinct_actions"] == 1 for b in multi) else 0.0,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }

    per_env = out_dir / "per_env"
    per_env.mkdir(parents=True, exist_ok=True)
    state_fields = ["front_index", "timestep", "greedy_label", "command_R_t",
                    "I_kl", "I_tv", "I_js", "I_flip",
                    "gt_flip_fraction", "gt_flip_exists", "gt_nearest_flip_scaled_l2", "gt_gt_method"]
    with (per_env / f"{name}_states.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["env"] + state_fields)
        w.writeheader()
        for r in all_rows:
            row = {"env": name}
            for k in state_fields:
                v = r.get(k)
                row[k] = json.dumps(v) if isinstance(v, list) else v
            w.writerow(row)
    with (per_env / f"{name}_state_basis.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["env"] + list(basis_rows[0].keys()))
        w.writeheader()
        for b in basis_rows:
            w.writerow({"env": name, **b})
    with (per_env / f"{name}_summary.json").open("w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Cross-env evaluation of the goal-influence method.")
    ap.add_argument("--envs", nargs="+", required=True, choices=sorted(ENV_SETTINGS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-label", default="comprehensive")
    args = ap.parse_args()

    out_dir = goal_output_artifacts_dir() / args.out_label
    (out_dir / "per_env").mkdir(parents=True, exist_ok=True)
    for name in args.envs:
        print(f"=== {name} ===", flush=True)
        try:
            s = evaluate_env(name, ENV_SETTINGS[name], args.device, out_dir)
            f = s["faithfulness"]
            def fmt(x):
                return f"{x:.3f}" if isinstance(x, float) and np.isfinite(x) else "n/a"
            print(f"  {s['n_objectives']}-obj  {s['n_actions']} actions | states={s['n_states']} "
                  f"(unique {s['n_unique_states']}, repeat {s['obs_repeat_ratio']}) | "
                  f"goal-active {s['frac_goal_active']} | gt={s['gt_method']} | {s['elapsed_s']}s")
            for sc in SCORES:
                print(f"    {sc.upper():<5} sp(flip%)={fmt(f[sc]['spearman_flip_fraction'])} "
                      f"AUROC={fmt(f[sc]['auroc_detect_goal_active'])} "
                      f"sp(near)={fmt(f[sc]['spearman_nearest_flip'])}")
            sb = s["state_basis"]
            print(f"    basis: cells={sb['n_cells']} multi={sb['n_multi_visit']} forks={sb['n_forks']} "
                  f"MI(forks)={sb['mean_mi_forks']:.3f} MI(non-fork)={sb['mean_mi_nonfork_multi']:.3f}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            with (out_dir / "per_env" / f"{name}_summary.json").open("w", encoding="utf-8") as h:
                json.dump({"env": name, "failed": f"{type(e).__name__}: {e}"}, h, indent=2)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
