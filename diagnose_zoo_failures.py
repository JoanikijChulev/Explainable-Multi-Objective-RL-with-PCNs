"""Diagnose ZOO counterfactual failures with a comprehensive command-grid sweep.

For each FAILED case in a `evaluate_pcn_zoo_counterfactuals.py` run, decide whether
it failed because:

  * INFEASIBLE foil  -- NO command R anywhere in the feasible box makes the model
                        pick the foil action (the policy never learned to produce
                        it in that state), so there is no counterfactual to find; or
  * ZOO FAILURE      -- a command that flips the action *does* exist, but the
                        black-box ZOO search did not find it.

Method: brute force. Given the case's fixed state (obs), sweep a dense grid over
the ENTIRE command/reward box -- every reward dimension, every grid value -- and
evaluate the true margin m(R) = logp(foil) - max_other_logp at every node. If the
max over the whole grid reaches m >= margin, a counterfactual exists (ZOO missed
it); otherwise the foil is infeasible. This checks every command input given the
state, up to the grid resolution.

three_tree is excluded by default (it had no failures anyway).

Usage:
    python diagnose_zoo_failures.py [--eval-dir DIR] [--runs bp,dst]
                                    [--grid 25] [--max-cases N]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_cw_vs_zoo_paper_experiments as paper
import interactive_pcn_zoo_cf as cf
from device_utils import preferred_device


def latest_eval_dir():
    root = Path("output_artifacts") / "pcn_zoo_cf_evaluations"
    dirs = sorted(p for p in root.glob("*") if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"no eval runs under {root}")
    return dirs[-1]


def read_failures(eval_dir, exclude, runs_filter):
    rows = []
    with (eval_dir / "results_cases.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("final_success") == "true" or r["run"] in exclude:
                continue
            if runs_filter and r["run"] not in runs_filter:
                continue
            rows.append(r)
    return rows


def grid_max_margin(model, obs, bounds, reward_dim, grid_per_dim, max_points,
                    action_mask, foil, device, batch):
    """Sweep the dense g-per-dim lattice over the whole command box and return the
    largest margin found (and the command that achieves it).

    The lattice (g**reward_dim nodes) is generated and evaluated in streaming
    chunks of `batch` rows, so peak memory is O(batch) regardless of g -- nothing
    near the full g**reward_dim array is ever materialised. g is auto-coarsened
    only if g**reward_dim would exceed max_points.
    """
    low, high = np.asarray(bounds[0], np.float32), np.asarray(bounds[1], np.float32)
    g = int(grid_per_dim)
    while g > 5 and g ** reward_dim > max_points:
        g -= 1
    axes = [np.linspace(low[d], high[d], g, dtype=np.float32) for d in range(reward_dim)]
    total = g ** reward_dim
    divs = [g ** (reward_dim - 1 - d) for d in range(reward_dim)]  # row-major unravel

    foil = int(foil)
    valid = np.flatnonzero(np.asarray(action_mask, bool))
    others = valid[valid != foil]

    obs_arr = np.asarray(obs)
    b0 = int(min(batch, total))
    obs_dev = torch.as_tensor(
        np.ascontiguousarray(np.broadcast_to(obs_arr, (b0,) + obs_arr.shape))
    ).to(device)

    best_m, best_cmd = -np.inf, None
    for s in range(0, total, batch):
        n = min(batch, total - s)
        idx = np.arange(s, s + n, dtype=np.int64)
        chunk = np.empty((n, reward_dim), dtype=np.float32)
        for d in range(reward_dim):
            chunk[:, d] = axes[d][(idx // divs[d]) % g]
        with torch.no_grad():
            lp = model(
                obs_dev[:n],
                torch.as_tensor(chunk, dtype=torch.float32).to(device),
            ).detach().cpu().numpy()
        m = lp[:, foil] - lp[:, others].max(axis=1)
        k = int(np.argmax(m))
        if m[k] > best_m:
            best_m, best_cmd = float(m[k]), chunk[k].copy()
    return best_m, best_cmd, g, total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default=None, help="eval run dir (default: most recent)")
    ap.add_argument("--runs", default=None, help="comma-separated run filter")
    ap.add_argument("--exclude", default="three_tree", help="comma-separated runs to skip")
    ap.add_argument("--margin", type=float, default=None, help="success margin (default: from config.json)")
    ap.add_argument("--grid", type=int, default=25, help="grid points per reward dimension")
    ap.add_argument("--max-points", type=int, default=300000, help="cap on grid nodes per case (auto-coarsens)")
    ap.add_argument("--batch", type=int, default=65536, help="forward-pass / streaming chunk size")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--out", default=None, help="output CSV (default: <eval-dir>/failure_diagnosis.csv)")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir) if args.eval_dir else latest_eval_dir()
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    runs_filter = {s.strip() for s in args.runs.split(",")} if args.runs else None
    margin = args.margin
    if margin is None:
        margin = float(json.loads((eval_dir / "config.json").read_text())["settings"]["margin"])

    failures = read_failures(eval_dir, exclude, runs_filter)
    if args.max_cases:
        failures = failures[: args.max_cases]
    print(f"Eval dir: {eval_dir}")
    print(f"Failures to diagnose: {len(failures)}  (margin={margin}, grid={args.grid}/dim)")

    device = preferred_device()
    out_rows, by_run = [], {}
    rc, cur_run = None, None

    for i, row in enumerate(failures, 1):
        if row["run"] != cur_run:
            if rc is not None:
                paper.close_run(rc)
            rc = paper.load_run_by_name(row["run"], None, device)
            cur_run = row["run"]
        try:
            case = paper.rollout_single_case(
                rc, int(row["front_index"]), int(row["timestep"]), int(row["foil_action"])
            )
            bounds = cf.command_bounds(rc["setup"])
            base = np.asarray(case["remaining_command"], np.float32)
            # In stochastic envs (e.g. resource-gathering) the greedy rollout is not
            # reproducible, so the rebuilt state can differ from the one ZOO faced.
            # Only trust the verdict when the rebuilt command matches the logged one.
            logged = np.asarray(json.loads(row["remaining_command"]), np.float32)
            if base.shape != logged.shape or not np.allclose(base, logged, atol=1e-3):
                rec = {"oracle_max_margin": "", "oracle_command": "", "oracle_scaled_l2": "",
                       "grid_per_dim": "", "grid_points": "", "verdict": "state_mismatch"}
            else:
                best_m, best_cmd, g, npts = grid_max_margin(
                    rc["model"], case["obs"], bounds, len(base), args.grid,
                    args.max_points, case["action_mask"], case["foil_action"],
                    device, args.batch,
                )
                feasible = best_m >= margin
                verdict = "zoo_failure" if feasible else "infeasible_foil"
                scaled = (
                    round(float(cf.scaled_command_distance(best_cmd, base, cf.command_scale(bounds))), 4)
                    if feasible else ""
                )
                rec = {
                    "oracle_max_margin": round(best_m, 4),
                    "oracle_command": json.dumps([round(float(x), 4) for x in best_cmd]) if feasible else "",
                    "oracle_scaled_l2": scaled, "grid_per_dim": g, "grid_points": npts,
                    "verdict": verdict,
                }
        except Exception as exc:
            rec = {"oracle_max_margin": "", "oracle_command": "", "oracle_scaled_l2": "",
                   "grid_per_dim": "", "grid_points": "", "verdict": f"error:{type(exc).__name__}"}

        out_rows.append({
            "run": row["run"], "env": row["env"], "front_index": row["front_index"],
            "timestep": row["timestep"], "foil_action": row["foil_action"], "foil_label": row["foil_label"],
            "original_action": row["original_action"], "original_label": row["original_label"],
            "reward_dim": row["reward_dim"], "zoo_final_margin": row.get("final_target_margin", ""),
            "zoo_failure_reason": row.get("failure_reason", ""), **rec,
        })
        by_run.setdefault(row["run"], []).append(rec["verdict"])
        if i % 25 == 0:
            print(f"  {i}/{len(failures)} done", flush=True)
    if rc is not None:
        paper.close_run(rc)

    out_path = Path(args.out) if args.out else eval_dir / "failure_diagnosis.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print("\n=== Verdict summary (failure attributions) ===")
    n_inf = sum(1 for r in out_rows if r["verdict"] == "infeasible_foil")
    n_zoo = sum(1 for r in out_rows if r["verdict"] == "zoo_failure")
    for run in sorted(by_run):
        v = by_run[run]
        print(f"  {run:12s} failures={len(v):4d}  infeasible_foil={v.count('infeasible_foil'):4d}"
              f"  zoo_failure={v.count('zoo_failure'):4d}  other={len(v)-v.count('infeasible_foil')-v.count('zoo_failure')}")
    print(f"  {'TOTAL':12s} failures={len(out_rows):4d}  infeasible_foil={n_inf:4d}"
          f"  zoo_failure={n_zoo:4d}  other={len(out_rows)-n_inf-n_zoo}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
