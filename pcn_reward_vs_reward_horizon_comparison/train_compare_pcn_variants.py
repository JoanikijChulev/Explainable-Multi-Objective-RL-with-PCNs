import argparse
import json
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from artifacts import (
    checkpoints_dir,
    create_run_dir,
    load_run_metadata,
    output_artifacts_dir,
    resolve_run_dir,
    validate_run_name,
    write_run_metadata,
)
from device_utils import device_description, preferred_device
from eval_pcn import (
    is_horizon_conditioned,
    load_logged_commands,
    resolve_checkpoint_path,
    run_episode as greedy_run_episode,
)
from morl_metrics import expected_utility, hypervolume_score
from pcn.env_setup import build_experiment_setup as build_reward_setup
from pcn.env_setup import canonical_env_name
from pcn.env_setup_horizon import build_experiment_setup as build_horizon_setup
from pcn.horizon_pcn import train as train_horizon
from pcn.pcn import train as train_reward
from plot_artifacts import save_training_plots


VARIANTS = ('reward', 'horizon')


def set_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_points(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points[None, :]
    if len(points) == 0:
        return points
    return points[~np.isnan(points).any(axis=1)]


def non_dominated(points):
    points = clean_points(points)
    if len(points) == 0:
        return points
    keep = np.ones(points.shape[0], dtype=bool)
    for i, candidate in enumerate(points):
        if keep[i]:
            keep[keep] = np.any(points[keep] > candidate, axis=1)
            keep[i] = True
    return points[keep]


def latest_eval_points_from_log(run_dir):
    log_path = Path(run_dir) / 'log.h5'
    if not log_path.is_file():
        return np.empty((0, 0), dtype=np.float32)
    with h5py.File(log_path, 'r') as log:
        if 'eval/return/value/ndarray' in log:
            return clean_points(log['eval/return/value/ndarray'][-1])
        if 'train/leaves/r/ndarray' in log:
            return clean_points(log['train/leaves/r/ndarray'][-1])
    return np.empty((0, 0), dtype=np.float32)


def move_model_scaling_to_device(model, device):
    if hasattr(model, 'scaling_factor') and torch.is_tensor(model.scaling_factor):
        model.scaling_factor = model.scaling_factor.to(device)


def checkpoint_count(run_dir):
    return len(list(checkpoints_dir(run_dir).glob('model_*.pt')))


def resolve_shared_training_budget(args, device):
    reference_setup = build_reward_setup(
        args.env,
        device=device,
        fruit_tree_depth=args.fruit_tree_depth,
        include_model=False,
    )
    try:
        return {
            'total_steps': int(args.total_steps if args.total_steps is not None else reference_setup.total_steps),
            'batch_size': int(args.batch_size if args.batch_size is not None else reference_setup.batch_size),
            'n_model_updates': int(args.n_model_updates if args.n_model_updates is not None else reference_setup.n_model_updates),
            'n_step_episodes': int(args.n_step_episodes),
            'n_er_episodes': int(args.n_er_episodes if args.n_er_episodes is not None else reference_setup.n_er_episodes),
            'gamma': float(args.gamma),
            'max_size': int(args.max_size if args.max_size is not None else reference_setup.max_size),
        }
    finally:
        reference_setup.env.close()
        if reference_setup.er_env is not None:
            reference_setup.er_env.close()


def build_metadata(setup, run_name, variant, budget, args, device):
    metadata = {
        'run_name': run_name,
        'env': setup.env_name,
        'pcn_variant': variant,
        'horizon_conditioned': variant == 'horizon',
        'model_source': None,
        'device': device,
        'learning_rate': setup.learning_rate,
        'min_learning_rate': setup.min_learning_rate if variant == 'reward' else None,
        'batch_size': int(budget['batch_size']),
        'total_steps': int(budget['total_steps']),
        'n_model_updates': int(budget['n_model_updates']),
        'n_step_episodes': int(budget['n_step_episodes']),
        'n_er_episodes': int(budget['n_er_episodes']),
        'gamma': float(budget['gamma']),
        'max_size': int(budget['max_size']),
        'max_return': setup.max_return.astype('float32').tolist(),
        'ref_point': setup.ref_point.astype('float32').tolist(),
        'comparison_group': args.run_prefix,
        'seed': args.seed,
    }
    if variant == 'reward':
        metadata.update(
            {
                'reward_neutral_suffix_trimming': True,
                'reward_time_tiebreak_commands': True,
                'command_inputs': ['desired_return'],
            }
        )
    else:
        metadata.update(
            {
                'optimizer': 'Adam',
                'command_inputs': ['desired_return', 'desired_horizon'],
            }
        )
    metadata.update(setup.metadata)
    return metadata


def training_args_from_setup(setup, budget):
    return {
        'learning_rate': setup.learning_rate,
        'batch_size': int(budget['batch_size']),
        'total_steps': int(budget['total_steps']),
        'n_model_updates': int(budget['n_model_updates']),
        'n_step_episodes': int(budget['n_step_episodes']),
        'n_er_episodes': int(budget['n_er_episodes']),
        'gamma': float(budget['gamma']),
        'max_return': setup.max_return,
        'max_size': int(budget['max_size']),
        'ref_point': setup.ref_point,
    }


def train_variant(variant, run_name, args, device, budget):
    builder = build_reward_setup if variant == 'reward' else build_horizon_setup
    trainer = train_reward if variant == 'reward' else train_horizon
    setup = builder(
        args.env,
        device=device,
        fruit_tree_depth=args.fruit_tree_depth,
    )
    metadata = build_metadata(setup, run_name, variant, budget, args, device)
    logdir = create_run_dir(run_name)
    write_run_metadata(logdir, metadata)

    kwargs = training_args_from_setup(setup, budget)
    print(f'training {variant} variant in {logdir}')
    print(
        f'{variant}: total_steps={kwargs["total_steps"]}, '
        f'n_er_episodes={kwargs["n_er_episodes"]}, '
        f'batch_size={kwargs["batch_size"]}, '
        f'n_model_updates={kwargs["n_model_updates"]}, '
        f'n_step_episodes={kwargs["n_step_episodes"]}, '
        f'max_size={kwargs["max_size"]}'
    )

    try:
        if variant == 'reward':
            trainer(
                setup.env,
                setup.model,
                min_learning_rate=setup.min_learning_rate,
                time_tiebreak_commands=True,
                logdir=logdir,
                er_env=setup.er_env,
                er_policy=setup.er_policy,
                **kwargs,
            )
        else:
            trainer(
                setup.env,
                setup.model,
                logdir=logdir,
                er_env=setup.er_env,
                er_policy=setup.er_policy,
                **kwargs,
            )
    finally:
        setup.env.close()
        if setup.er_env is not None:
            setup.er_env.close()

    save_training_plots(logdir)
    return logdir


def evaluate_variant(variant, run_dir, eval_n, device):
    metadata = load_run_metadata(run_dir)
    env_name = metadata['env']
    fruit_tree_depth = int(metadata.get('fruit_tree_depth', 6))
    horizon_conditioned = is_horizon_conditioned(metadata)
    builder = build_reward_setup if variant == 'reward' else build_horizon_setup
    setup = builder(
        env_name,
        device=device,
        fruit_tree_depth=fruit_tree_depth,
        include_model=False,
    )
    checkpoint_path, _ = resolve_checkpoint_path(run_dir, None)
    model = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device)
    move_model_scaling_to_device(model, device)
    gamma = float(metadata.get('gamma', 1.0))
    # Greedy (deterministic) rollouts over the full logged Pareto-front commands,
    # matching eval_pcn and the paper's deterministic execution policy (replaces the
    # previous stochastic, 50-sample replay eval). eval_n is no longer used.
    commands, horizons = load_logged_commands(
        run_dir / 'log.h5',
        horizon_conditioned=horizon_conditioned,
    )
    achieved_returns = []
    desired_returns = []
    try:
        for index in range(len(commands)):
            desired_horizon = None if horizons is None else float(horizons[index])
            transitions = greedy_run_episode(
                setup.env,
                model,
                commands[index],
                setup.max_return,
                desired_horizon=desired_horizon,
            )
            for timestep in reversed(range(len(transitions) - 1)):
                transitions[timestep].reward += gamma * transitions[timestep + 1].reward
            achieved_returns.append(np.asarray(transitions[0].reward, dtype=np.float32).flatten())
            desired_returns.append(np.asarray(commands[index], dtype=np.float32).flatten())
    finally:
        setup.env.close()
        if setup.er_env is not None:
            setup.er_env.close()
    achieved = np.asarray(achieved_returns, dtype=np.float32)
    desired = np.asarray(desired_returns, dtype=np.float32)
    distances = np.linalg.norm(desired - achieved, axis=0)
    return {
        'checkpoint': checkpoint_path,
        'achieved': clean_points(achieved),
        'desired': clean_points(desired),
        'distances': np.asarray(distances, dtype=np.float32),
        'ref_point': setup.ref_point.astype(np.float32),
    }


def save_front_plot(results, output_path):
    fronts = {
        variant: non_dominated(result['achieved'])
        for variant, result in results.items()
        if len(result['achieved']) > 0
    }
    if not fronts:
        return
    first_front = next(iter(fronts.values()))
    n_objectives = int(first_front.shape[-1])

    if n_objectives == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        for variant, front in fronts.items():
            front = front[np.argsort(front[:, 0])]
            ax.plot(front[:, 0], front[:, 1], marker='o', label=variant)
        ax.set_xlabel('Objective 0 Return')
        ax.set_ylabel('Objective 1 Return')
    elif n_objectives == 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        for variant, front in fronts.items():
            ax.scatter(front[:, 0], front[:, 1], front[:, 2], s=40, label=variant)
        ax.set_xlabel('Objective 0 Return')
        ax.set_ylabel('Objective 1 Return')
        ax.set_zlabel('Objective 2 Return')
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        objective_indexes = np.arange(n_objectives)
        for variant, front in fronts.items():
            if len(front) > 8:
                front = front[np.linspace(0, len(front) - 1, 8, dtype=np.int32)]
            for point in front:
                ax.plot(objective_indexes, point, alpha=0.45)
            ax.plot([], [], label=variant)
        ax.set_xticks(objective_indexes)
        ax.set_xlabel('Objective Index')
        ax.set_ylabel('Return')

    ax.set_title('PCN Variant Evaluation Fronts')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def to_serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value


def compare_runs(run_dirs, comparison_dir, eval_n, device):
    comparison_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    summary = {}
    for variant, run_dir in run_dirs.items():
        result = evaluate_variant(variant, run_dir, eval_n, device)
        results[variant] = result
        np.save(comparison_dir / f'{variant}_achieved_returns.npy', result['achieved'])
        np.save(comparison_dir / f'{variant}_desired_returns.npy', result['desired'])
        front = non_dominated(result['achieved'])
        hv_score, hv_excluded = hypervolume_score(
            result['achieved'],
            result['ref_point'],
            return_excluded=True,
        )
        summary[variant] = {
            'run_dir': str(run_dir),
            'checkpoint': str(result['checkpoint']),
            'checkpoint_count': checkpoint_count(run_dir),
            'evaluated_points': int(len(result['achieved'])),
            'non_dominated_points': int(len(front)),
            'expected_utility': expected_utility(result['achieved']),
            'hypervolume': hv_score,
            'hypervolume_reference_point': result['ref_point'].tolist(),
            'hypervolume_excluded_non_dominated_points': hv_excluded,
            'mean_command_distance_by_objective': result['distances'].tolist(),
            'last_logged_eval_points': int(len(latest_eval_points_from_log(run_dir))),
        }

    save_front_plot(results, comparison_dir / 'evaluation_fronts.png')
    with (comparison_dir / 'comparison_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(to_serializable(summary), handle, indent=2, sort_keys=True)
        handle.write('\n')
    return summary


def requested_variants(variant_arg):
    if variant_arg == 'both':
        return VARIANTS
    return (variant_arg,)


def default_run_prefix(env_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_env = canonical_env_name(env_name).replace('-', '_')
    return f'pcn_compare_{safe_env}_{timestamp}'


def check_run_names_available(names):
    root = output_artifacts_dir()
    for name in names:
        validate_run_name(name)
        candidate = root / name
        if candidate.exists():
            raise FileExistsError(f'run directory already exists: {candidate}')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Train reward-only and/or horizon-conditioned PCN variants with the repo defaults, '
            'then run a fresh replay-frontier evaluation and write comparison artifacts.'
        )
    )
    parser.add_argument(
        '--env',
        required=True,
        type=str,
        help='dst, collect_two/c2, branch-path/bp, reward-line/rl, three-tree/tt, minecart/mc, mo-mountaincar-timespeed-v0/momc, mo-lunar-lander-v3/moll (mo-lunar-lander-v2 alias), mo-reacher-v5/mor, walkroom2...walkroom9, fruit-tree-v0/ft, four-room-v0/4room, resource-gathering-v0/rsg, breakable-bottles-v0/bb',
    )
    parser.add_argument(
        '--variant',
        default='both',
        choices=('both', 'reward', 'horizon'),
        help='which variant(s) to train',
    )
    parser.add_argument(
        '--run-prefix',
        default=None,
        type=str,
        help='prefix for comparison_output_artifacts subfolders; defaults to timestamped pcn_compare_<env>_<time>',
    )
    parser.add_argument('--total-steps', default=None, type=int, help='override shared total steps for every trained variant')
    parser.add_argument('--batch-size', default=None, type=int, help='override batch size for both variants')
    parser.add_argument('--n-model-updates', default=None, type=int, help='override model updates per training cycle')
    parser.add_argument('--n-er-episodes', default=None, type=int, help='override random replay initialization episodes')
    parser.add_argument('--n-step-episodes', default=10, type=int, help='new policy episodes per training cycle')
    parser.add_argument('--max-size', default=None, type=int, help='override replay max size')
    parser.add_argument('--gamma', default=1.0, type=float, help='discount used for return relabeling')
    parser.add_argument('--eval-n', default=50, type=int, help='(deprecated/ignored) the final comparison now greedily rolls out the full logged Pareto front, like eval_pcn')
    parser.add_argument('--fruit-tree-depth', default=6, type=int, help='depth for fruit-tree-v0 (5, 6 or 7)')
    parser.add_argument('--seed', default=None, type=int, help='optional random seed reused before each variant training')
    parser.add_argument('--no-compare', action='store_true', help='train only; skip post-training comparison evaluation')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.run_prefix is None:
        args.run_prefix = default_run_prefix(args.env)
    validate_run_name(args.run_prefix)

    variants = requested_variants(args.variant)
    run_names = {variant: f'{args.run_prefix}_{variant}' for variant in variants}
    names_to_check = list(run_names.values())
    if not args.no_compare:
        names_to_check.append(f'{args.run_prefix}_comparison')
    try:
        check_run_names_available(names_to_check)
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc))

    device = preferred_device()
    print(f'using device: {device_description(device)}')
    budget = resolve_shared_training_budget(args, device)
    print(
        'shared budget: '
        f'total_steps={budget["total_steps"]}, '
        f'n_er_episodes={budget["n_er_episodes"]}, '
        f'batch_size={budget["batch_size"]}, '
        f'n_model_updates={budget["n_model_updates"]}, '
        f'n_step_episodes={budget["n_step_episodes"]}, '
        f'max_size={budget["max_size"]}'
    )
    run_dirs = {}
    for variant, run_name in run_names.items():
        set_seed(args.seed)
        run_dirs[variant] = train_variant(variant, run_name, args, device, budget)

    if not args.no_compare:
        comparison_dir = output_artifacts_dir() / f'{args.run_prefix}_comparison'
        summary = compare_runs(run_dirs, comparison_dir, args.eval_n, device)
        print(f'wrote comparison artifacts to {comparison_dir}')
        for variant, metrics in summary.items():
            print(
                f'{variant}: expected_utility={metrics["expected_utility"]:.6f}, '
                f'hypervolume={metrics["hypervolume"]:.6f}, '
                f'non_dominated_points={metrics["non_dominated_points"]}'
            )
    else:
        print('comparison skipped')
