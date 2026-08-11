import re

import numpy as np
import torch

from artifacts import checkpoints_dir, create_evaluation_dir, load_run_metadata, resolve_run_dir
from device_utils import device_description, preferred_device
from pcn.env_setup import build_experiment_setup, canonical_env_name
from pcn.pcn import Transition, action_mask_for_env, apply_action_mask
from pcn.replay_artifacts import FINAL_HORIZONS_DATASET, FINAL_RETURNS_DATASET
from plot_artifacts import save_eval_pareto_front_comparison


device = preferred_device()


def non_dominated(solutions, return_indexes=False):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            is_efficient[i] = 1
    if return_indexes:
        return solutions[is_efficient], is_efficient
    return solutions[is_efficient]


def format_return(vector):
    return np.array2string(np.asarray(vector), precision=3, floatmode='fixed')


def is_horizon_conditioned(metadata):
    return bool(metadata.get('horizon_conditioned')) or metadata.get('pcn_variant') == 'horizon'


def load_logged_commands(log_path, horizon_conditioned=False):
    import h5py

    horizons = None
    with h5py.File(log_path, 'r') as log:
        if FINAL_RETURNS_DATASET in log:
            pareto_front = np.asarray(log[FINAL_RETURNS_DATASET], dtype=np.float32)
            if horizon_conditioned:
                if FINAL_HORIZONS_DATASET not in log:
                    raise KeyError(
                        f'horizon-conditioned commands require {FINAL_HORIZONS_DATASET} in the log'
                    )
                horizons = np.asarray(log[FINAL_HORIZONS_DATASET], dtype=np.float32).reshape(-1)
        elif 'train/leaves/r/ndarray' in log:
            pareto_front = np.asarray(log['train/leaves/r/ndarray'][-1], dtype=np.float32)
            if horizon_conditioned:
                if 'train/leaves/h/ndarray' not in log:
                    raise KeyError(
                        'horizon-conditioned commands require train/leaves/h/ndarray in the log'
                    )
                horizons = np.asarray(log['train/leaves/h/ndarray'][-1], dtype=np.float32).reshape(-1)
        elif 'eval/return/desired/ndarray' in log:
            pareto_front = np.asarray(log['eval/return/desired/ndarray'][-1], dtype=np.float32)
            print('warning: train/leaves data is missing; falling back to the last logged eval desired returns')
            if horizon_conditioned:
                raise KeyError(
                    'horizon-conditioned commands need logged train/leaves/h horizons; '
                    'eval desired returns alone are not enough'
                )
        else:
            raise KeyError(f'run log does not contain train/leaves or eval/return/desired data: {log_path}')

    if pareto_front.ndim == 1:
        pareto_front = pareto_front[None, :]
    if pareto_front.ndim != 2:
        raise ValueError(f'logged commands must be a 2D array, got shape {pareto_front.shape}')
    if horizons is not None and len(horizons) != len(pareto_front):
        raise ValueError(
            'logged return and horizon command counts do not match: '
            f'{len(pareto_front)} returns versus {len(horizons)} horizons'
        )

    valid = np.isfinite(pareto_front).all(axis=1)
    if horizons is not None:
        valid = valid & np.isfinite(horizons)
        horizons = horizons[valid]
    pareto_front = pareto_front[valid]
    if len(pareto_front) == 0:
        raise ValueError(f'run log contains no finite commands: {log_path}')
    _, pareto_front_mask = non_dominated(pareto_front, return_indexes=True)
    pareto_front = pareto_front[pareto_front_mask]
    if horizons is not None:
        horizons = horizons[pareto_front_mask]
        # Horizon PCN commands use the episode length minus 2, matching train/eval.
        horizons = np.maximum(horizons - 2.0, 1.0).astype(np.float32)
    order = np.argsort(pareto_front[:, 0])
    pareto_front = pareto_front[order]
    if horizons is not None:
        horizons = horizons[order]
    return pareto_front, horizons


def resolve_checkpoint_path(model_dir, checkpoint_arg=None):
    checkpoint_root = checkpoints_dir(model_dir)
    checkpoints = list(checkpoint_root.glob('model_*.pt'))
    if not checkpoints:
        checkpoints = list(model_dir.glob('model_*.pt'))
    assert checkpoints, f'no model_*.pt checkpoints found in {checkpoint_root} or {model_dir}'
    checkpoints = sorted(
        checkpoints,
        key=lambda p: int(re.search(r'model_(\d+)\.pt$', p.name).group(1)),
    )

    if checkpoint_arg is None:
        return checkpoints[-1], checkpoints

    checkpoint_arg = str(checkpoint_arg).strip()
    candidate = model_dir / checkpoint_arg
    if candidate.is_file():
        return candidate.resolve(), checkpoints

    from pathlib import Path

    candidate = Path(checkpoint_arg)
    if candidate.is_file():
        return candidate.resolve(), checkpoints

    if checkpoint_arg.isdigit():
        checkpoint_name = f'model_{int(checkpoint_arg)}.pt'
    else:
        checkpoint_name = checkpoint_arg
        if not checkpoint_name.endswith('.pt'):
            checkpoint_name = f'{checkpoint_name}.pt'

    matching = [checkpoint for checkpoint in checkpoints if checkpoint.name == checkpoint_name]
    if matching:
        return matching[0], checkpoints

    available = ', '.join(checkpoint.name for checkpoint in checkpoints[-10:])
    raise FileNotFoundError(
        f'could not find checkpoint "{checkpoint_arg}" in {model_dir}. '
        f'Latest available examples: {available}'
    )


def print_interactive_solutions(pareto_front):
    print('available solutions:')
    for index, point in enumerate(pareto_front):
        print(f'[{index:>3}] desired={format_return(point)}')
    print('enter an index to evaluate, or Q to quit')


def choose_interactive_policy(pareto_front):
    print_interactive_solutions(pareto_front)
    while True:
        raw = input('select policy -> ').strip()
        if raw.lower() in {'q', 'quit', 'exit'}:
            return None
        if raw.lower() in {'l', 'list'}:
            print_interactive_solutions(pareto_front)
            continue
        try:
            selected = int(raw)
        except ValueError:
            print('invalid input. enter a policy index, L to list again, or Q to quit.')
            continue
        if not (0 <= selected < len(pareto_front)):
            print(f'index out of range. choose 0 to {len(pareto_front)-1}, or Q to quit.')
            continue
        return selected


def greedy_action(env, model, obs, desired_return, desired_horizon=None):
    obs = np.asarray([obs])
    desired_return = np.asarray([desired_return], dtype=np.float32)
    obs_tensor = torch.as_tensor(obs).to(device)
    desired_return_tensor = torch.as_tensor(desired_return).to(device)
    if desired_horizon is None:
        log_probs = model(obs_tensor, desired_return_tensor)
    else:
        desired_horizon_tensor = torch.as_tensor(
            np.asarray([desired_horizon], dtype=np.float32),
            device=device,
        ).unsqueeze(1)
        log_probs = model(obs_tensor, desired_return_tensor, desired_horizon_tensor)
    log_probs = log_probs.detach().cpu().numpy()[0]
    action_mask = action_mask_for_env(env)
    masked_log_probs = apply_action_mask(log_probs, action_mask)
    return int(np.argmax(masked_log_probs))


def run_episode(env, model, desired_return, max_return, desired_horizon=None, seed=None):
    transitions = []
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    remaining_return = np.asarray(desired_return, dtype=np.float32).copy()
    remaining_horizon = None if desired_horizon is None else np.float32(desired_horizon)
    while not (terminated or truncated):
        action = greedy_action(env, model, obs, remaining_return, desired_horizon=remaining_horizon)
        n_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        transitions.append(Transition(
            observation=obs,
            action=action,
            reward=np.float32(reward).copy(),
            next_observation=n_obs,
            terminal=done
        ))

        obs = n_obs
        remaining_return = np.clip(remaining_return - reward, None, max_return, dtype=np.float32)
        if remaining_horizon is not None:
            remaining_horizon = np.float32(max(float(remaining_horizon) - 1.0, 1.0))
    return transitions


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PCN')
    parser.add_argument('model', type=str, help='run directory path or comparison_output_artifacts run name')
    parser.add_argument('--interactive', action='store_true', help='interactive policy selection')
    parser.add_argument(
        '--checkpoint',
        default=None,
        type=str,
        help='checkpoint to evaluate: default is latest, or pass a checkpoint number, filename, or full path',
    )
    parser.add_argument(
        '--seed',
        default=None,
        type=int,
        help='optional base seed for reproducible rollouts; default is None (fresh random starts each run). '
             'When set, each command resets the env with seed+command_index.',
    )
    parser.set_defaults(interactive=False)
    args = parser.parse_args()

    if args.seed is not None and args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    try:
        model_dir = resolve_run_dir(args.model)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    metadata = load_run_metadata(model_dir)

    env_name = metadata.get('env')
    if env_name is None:
        for candidate in ('dst', 'collect_two', 'branch-path', 'reward-line', 'three-tree', 'minecart', 'mo-mountaincar-timespeed-v0', 'mo-lunar-lander-v3', 'mo-reacher-v5', 'fruit-tree-v0', 'four-room-v0', 'resource-gathering-v0', 'breakable-bottles-v0'):
            if candidate in str(model_dir):
                env_name = candidate
                break
    if env_name is None:
        match = re.search(r'walkroom(\d+)', str(model_dir))
        if match is not None:
            env_name = f'walkroom{match.group(1)}'
    if env_name is None and re.search(r'collect[-_]?two', str(model_dir), flags=re.IGNORECASE):
        env_name = 'collect_two'
    if env_name is None and re.search(r'branch[-_]?path', str(model_dir), flags=re.IGNORECASE):
        env_name = 'branch-path'
    if env_name is None and re.search(r'reward[-_]?line', str(model_dir), flags=re.IGNORECASE):
        env_name = 'reward-line'
    if env_name is None and re.search(r'three[-_]?tree', str(model_dir), flags=re.IGNORECASE):
        env_name = 'three-tree'
    if env_name is None and re.search(r'ordered[-_]?pair', str(model_dir), flags=re.IGNORECASE):
        env_name = 'collect_two'
    if env_name is None and re.search(r'fruit[-_]?tree', str(model_dir), flags=re.IGNORECASE):
        env_name = 'fruit-tree-v0'
    if env_name is None and re.search(r'(mo[-_])?mountain[-_]?car|timespeed|momc', str(model_dir), flags=re.IGNORECASE):
        env_name = 'mo-mountaincar-timespeed-v0'
    if env_name is None and re.search(r'(mo[-_])?lunar[-_]?lander|moll', str(model_dir), flags=re.IGNORECASE):
        env_name = 'mo-lunar-lander-v3'
    if env_name is None and re.search(r'(mo[-_])?reacher|mor', str(model_dir), flags=re.IGNORECASE):
        env_name = 'mo-reacher-v5'
    if env_name is None and re.search(r'four[-_]?room', str(model_dir), flags=re.IGNORECASE):
        env_name = 'four-room-v0'
    if env_name is None and re.search(r'resource[-_]?gathering', str(model_dir), flags=re.IGNORECASE):
        env_name = 'resource-gathering-v0'
    if env_name is None and re.search(r'breakable[-_]?bottles', str(model_dir), flags=re.IGNORECASE):
        env_name = 'breakable-bottles-v0'

    assert env_name is not None, 'log of unknown env'
    fruit_tree_depth = metadata.get('fruit_tree_depth', 6)
    if canonical_env_name(env_name) == 'fruit-tree-v0' and 'fruit_tree_depth' not in metadata:
        print('warning: fruit-tree run metadata is missing fruit_tree_depth; defaulting to depth 6')
    setup = build_experiment_setup(
        canonical_env_name(env_name),
        device=device,
        fruit_tree_depth=fruit_tree_depth,
        include_model=False,
    )
    env = setup.env
    max_return = setup.max_return
    horizon_conditioned = is_horizon_conditioned(metadata)

    try:
        checkpoint_path, checkpoints = resolve_checkpoint_path(model_dir, args.checkpoint)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    model = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device)
    model.scaling_factor = model.scaling_factor.to(device)
    print(f'using device: {device_description(device)}')
    print(f'using checkpoint: {checkpoint_path.name}')
    if horizon_conditioned:
        print('detected horizon-conditioned PCN run')

    pareto_front, pareto_horizons = load_logged_commands(
        model_dir / 'log.h5',
        horizon_conditioned=horizon_conditioned,
    )

    inp = -1
    evaluation_dir = None
    if not args.interactive:
        evaluation_dir = create_evaluation_dir(model_dir)
        print('=' * 38)
        print('not interactive, this may take a while')
        print('=' * 38)
        print(f'saving evaluation artifacts to {evaluation_dir}')
        achieved_returns = []
        achieved_horizons = []
    while True:
        if args.interactive:
            inp = choose_interactive_policy(pareto_front)
            if inp is None:
                print('interactive evaluation finished')
                break
        else:
            inp += 1
            if inp >= len(pareto_front):
                break
        desired_return = pareto_front[inp]
        desired_horizon = None if pareto_horizons is None else pareto_horizons[inp]

        episode_seed = None if args.seed is None or args.seed < 0 else int(args.seed) + inp
        transitions = run_episode(
            env, model, desired_return, max_return, desired_horizon=desired_horizon, seed=episode_seed
        )
        gamma = 1
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].reward += gamma * transitions[i + 1].reward
        return_ = transitions[0].reward.flatten()
        achieved_horizon = float(len(transitions))
        horizon_text = '' if desired_horizon is None else f', desired-horizon: {float(desired_horizon):.2f}'
        print(
            f'ran model with desired-return: {desired_return.flatten()}{horizon_text}, '
            f'got {return_}, achieved-length: {achieved_horizon:.2f}'
        )
        if not args.interactive:
            achieved_returns.append(return_.copy())
            achieved_horizons.append(achieved_horizon)
    if not args.interactive and achieved_returns:
        achieved_returns = np.asarray(achieved_returns, dtype=np.float32)
        save_eval_pareto_front_comparison(
            pareto_front,
            achieved_returns,
            evaluation_dir / 'logged_vs_achieved_pareto_front.png',
        )
        n_objectives = achieved_returns.shape[-1]
        has_horizon = pareto_horizons is not None

        def write_returns_csv(path, indexes):
            header = (
                ['command_index']
                + [f'desired_{o}' for o in range(n_objectives)]
                + [f'achieved_{o}' for o in range(n_objectives)]
            )
            if has_horizon:
                header += ['desired_horizon', 'achieved_horizon']
            rows = [','.join(header)]
            for index in indexes:
                index = int(index)
                desired = np.asarray(pareto_front[index], dtype=np.float32).flatten()
                achieved = achieved_returns[index].flatten()
                row = (
                    [str(index)]
                    + [f'{value:.6f}' for value in desired]
                    + [f'{value:.6f}' for value in achieved]
                )
                if has_horizon:
                    row += [f'{float(pareto_horizons[index]):.2f}', str(int(achieved_horizons[index]))]
                rows.append(','.join(row))
            path.write_text('\n'.join(rows) + '\n', encoding='utf-8')

        # every evaluated command (in front order): desired vs achieved
        write_returns_csv(evaluation_dir / 'all_command_returns.csv', range(len(achieved_returns)))

        # the non-dominated subset of achieved returns (the achieved Pareto front),
        # using the same non_dominated filter the plot uses
        _, nd_mask = non_dominated(achieved_returns, return_indexes=True)
        nd_indexes = np.nonzero(nd_mask)[0]
        write_returns_csv(evaluation_dir / 'non_dominated_achieved.csv', nd_indexes)

        print(
            f'evaluated {len(achieved_returns)} commands; '
            f'non-dominated achieved points: {len(nd_indexes)} '
            f'(saved to {evaluation_dir})'
        )
