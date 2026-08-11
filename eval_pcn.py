import re

import numpy as np
import torch

from artifacts import checkpoints_dir, create_evaluation_dir, load_run_metadata, resolve_run_dir
from device_utils import device_description, preferred_device
from pcn.env_setup import build_experiment_setup, canonical_env_name
from pcn.pcn import Transition, action_mask_for_env, apply_action_mask
from plot_artifacts import save_eval_pareto_front_comparison


device = preferred_device()


def objective_return_labels(env_name, n_objectives):
    if canonical_env_name(env_name) == 'minecart' and int(n_objectives) == 3:
        return ['Ore 0 Return', 'Ore 1 Return', 'Fuel Return']
    return [f'Objective {index} Return' for index in range(int(n_objectives))]


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


def load_logged_front(log, objectives):
    if 'train/leaves/r/ndarray' in log:
        pareto_front = np.asarray(log['train/leaves/r/ndarray'][-1], dtype=np.float32)
        pareto_front = pareto_front[~np.isnan(pareto_front).any(axis=1)]
    elif 'eval/return/desired/ndarray' in log:
        pareto_front = np.asarray(log['eval/return/desired/ndarray'][-1], dtype=np.float32)
        pareto_front = pareto_front[~np.isnan(pareto_front).any(axis=1)]
        print('warning: train/leaves data is missing; falling back to the last logged eval desired returns')
    else:
        raise KeyError(
            'run log does not contain train/leaves or eval/return/desired data; '
            'cannot recover commands for evaluation'
        )

    _, pareto_front_i = non_dominated(pareto_front[:, objectives], return_indexes=True)
    pareto_front = pareto_front[pareto_front_i]
    order = np.argsort(pareto_front, axis=0)
    return pareto_front[order[:, 0]]


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


def greedy_action(env, model, obs, desired_return):
    obs = np.asarray([obs])
    desired_return = np.asarray([desired_return], dtype=np.float32)
    log_probs = model(
        torch.as_tensor(obs).to(device),
        torch.as_tensor(desired_return).to(device),
    )
    log_probs = log_probs.detach().cpu().numpy()[0]
    action_mask = action_mask_for_env(env)
    masked_log_probs = apply_action_mask(log_probs, action_mask)
    return int(np.argmax(masked_log_probs))


def run_episode(env, model, desired_return, max_return):
    transitions = []
    obs, _ = env.reset()
    terminated = truncated = False
    remaining_return = np.asarray(desired_return, dtype=np.float32).copy()
    while not (terminated or truncated):
        action = greedy_action(env, model, obs, remaining_return)
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
    return transitions


if __name__ == '__main__':
    import argparse
    import h5py

    parser = argparse.ArgumentParser(description='PCN')
    parser.add_argument('model', type=str, help='run directory path or output_artifacts run name')
    parser.add_argument('--interactive', action='store_true', help='interactive policy selection')
    parser.add_argument(
        '--checkpoint',
        default=None,
        type=str,
        help='checkpoint to evaluate: default is latest, or pass a checkpoint number, filename, or full path',
    )
    parser.set_defaults(interactive=False)
    args = parser.parse_args()
    try:
        model_dir = resolve_run_dir(args.model)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    metadata = load_run_metadata(model_dir)

    env_name = metadata.get('env')
    if env_name is None:
        for candidate in ('dst', 'collect_two', 'branch-path', 'reward-line', 'three-tree', 'minecart', 'fruit-tree-v0', 'four-room-v0', 'resource-gathering-v0', 'breakable-bottles-v0'):
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
    objectives = range(max_return.shape[-1])

    log = model_dir / 'log.h5'
    log = h5py.File(log)
    try:
        checkpoint_path, checkpoints = resolve_checkpoint_path(model_dir, args.checkpoint)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    model = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device)
    model.scaling_factor = model.scaling_factor.to(device)
    print(f'using device: {device_description(device)}')
    print(f'using checkpoint: {checkpoint_path.name}')

    with log:
        pareto_front = load_logged_front(log, objectives)

    inp = -1
    evaluation_dir = None
    if not args.interactive:
        evaluation_dir = create_evaluation_dir(model_dir)
        print('=' * 38)
        print('not interactive, this may take a while')
        print('=' * 38)
        print(f'saving evaluation artifacts to {evaluation_dir}')
        achieved_returns = []
    while True:
        if args.interactive:
            inp = choose_interactive_policy(pareto_front[:, objectives])
            if inp is None:
                print('interactive evaluation finished')
                break
        else:
            inp += 1
            if inp >= len(pareto_front):
                break
        desired_return = pareto_front[inp]

        transitions = run_episode(env, model, desired_return, max_return)
        gamma = 1
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].reward += gamma * transitions[i + 1].reward
        return_ = transitions[0].reward.flatten()
        achieved_horizon = float(len(transitions))
        print(
            f'ran model with desired-return: {desired_return.flatten()}, '
            f'got {return_}, achieved-length: {achieved_horizon:.2f}'
        )
        if not args.interactive:
            achieved_returns.append(return_.copy())
            with open(evaluation_dir / f'policy_{inp}.txt', 'w') as f:
                f.write(', '.join(str(r) for r in return_))
    if not args.interactive and achieved_returns:
        achieved_returns = np.asarray(achieved_returns, dtype=np.float32)
        save_eval_pareto_front_comparison(
            pareto_front,
            achieved_returns,
            evaluation_dir / 'logged_vs_achieved_pareto_front.png',
            objective_labels=objective_return_labels(env_name, achieved_returns.shape[-1]),
        )
