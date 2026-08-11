import argparse
import re

import numpy as np
import torch

from artifacts import create_custom_run_dir, load_run_metadata, output_artifacts_dir
from device_utils import device_description
from eval_pcn import is_horizon_conditioned, load_logged_commands, resolve_checkpoint_path
from pcn.env_setup import build_experiment_setup
from render_pcn import (
    device,
    format_return,
    infer_env_name,
    render_mode_for_env,
    resolve_run_dir_and_checkpoint,
    rollout_episode_with_frames,
    save_policy_render_artifacts,
)


def prompt_checkpoint(run_dir, checkpoint_from_model=None):
    if checkpoint_from_model is not None:
        return checkpoint_from_model

    default_checkpoint, checkpoints = resolve_checkpoint_path(run_dir, None)
    print('available checkpoints:')
    for checkpoint in checkpoints:
        print(f'  - {checkpoint.name}')

    raw = input(f'select checkpoint [default: {default_checkpoint.name}] -> ').strip()
    if not raw:
        return default_checkpoint
    selected_checkpoint, _ = resolve_checkpoint_path(run_dir, raw)
    return selected_checkpoint


def prompt_run():
    runs = sorted(path.name for path in output_artifacts_dir().iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError('no run directories found under comparison_output_artifacts/')

    print('available runs:')
    for index, run_name in enumerate(runs):
        print(f'  [{index}] {run_name}')

    raw = input('select a run by index or name -> ').strip()
    if raw.isdigit():
        selected = int(raw)
        if not (0 <= selected < len(runs)):
            raise ValueError(f'run index out of range: choose 0 to {len(runs) - 1}')
        return runs[selected]
    if raw in runs:
        return raw
    raise ValueError(f'unknown run selection: {raw}')


def prompt_horizon(pareto_front, pareto_horizons, desired_return):
    nearest = int(np.argmin(np.linalg.norm(pareto_front - desired_return[None, :], axis=1)))
    default_horizon = float(pareto_horizons[nearest])
    raw = input(
        f'desired horizon [default: {default_horizon:.1f}, from nearest logged command] -> '
    ).strip()
    if not raw:
        return np.float32(default_horizon)
    return np.float32(float(raw))


def prompt_command(pareto_front, objective_count, pareto_horizons=None):
    print('logged command suggestions:')
    for index, point in enumerate(pareto_front):
        horizon_text = ''
        if pareto_horizons is not None:
            horizon_text = f' desired_horizon={float(pareto_horizons[index]):.1f}'
        print(f'  [{index}] desired_return={format_return(point)}{horizon_text}')

    raw = input(
        'choose a logged command index, or type a desired return vector (comma or space separated) -> '
    ).strip()

    if raw.isdigit():
        selected = int(raw)
        if not (0 <= selected < len(pareto_front)):
            raise ValueError(f'policy index out of range: choose 0 to {len(pareto_front) - 1}')
        desired_return = pareto_front[selected].copy()
        desired_horizon = None
        if pareto_horizons is not None:
            desired_horizon = np.float32(pareto_horizons[selected])
        return desired_return, desired_horizon

    pieces = [piece for piece in re.split(r'[\s,]+', raw) if piece]
    if len(pieces) != objective_count:
        raise ValueError(
            f'expected {objective_count} desired-return values, got {len(pieces)}'
        )
    desired_return = np.asarray([float(piece) for piece in pieces], dtype=np.float32)
    desired_horizon = None
    if pareto_horizons is not None:
        desired_horizon = prompt_horizon(pareto_front, pareto_horizons, desired_return)
    return desired_return, desired_horizon


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interactively choose a PCN checkpoint and custom desired return, then render the rollout.'
    )
    parser.add_argument(
        'model',
        nargs='?',
        default=None,
        type=str,
        help='run directory path, comparison_output_artifacts run name, or checkpoint path',
    )
    parser.add_argument('--fps', default=2, type=int, help='output animation frames per second')
    args = parser.parse_args()

    if args.model is None:
        args.model = prompt_run()

    run_dir, checkpoint_from_model = resolve_run_dir_and_checkpoint(args.model)
    metadata = load_run_metadata(run_dir)
    env_name = infer_env_name(run_dir, metadata)
    if env_name.startswith('walkroom'):
        parser.error(
            'walkroom rendering is unsupported: this custom environment does not provide a working '
            'step-by-step render for animation output.'
        )
    fruit_tree_depth = metadata.get('fruit_tree_depth', 6)
    if env_name == 'fruit-tree-v0' and 'fruit_tree_depth' not in metadata:
        print('warning: fruit-tree run metadata is missing fruit_tree_depth; defaulting to depth 6')

    setup = build_experiment_setup(
        env_name,
        device=device,
        fruit_tree_depth=fruit_tree_depth,
        include_model=False,
        render_mode=render_mode_for_env(env_name),
    )
    env = setup.env
    horizon_conditioned = is_horizon_conditioned(metadata)
    if horizon_conditioned:
        print('detected horizon-conditioned PCN run')
    checkpoint_path = prompt_checkpoint(run_dir, checkpoint_from_model=checkpoint_from_model)
    pareto_front, pareto_horizons = load_logged_commands(
        run_dir / 'log.h5',
        horizon_conditioned=horizon_conditioned,
    )
    objective_count = int(pareto_front.shape[-1])
    desired_return, desired_horizon = prompt_command(
        pareto_front,
        objective_count,
        pareto_horizons=pareto_horizons,
    )

    fps = args.fps if args.fps is not None else int(getattr(env.unwrapped, 'metadata', {}).get('render_fps', 4))
    fps = max(fps, 1)

    model = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device)
    model.scaling_factor = model.scaling_factor.to(device)

    policy_dir = create_custom_run_dir(run_dir)

    try:
        render_result = rollout_episode_with_frames(
            env_name,
            env,
            model,
            desired_return=desired_return,
            desired_horizon=desired_horizon,
            max_return=setup.max_return,
        )
    finally:
        env.close()

    save_policy_render_artifacts(
        policy_dir,
        policy_index=-1,
        checkpoint_path=checkpoint_path,
        env_name=env_name,
        desired_return=desired_return,
        desired_horizon=desired_horizon,
        render_result=render_result,
        fps=fps,
    )

    horizon_text = ''
    if desired_horizon is not None:
        horizon_text = f', desired-horizon {float(desired_horizon):.1f}'
    print(f'using device: {device_description(device)}')
    print(f'using checkpoint: {checkpoint_path.name}')
    print(f'saved custom render artifacts to {policy_dir}')
    print(
        f'desired-return {format_return(desired_return)}{horizon_text}, '
        f'achieved {format_return(render_result["achieved_return"])}, '
        f'achieved-length {int(render_result["achieved_horizon"])}'
    )
