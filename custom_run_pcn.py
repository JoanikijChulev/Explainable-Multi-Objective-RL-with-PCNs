import argparse
import re

import torch

from artifacts import create_custom_run_dir, load_run_metadata, output_artifacts_dir
from device_utils import device_description
from eval_pcn import resolve_checkpoint_path
from pcn.env_setup import build_experiment_setup
from render_pcn import (
    device,
    format_return,
    infer_env_name,
    load_logged_front,
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
        raise FileNotFoundError('no run directories found under output_artifacts/')

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


def prompt_command(pareto_front, objective_count):
    print('logged command suggestions:')
    for index, point in enumerate(pareto_front):
        print(f'  [{index}] desired_return={format_return(point)}')

    raw = input(
        'choose a logged command index, or type a desired return vector (comma or space separated) -> '
    ).strip()

    if raw.isdigit():
        selected = int(raw)
        if not (0 <= selected < len(pareto_front)):
            raise ValueError(f'policy index out of range: choose 0 to {len(pareto_front) - 1}')
        return pareto_front[selected].copy()

    pieces = [piece for piece in re.split(r'[\s,]+', raw) if piece]
    if len(pieces) != objective_count:
        raise ValueError(
            f'expected {objective_count} desired-return values, got {len(pieces)}'
        )
    return torch.tensor([float(piece) for piece in pieces], dtype=torch.float32).numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interactively choose a PCN checkpoint and custom desired return, then render the rollout.'
    )
    parser.add_argument(
        'model',
        nargs='?',
        default=None,
        type=str,
        help='run directory path, output_artifacts run name, or checkpoint path',
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
    checkpoint_path = prompt_checkpoint(run_dir, checkpoint_from_model=checkpoint_from_model)
    pareto_front = load_logged_front(run_dir / 'log.h5')
    objective_count = int(pareto_front.shape[-1])
    desired_return = prompt_command(pareto_front, objective_count)

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
        render_result=render_result,
        fps=fps,
    )

    print(f'using device: {device_description(device)}')
    print(f'using checkpoint: {checkpoint_path.name}')
    print(f'saved custom render artifacts to {policy_dir}')
    print(
        f'desired-return {format_return(desired_return)}, '
        f'achieved {format_return(render_result["achieved_return"])}, '
        f'achieved-length {int(render_result["achieved_horizon"])}'
    )
