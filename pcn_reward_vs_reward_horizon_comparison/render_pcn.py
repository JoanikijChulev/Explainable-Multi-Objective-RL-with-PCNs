import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from artifacts import create_animation_render_dir, load_run_metadata, resolve_run_dir
from device_utils import device_description, preferred_device
from eval_pcn import (
    choose_interactive_policy,
    is_horizon_conditioned,
    load_logged_commands,
    resolve_checkpoint_path,
)
from pcn.env_setup import build_experiment_setup, canonical_env_name
from pcn.pcn import action_mask_for_env, apply_action_mask


device = preferred_device()


def format_return(vector):
    return np.array2string(np.asarray(vector), precision=3, floatmode='fixed')


def resolve_run_dir_and_checkpoint(model_arg):
    candidate = Path(model_arg)
    if candidate.is_file() and candidate.suffix == '.pt':
        checkpoint_path = candidate.resolve()
        if (checkpoint_path.parent / 'log.h5').is_file():
            return checkpoint_path.parent.resolve(), checkpoint_path
        if checkpoint_path.parent.name == 'checkpoints' and (checkpoint_path.parent.parent / 'log.h5').is_file():
            return checkpoint_path.parent.parent.resolve(), checkpoint_path
        raise FileNotFoundError(f'could not infer run directory from checkpoint path: {candidate}')
    return resolve_run_dir(model_arg), None


def infer_env_name(model_dir, metadata, env_override=None):
    if env_override is not None:
        return canonical_env_name(env_override)

    env_name = metadata.get('env')
    if env_name is None:
        for candidate in (
            'dst',
            'collect_two',
            'branch-path',
            'reward-line',
            'three-tree',
            'minecart',
            'mo-mountaincar-timespeed-v0',
            'mo-lunar-lander-v3',
            'mo-reacher-v5',
            'fruit-tree-v0',
            'four-room-v0',
            'resource-gathering-v0',
            'breakable-bottles-v0',
        ):
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
    if env_name is None:
        raise ValueError(f'could not infer environment for run directory: {model_dir}')
    return canonical_env_name(env_name)


def render_mode_for_env(env_name):
    return 'rgb_array'


def to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    return value


def documented_observation(env_name, env, model_observation):
    base_env = env.unwrapped

    if env_name == 'dst':
        if hasattr(base_env, 's') and hasattr(base_env, 'shape'):
            state_index = int(base_env.s)
            row, col = np.unravel_index(state_index, base_env.shape)
        else:
            row, col = np.asarray(base_env.current_state, dtype=np.int32)
            state_index = int(np.ravel_multi_index((row, col), base_env.sea_map.shape))
        return {
            'state_index': state_index,
            'grid_position': [int(row), int(col)],
        }

    if env_name == 'collect_two':
        return {
            'position': [int(value) for value in base_env.position],
            'collected': [int(value) for value in base_env.collected],
            'collection_order': list(base_env.collection_order),
            'steps': int(base_env.steps),
        }

    if env_name == 'branch-path':
        return {
            'position': [int(value) for value in base_env.position],
            'collected': [int(value) for value in base_env.collected],
            'collection_order': list(base_env.collection_order),
            'steps': int(base_env.steps),
            'total_A_collected': int(base_env._get_info()['total_A_collected']),
            'total_B_collected': int(base_env._get_info()['total_B_collected']),
        }

    if env_name == 'reward-line':
        return base_env._get_info()

    if env_name == 'three-tree':
        return base_env._get_info()

    if env_name == 'minecart':
        state = np.asarray(base_env.get_state(), dtype=np.float32)
        orientation_degrees = float(np.degrees(np.arctan2(state[3], state[4])) % 360.0)
        return {
            'position': to_serializable(state[:2]),
            'speed': float(state[2]),
            'orientation_sin_cos': to_serializable(state[3:5]),
            'orientation_degrees': orientation_degrees,
            'content': to_serializable(state[5:]),
        }

    if env_name == 'fruit-tree-v0':
        state = np.asarray(base_env.current_state, dtype=np.int32)
        return {
            'row': int(state[0]),
            'index_in_row': int(state[1]),
        }

    if env_name == 'four-room-v0':
        (row, col), collected = base_env.state
        return {
            'position': [int(row), int(col)],
            'collected_items': [int(value) for value in collected],
        }

    if env_name == 'resource-gathering-v0':
        return {
            'x': int(base_env.current_pos[0]),
            'y': int(base_env.current_pos[1]),
            'has_gold': int(base_env.has_gold),
            'has_gem': int(base_env.has_gem),
        }

    if env_name == 'breakable-bottles-v0':
        return {
            'location': int(base_env.location),
            'bottles_carrying': int(base_env.bottles_carrying),
            'bottles_delivered': int(base_env.bottles_delivered),
            'bottles_dropped': [int(value) for value in base_env.bottles_dropped],
        }

    return to_serializable(model_observation)


def format_observation_lines(observation):
    observation = to_serializable(observation)
    if isinstance(observation, dict):
        lines = ['observation:']
        for key, value in observation.items():
            lines.append(f'  {key}={json.dumps(value)}')
        return lines
    return [f'observation={json.dumps(observation)}']


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


def _frame_to_image(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.dtype != np.uint8:
        if np.max(frame) <= 1.0:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame)


def _annotate_frame(frame, info_lines):
    base = _frame_to_image(frame)
    font = ImageFont.load_default()
    probe = Image.new('RGB', (1, 1), color='white')
    draw = ImageDraw.Draw(probe)
    boxes = [draw.textbbox((0, 0), line or ' ', font=font) for line in info_lines]
    line_height = max((box[3] - box[1]) for box in boxes) + 4
    text_width = max((box[2] - box[0]) for box in boxes)
    banner_height = 12 + line_height * len(info_lines)
    width = max(base.width, text_width + 16)

    canvas = Image.new('RGB', (width, base.height + banner_height), color='white')
    canvas.paste(base, ((width - base.width) // 2, banner_height))
    draw = ImageDraw.Draw(canvas)
    y = 6
    for line in info_lines:
        draw.text((8, y), line, fill='black', font=font)
        y += line_height
    return canvas


def capture_annotated_frame(
    env,
    display_observation,
    step_index,
    desired_return,
    remaining_return,
    achieved_return,
    desired_horizon=None,
    remaining_horizon=None,
    action=None,
    reward=None,
):
    frame = env.render()
    action_text = 'reset' if action is None else str(action)
    reward_text = 'n/a' if reward is None else format_return(reward)
    lines = [
        f'step={step_index} action={action_text} reward={reward_text}',
        f'desired_return={format_return(desired_return)}',
        f'remaining_return={format_return(remaining_return)}',
        f'achieved_return={format_return(achieved_return)}',
    ]
    if desired_horizon is not None:
        lines.insert(2, f'desired_horizon={float(desired_horizon):.3f}')
        lines.insert(4, f'remaining_horizon={float(remaining_horizon):.3f}')
    lines.extend(format_observation_lines(display_observation))
    return _annotate_frame(frame, lines)


def rollout_episode_with_frames(env_name, env, model, desired_return, max_return, desired_horizon=None):
    desired_return = np.asarray(desired_return, dtype=np.float32).copy()
    remaining_return = desired_return.copy()
    achieved_return = np.zeros_like(desired_return, dtype=np.float32)
    desired_horizon = None if desired_horizon is None else np.float32(desired_horizon)
    remaining_horizon = desired_horizon

    frames = []
    trajectory = []

    obs, _ = env.reset()
    display_observation = documented_observation(env_name, env, obs)
    frames.append(
        capture_annotated_frame(
            env,
            display_observation=display_observation,
            step_index=0,
            desired_return=desired_return,
            remaining_return=remaining_return,
            achieved_return=achieved_return,
            desired_horizon=desired_horizon,
            remaining_horizon=remaining_horizon,
        )
    )

    step_index = 0
    terminated = truncated = False
    while not (terminated or truncated):
        display_observation = documented_observation(env_name, env, obs)
        command_return = remaining_return.copy()
        command_horizon = remaining_horizon
        action = greedy_action(env, model, obs, command_return, desired_horizon=command_horizon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_display_observation = documented_observation(env_name, env, next_obs)
        reward = np.asarray(reward, dtype=np.float32)
        achieved_return = achieved_return + reward
        remaining_return = np.clip(remaining_return - reward, None, max_return, dtype=np.float32)
        if remaining_horizon is not None:
            remaining_horizon = np.float32(max(float(remaining_horizon) - 1.0, 1.0))
        step_index += 1

        transition = {
            'step': step_index,
            'observation': to_serializable(display_observation),
            'action': action,
            'reward': to_serializable(reward),
            'achieved_return': to_serializable(achieved_return),
            'command_return_before_step': to_serializable(command_return),
            'remaining_return_after_step': to_serializable(remaining_return),
            'terminated': bool(terminated),
            'truncated': bool(truncated),
            'info': to_serializable(info),
        }
        if desired_horizon is not None:
            transition['command_horizon_before_step'] = float(command_horizon)
            transition['remaining_horizon_after_step'] = float(remaining_horizon)
        trajectory.append(transition)

        obs = next_obs
        frames.append(
            capture_annotated_frame(
                env,
                display_observation=next_display_observation,
                step_index=step_index,
                desired_return=desired_return,
                remaining_return=remaining_return,
                achieved_return=achieved_return,
                desired_horizon=desired_horizon,
                remaining_horizon=remaining_horizon,
                action=action,
                reward=reward,
            )
        )

    return {
        'frames': frames,
        'trajectory': trajectory,
        'achieved_return': achieved_return,
        'achieved_horizon': step_index,
        'terminated': bool(terminated),
        'truncated': bool(truncated),
    }


def save_step_frames(frames, steps_dir):
    steps_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(steps_dir / f'step_{index:04d}.png')


def save_gif(frames, output_path, fps):
    duration_ms = max(int(round(1000 / max(fps, 1))), 1)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def save_mp4_from_frames(steps_dir, output_path, fps):
    input_pattern = steps_dir / 'step_%04d.png'
    command = [
        'ffmpeg',
        '-y',
        '-loglevel',
        'error',
        '-framerate',
        str(fps),
        '-i',
        str(input_pattern),
        '-vf',
        'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-pix_fmt',
        'yuv420p',
        str(output_path),
    ]
    subprocess.run(command, check=True)


def save_policy_render_artifacts(
    policy_dir,
    policy_index,
    checkpoint_path,
    env_name,
    desired_return,
    desired_horizon,
    render_result,
    fps,
):
    steps_dir = policy_dir / 'steps'
    save_step_frames(render_result['frames'], steps_dir)
    save_gif(render_result['frames'], policy_dir / 'animation.gif', fps=fps)

    mp4_status = {'created': False, 'error': None}
    try:
        save_mp4_from_frames(steps_dir, policy_dir / 'animation.mp4', fps=fps)
        mp4_status['created'] = True
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        mp4_status['error'] = str(exc)

    summary = {
        'policy_index': policy_index,
        'env': env_name,
        'checkpoint': checkpoint_path.name,
        'desired_return': to_serializable(np.asarray(desired_return, dtype=np.float32)),
        'desired_horizon': None if desired_horizon is None else float(desired_horizon),
        'achieved_return': to_serializable(render_result['achieved_return']),
        'achieved_horizon': int(render_result['achieved_horizon']),
        'terminated': bool(render_result['terminated']),
        'truncated': bool(render_result['truncated']),
        'fps': int(fps),
        'frame_count': len(render_result['frames']),
        'gif_path': 'animation.gif',
        'mp4_path': 'animation.mp4' if mp4_status['created'] else None,
        'mp4_error': mp4_status['error'],
    }
    with (policy_dir / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')
    with (policy_dir / 'trajectory.json').open('w', encoding='utf-8') as handle:
        json.dump(render_result['trajectory'], handle, indent=2)
        handle.write('\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render PCN policy rollouts to step frames, GIFs and MP4 videos.')
    parser.add_argument('model', type=str, help='run directory path, comparison_output_artifacts run name, or checkpoint path')
    parser.add_argument('--env', default=None, type=str, help='override environment if it cannot be inferred')
    parser.add_argument(
        '--checkpoint',
        default=None,
        type=str,
        help='checkpoint to render: default is latest, or pass a checkpoint number, filename, or full path',
    )
    parser.add_argument('--policy-index', default=None, type=int, help='render only one Pareto-front policy by index')
    parser.add_argument('--interactive', action='store_true', help='interactively choose one policy to render')
    parser.add_argument('--fps', default=2, type=int, help='output animation frames per second')
    args = parser.parse_args()

    run_dir, checkpoint_from_model = resolve_run_dir_and_checkpoint(args.model)
    metadata = load_run_metadata(run_dir)
    horizon_conditioned = is_horizon_conditioned(metadata)
    env_name = infer_env_name(run_dir, metadata, env_override=args.env)
    if env_name.startswith('walkroom'):
        parser.error(
            'walkroom rendering is unsupported: this custom environment does not provide a working '
            'step-by-step render for animation output.'
        )
    fruit_tree_depth = metadata.get('fruit_tree_depth', 6)
    if env_name == 'fruit-tree-v0' and 'fruit_tree_depth' not in metadata:
        print('warning: fruit-tree run metadata is missing fruit_tree_depth; defaulting to depth 6')

    render_mode = render_mode_for_env(env_name)
    setup = build_experiment_setup(
        env_name,
        device=device,
        fruit_tree_depth=fruit_tree_depth,
        include_model=False,
        render_mode=render_mode,
    )
    env = setup.env

    checkpoint_arg = args.checkpoint if args.checkpoint is not None else checkpoint_from_model
    checkpoint_path, _ = resolve_checkpoint_path(run_dir, checkpoint_arg)
    model = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device)
    if hasattr(model, 'scaling_factor') and torch.is_tensor(model.scaling_factor):
        model.scaling_factor = model.scaling_factor.to(device)

    pareto_front, pareto_horizons = load_logged_commands(
        run_dir / 'log.h5',
        horizon_conditioned=horizon_conditioned,
    )
    if len(pareto_front) == 0:
        raise ValueError(f'no non-dominated logged policies available in {run_dir / "log.h5"}')

    if args.interactive:
        selected = choose_interactive_policy(pareto_front)
        if selected is None:
            print('interactive rendering finished')
            raise SystemExit(0)
        policy_indexes = [selected]
    elif args.policy_index is not None:
        if not (0 <= args.policy_index < len(pareto_front)):
            parser.error(f'policy index out of range: choose 0 to {len(pareto_front) - 1}')
        policy_indexes = [args.policy_index]
    else:
        policy_indexes = list(range(len(pareto_front)))

    fps = args.fps
    if fps is None:
        fps = int(getattr(env.unwrapped, 'metadata', {}).get('render_fps', 4))
        fps = max(fps, 1)

    render_root = create_animation_render_dir(run_dir)
    print(f'using device: {device_description(device)}')
    print(f'using checkpoint: {checkpoint_path.name}')
    if horizon_conditioned:
        print('detected horizon-conditioned PCN run')
    print(f'rendering {len(policy_indexes)} policy rollout(s) to {render_root}')

    root_summary = []
    try:
        for policy_index in policy_indexes:
            desired_return = pareto_front[policy_index]
            desired_horizon = None
            if pareto_horizons is not None:
                desired_horizon = pareto_horizons[policy_index]
            policy_dir = render_root / f'policy_{policy_index:03d}'
            policy_dir.mkdir(parents=True, exist_ok=False)

            render_result = rollout_episode_with_frames(
                env_name,
                env,
                model,
                desired_return=desired_return,
                desired_horizon=desired_horizon,
                max_return=setup.max_return,
            )
            summary = save_policy_render_artifacts(
                policy_dir,
                policy_index=policy_index,
                checkpoint_path=checkpoint_path,
                env_name=env_name,
                desired_return=desired_return,
                desired_horizon=desired_horizon,
                render_result=render_result,
                fps=fps,
            )
            root_summary.append(summary)
            horizon_text = ''
            if desired_horizon is not None:
                horizon_text = f', desired-horizon {float(desired_horizon):.3f}'
            print(
                f'rendered policy {policy_index}: desired-return {format_return(desired_return)}'
                f'{horizon_text}, achieved {format_return(render_result["achieved_return"])}, '
                f'achieved-length {int(render_result["achieved_horizon"])}'
            )
    finally:
        env.close()

    with (render_root / 'render_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(root_summary, handle, indent=2, sort_keys=True)
        handle.write('\n')
