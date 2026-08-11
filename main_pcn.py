import argparse

import torch

from artifacts import (
    create_run_dir,
    load_run_metadata,
    resolve_run_dir,
    training_state_path,
    validate_run_name,
    write_run_metadata,
)
from device_utils import device_description, preferred_device
from eval_pcn import resolve_checkpoint_path
from pcn.env_setup import build_experiment_setup


if __name__ == '__main__':
    from pcn.pcn import load_training_state, train
    from plot_artifacts import save_training_plots

    parser = argparse.ArgumentParser(description='PCN')
    parser.add_argument(
        '--env',
        default=None,
        type=str,
        help='dst, collect_two/c2, branch-path/bp, reward-line/rl, three-tree/tt, minecart/mc, walkroom2...walkroom9, fruit-tree-v0/ft, four-room-v0/4room, resource-gathering-v0/rsg, breakable-bottles-v0/bb',
    )
    parser.add_argument(
        '--run-name',
        default=None,
        type=str,
        help='name of the output_artifacts subfolder for this training run',
    )
    parser.add_argument(
        '--resume-run',
        default=None,
        type=str,
        help='existing run directory path or output_artifacts run name to resume in place',
    )
    parser.add_argument('--model', default=None, type=str, help='load model')
    parser.add_argument(
        '--total-steps',
        default=None,
        type=int,
        help='override total training step budget; when resuming, this is the new absolute total-step target',
    )
    parser.add_argument(
        '--fruit-tree-depth',
        default=6,
        type=int,
        help='depth for fruit-tree-v0 (5, 6 or 7)',
    )
    args = parser.parse_args()

    device = preferred_device()
    print(f'using device: {device_description(device)}')

    if args.resume_run is not None and args.model is not None:
        parser.error('--resume-run cannot be combined with --model')

    resume_state = None
    resume_metadata = {}
    logdir = None
    metadata = {}
    gamma = 1.0
    n_step_episodes = 10

    if args.resume_run is None:
        if args.env is None:
            parser.error('--env is required unless --resume-run is set')
        if args.run_name is None:
            parser.error('--run-name is required unless --resume-run is set')

        try:
            validate_run_name(args.run_name)
        except ValueError as exc:
            parser.error(str(exc))

        try:
            setup = build_experiment_setup(
                args.env,
                device=device,
                fruit_tree_depth=args.fruit_tree_depth,
            )
        except (ImportError, ValueError) as exc:
            parser.error(str(exc))

        env = setup.env
        model = setup.model

        if args.model is not None:
            model = torch.load(args.model, map_location=device, weights_only=False).to(device)
            model.scaling_factor = model.scaling_factor.to(device)

        try:
            logdir = create_run_dir(args.run_name)
        except FileExistsError as exc:
            parser.error(str(exc))

        total_steps = int(args.total_steps if args.total_steps is not None else setup.total_steps)
        metadata = {
            'run_name': args.run_name,
            'env': setup.env_name,
            'model_source': args.model,
            'device': device,
            'learning_rate': setup.learning_rate,
            'min_learning_rate': setup.min_learning_rate,
            'batch_size': setup.batch_size,
            'total_steps': total_steps,
            'n_model_updates': setup.n_model_updates,
            'n_step_episodes': n_step_episodes,
            'n_er_episodes': setup.n_er_episodes,
            'gamma': gamma,
            'max_size': setup.max_size,
            'reward_neutral_suffix_trimming': True,
            'reward_time_tiebreak_commands': True,
            'max_return': setup.max_return.astype('float32').tolist(),
            'ref_point': setup.ref_point.astype('float32').tolist(),
        }
        metadata.update(setup.metadata)
    else:
        try:
            logdir = resolve_run_dir(args.resume_run)
        except FileNotFoundError as exc:
            parser.error(str(exc))

        resume_metadata = load_run_metadata(logdir)
        env_name = resume_metadata.get('env', args.env)
        if env_name is None:
            parser.error('could not infer env for resume run; pass --env explicitly')
        fruit_tree_depth = int(resume_metadata.get('fruit_tree_depth', args.fruit_tree_depth))
        gamma = float(resume_metadata.get('gamma', gamma))
        n_step_episodes = int(resume_metadata.get('n_step_episodes', n_step_episodes))

        state_path = training_state_path(logdir)
        if not state_path.is_file():
            parser.error(
                f'could not find resume state file: {state_path}. '
                'This run predates resume support; use --model on the latest checkpoint to warm-start a new run.'
            )

        try:
            checkpoint_path, _ = resolve_checkpoint_path(logdir, None)
        except FileNotFoundError as exc:
            parser.error(str(exc))

        try:
            setup = build_experiment_setup(
                env_name,
                device=device,
                fruit_tree_depth=fruit_tree_depth,
            )
        except (ImportError, ValueError) as exc:
            parser.error(str(exc))

        env = setup.env
        model = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device)
        model.scaling_factor = model.scaling_factor.to(device)
        resume_state = load_training_state(logdir, map_location='cpu')

        resumed_step = int(resume_state['step'])
        total_steps = int(args.total_steps if args.total_steps is not None else resume_metadata.get('total_steps', setup.total_steps))
        if total_steps <= resumed_step:
            parser.error(
                f'--total-steps must be greater than the resumed step count ({resumed_step})'
            )
        metadata = {
            'run_name': resume_metadata.get('run_name', logdir.name),
            'env': setup.env_name,
            'model_source': resume_metadata.get('model_source'),
            'learning_rate': setup.learning_rate,
            'min_learning_rate': setup.min_learning_rate,
            'batch_size': setup.batch_size,
            'n_model_updates': setup.n_model_updates,
            'n_step_episodes': n_step_episodes,
            'n_er_episodes': setup.n_er_episodes,
            'gamma': gamma,
            'max_size': setup.max_size,
            'reward_neutral_suffix_trimming': True,
            'reward_time_tiebreak_commands': True,
            'max_return': setup.max_return.astype('float32').tolist(),
            'ref_point': setup.ref_point.astype('float32').tolist(),
        }
        metadata.update(
            {
                'device': device,
                'total_steps': total_steps,
                'resume_checkpoint': checkpoint_path.name,
                'resumed_from_step': resumed_step,
            }
        )
        metadata.update(setup.metadata)

    write_run_metadata(logdir, metadata)

    train(
        env,
        model,
        learning_rate=setup.learning_rate,
        min_learning_rate=setup.min_learning_rate,
        batch_size=setup.batch_size,
        total_steps=total_steps,
        n_model_updates=setup.n_model_updates,
        n_step_episodes=n_step_episodes,
        n_er_episodes=setup.n_er_episodes,
        gamma=gamma,
        max_size=setup.max_size,
        max_return=setup.max_return,
        ref_point=setup.ref_point,
        logdir=logdir,
        resume_state=resume_state,
    )
    save_training_plots(logdir)
