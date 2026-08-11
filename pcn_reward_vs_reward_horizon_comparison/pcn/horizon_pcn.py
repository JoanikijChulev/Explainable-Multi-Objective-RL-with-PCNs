import heapq
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from artifacts import checkpoints_dir, training_state_path
from device_utils import preferred_device
from logger import Logger
from morl_metrics import expected_utility, hypervolume_score
from pcn.replay_artifacts import write_final_replay_commands


def crowding_distance(points):
    if len(points) == 0:
        return np.array([], dtype=np.float32)
    if len(points) <= 2:
        return np.full((len(points),), points.shape[-1], dtype=np.float32)

    # first normalize across dimensions
    points = (points - points.min(axis=0)) / (np.ptp(points, axis=0) + 1e-8)
    # sort points per dimension
    dim_sorted = np.argsort(points, axis=0)
    point_sorted = np.take_along_axis(points, dim_sorted, axis=0)
    # compute distances between lower and higher point
    distances = np.abs(point_sorted[:-2] - point_sorted[2:])
    # pad extrema's with 1, for each dimension
    distances = np.pad(distances, ((1,), (0,)), constant_values=1)
    # sum distances of each dimension of the same point
    crowding = np.zeros(points.shape)
    crowding[dim_sorted, np.arange(points.shape[-1])] = distances
    return np.sum(crowding, axis=-1)


@dataclass
class Transition(object):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminal: bool


device = preferred_device()


def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        return next_obs, reward, bool(terminated or truncated), info
    return out


def action_mask_for_env(env):
    mask_fn = getattr(env, 'get_action_mask', None)
    if not callable(mask_fn):
        get_wrapper_attr = getattr(env, 'get_wrapper_attr', None)
        if callable(get_wrapper_attr):
            try:
                mask_fn = get_wrapper_attr('get_action_mask')
            except AttributeError:
                mask_fn = None
    if not callable(mask_fn):
        return None
    action_mask = np.asarray(mask_fn(), dtype=bool)
    if action_mask.ndim != 1 or not np.any(action_mask):
        return None
    return action_mask


def apply_action_mask(log_probs, action_mask):
    if action_mask is None:
        return np.asarray(log_probs, dtype=np.float32)
    masked_log_probs = np.asarray(log_probs, dtype=np.float32).copy()
    masked_log_probs[~action_mask] = -1e9
    return masked_log_probs


def action_probabilities(log_probs, action_mask):
    masked_log_probs = apply_action_mask(log_probs, action_mask)
    probs = np.exp(masked_log_probs - np.max(masked_log_probs))
    probs = probs / np.sum(probs)
    return probs, masked_log_probs


def sample_random_action(env):
    action_mask = action_mask_for_env(env)
    if action_mask is None:
        return np.random.randint(0, env.nA)
    valid_actions = np.flatnonzero(action_mask)
    return int(np.random.choice(valid_actions))


def get_non_dominated(solutions):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1
    return is_efficient


def reward_array(reward):
    return np.asarray(reward, dtype=np.float32)


def compute_hypervolume(q_set, ref):
    nA = len(q_set)
    q_values = np.zeros(nA)
    for i in range(nA):
        q_values[i] = hypervolume_score(q_set[i], ref)
    return q_values


def relabel_episode_returns(transitions, gamma=1.0):
    # compute return
    for i in reversed(range(len(transitions) - 1)):
        transitions[i].reward = (
            reward_array(transitions[i].reward)
            + gamma * reward_array(transitions[i + 1].reward)
        )
    return transitions


def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
        except RuntimeError:
            pass
    return state


def restore_rng_state(state):
    if not state:
        return
    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    if 'torch_cuda' in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state['torch_cuda'])
        except RuntimeError:
            pass


def move_optimizer_state_to_device(opt, target_device):
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(target_device)


def save_training_state(
    run_dir,
    *,
    step,
    total_episodes,
    n_checkpoints,
    experience_replay,
    optimizer,
):
    torch.save(
        {
            'version': 1,
            'step': int(step),
            'total_episodes': int(total_episodes),
            'n_checkpoints': int(n_checkpoints),
            'experience_replay': experience_replay,
            'optimizer_state_dict': optimizer.state_dict(),
            'rng_state': capture_rng_state(),
        },
        training_state_path(run_dir),
    )


def load_training_state(run_dir, map_location=None):
    state_path = training_state_path(run_dir)
    return torch.load(state_path, map_location=map_location, weights_only=False)


def nlargest(n, experience_replay, threshold=.2):
    returns = np.array([e[2][0].reward for e in experience_replay])
    # crowding distance of each point, check ones that are too close together
    distances = crowding_distance(returns)
    sma = np.argwhere(distances <= threshold).flatten()

    nd_i = get_non_dominated(returns)
    nd = returns[nd_i]
    # we will compute distance of each point with each non-dominated point,
    # duplicate each point with number of nd to compute respective distance
    l2 = np.min(np.linalg.norm(returns[:, None, :] - nd[None, :, :], axis=-1), axis=-1) * -1
    # all points that are too close together (crowding distance < threshold) get a penalty
    nd_i = np.nonzero(nd_i)[0]
    _, unique_i = np.unique(nd, axis=0, return_index=True)
    unique_i = nd_i[unique_i]
    duplicates = np.ones(len(l2), dtype=bool)
    duplicates[unique_i] = False
    l2[duplicates] -= 1e-5
    l2[sma] *= 2

    sorted_i = np.argsort(l2)
    largest = [experience_replay[i] for i in sorted_i[-n:]]
    # before returning largest elements, update all distances in heap
    for i in range(len(l2)):
        experience_replay[i] = (l2[i], experience_replay[i][1], experience_replay[i][2])
    heapq.heapify(experience_replay)
    return largest


def add_episode(transitions, experience_replay, gamma=1.0, max_size=100, step=0):
    relabel_episode_returns(transitions, gamma=gamma)
    # pop smallest episode of heap if full, add new episode
    # heap is sorted by negative distance, (updated in nlargest)
    # put positive number to ensure that new item stays in the heap
    if len(experience_replay) == max_size:
        heapq.heappushpop(experience_replay, (1, step, transitions))
    else:
        heapq.heappush(experience_replay, (1, step, transitions))


def choose_action(model, obs, desired_return, desired_horizon, env=None):
    log_probs = model(
        torch.as_tensor(np.asarray([obs])).to(device),
        torch.as_tensor(np.asarray([desired_return], dtype=np.float32)).to(device),
        torch.as_tensor(np.asarray([desired_horizon], dtype=np.float32)).unsqueeze(1).to(device),
    )
    log_probs = log_probs.detach().cpu().numpy()[0]
    action_mask = action_mask_for_env(env) if env is not None else None
    probs, _ = action_probabilities(log_probs, action_mask)
    action = np.random.choice(np.arange(len(log_probs)), p=probs)
    return int(action)


def run_episode(env, model, desired_return, desired_horizon, max_return):
    transitions = []
    obs = reset_env(env)
    done = False
    desired_return = np.asarray(desired_return, dtype=np.float32).copy()
    desired_horizon = np.float32(desired_horizon)
    while not done:
        action = choose_action(model, obs, desired_return, desired_horizon, env=env)
        n_obs, reward, done, _ = step_env(env, action)

        transitions.append(Transition(
            observation=obs,
            action=action,
            reward=np.float32(reward).copy(),
            next_observation=n_obs,
            terminal=done
        ))

        obs = n_obs
        # clip desired return, to return-upper-bound,
        # to avoid negative returns giving impossible desired returns
        desired_return = np.clip(desired_return - reward, None, max_return, dtype=np.float32)
        # clip desired horizon to avoid negative horizons
        desired_horizon = np.float32(max(desired_horizon - 1, 1.0))
    return transitions


def choose_commands(experience_replay, n_episodes):
    # get best episodes, according to their crowding distance
    episodes = nlargest(n_episodes, experience_replay)
    returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
    # keep only non-dominated returns
    nd_i = get_non_dominated(np.array(returns))
    returns = np.array(returns)[nd_i]
    horizons = np.array(horizons)[nd_i]
    # pick random return from random best episode
    r_i = np.random.randint(0, len(returns))
    desired_horizon = np.float32(horizons[r_i] - 2)
    # mean and std per objective
    _, objective_std = np.mean(returns, axis=0), np.std(returns, axis=0)
    # desired return is sampled from [M, M+S], to try to do better than mean return
    desired_return = returns[r_i].copy()
    # random objective
    objective_index = np.random.randint(0, len(desired_return))
    desired_return[objective_index] += np.random.uniform(high=objective_std[objective_index])
    desired_return = np.float32(desired_return)
    return desired_return, desired_horizon


def update_model(model, opt, experience_replay, batch_size):
    batch = []
    # randomly choose episodes from experience buffer
    selected_episodes = np.random.choice(
        np.arange(len(experience_replay)),
        size=batch_size,
        replace=True,
    )
    for episode_index in selected_episodes:
        # episode is tuple (return, transitions)
        episode = experience_replay[episode_index][2]
        # choose random timestep from episode,
        # use its return and leftover timesteps as desired return and horizon
        timestep = np.random.randint(0, len(episode))
        # reward contains return until end of episode
        state = episode[timestep].observation
        action = episode[timestep].action
        reward_to_go = np.float32(episode[timestep].reward)
        horizon_to_go = np.float32(len(episode) - timestep)
        batch.append((state, action, reward_to_go, horizon_to_go))

    obs, actions, desired_return, desired_horizon = zip(*batch)
    log_prob = model(
        torch.as_tensor(np.asarray(obs)).to(device),
        torch.as_tensor(np.asarray(desired_return, dtype=np.float32)).to(device),
        torch.as_tensor(np.asarray(desired_horizon, dtype=np.float32)).unsqueeze(1).to(device),
    )

    opt.zero_grad(set_to_none=True)
    # one-hot of action for CE loss
    actions = F.one_hot(torch.as_tensor(actions, dtype=torch.long, device=device), len(log_prob[0]))
    # cross-entropy loss
    loss = torch.sum(-actions * log_prob, -1)
    loss = loss.mean()
    loss.backward()
    opt.step()

    return loss, log_prob


def eval(env, model, experience_replay, max_return, gamma=1.0, n=10):
    episodes = nlargest(n, experience_replay)
    returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
    returns = np.asarray(returns, dtype=np.float32)
    horizons = np.asarray(horizons, dtype=np.float32)
    n_eval = len(returns)
    e_returns = np.full((n, returns.shape[-1]), np.nan, dtype=np.float32)
    desired_returns = np.full((n, returns.shape[-1]), np.nan, dtype=np.float32)
    desired_returns[:n_eval] = returns
    actual_returns = []
    for episode_index in range(n_eval):
        transitions = run_episode(
            env,
            model,
            returns[episode_index],
            np.float32(horizons[episode_index] - 2),
            max_return,
        )
        # compute return
        for timestep in reversed(range(len(transitions) - 1)):
            transitions[timestep].reward += gamma * transitions[timestep + 1].reward
        actual_returns.append(transitions[0].reward)

    actual_returns = np.asarray(actual_returns, dtype=np.float32)
    e_returns[:n_eval] = actual_returns
    distances = np.linalg.norm(returns - actual_returns, axis=0)
    return e_returns, desired_returns, distances


def train(env,
          model,
          learning_rate=1e-2,
          batch_size=1024,
          total_steps=1e7,
          n_model_updates=100,
          n_step_episodes=10,
          n_er_episodes=500,
          gamma=1.0,
          max_return=250.0,
          max_size=500,
          ref_point=np.array([0, 0]),
          logdir='runs/',
          er_env=None,
          er_policy=None,
          resume_state=None):
    step = 0
    total_episodes = n_er_episodes
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger = Logger(logdir=logdir, append=resume_state is not None)
    checkpoint_dir = checkpoints_dir(logger.logdir, create=True)
    n_checkpoints = 0
    experience_replay = []
    if resume_state is not None:
        step = int(resume_state['step'])
        total_episodes = int(resume_state.get('total_episodes', n_er_episodes))
        n_checkpoints = int(resume_state.get('n_checkpoints', 0))
        experience_replay = resume_state['experience_replay']
        opt.load_state_dict(resume_state['optimizer_state_dict'])
        move_optimizer_state_to_device(opt, device)
        restore_rng_state(resume_state.get('rng_state'))
        print(
            f'resuming horizon-conditioned training at step {step} '
            f'with {len(experience_replay)} replay episodes'
        )
    else:
        # fill buffer with random (or seed-policy) episodes
        fill_env = er_env if er_env is not None else env
        for _ in range(n_er_episodes):
            transitions = []
            obs = reset_env(fill_env)
            if er_policy is not None and hasattr(er_policy, 'reset'):
                er_policy.reset()
            done = False
            while not done:
                if er_policy is not None:
                    action = er_policy(fill_env, obs)
                else:
                    action = sample_random_action(fill_env)
                n_obs, reward, done, _ = step_env(fill_env, action)
                transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, done))
                obs = n_obs
                step += 1
            # add episode in-place
            add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)

    if resume_state is not None:
        checkpoint_interval = max(int(np.ceil(max(total_steps - step, 1) / 10.0)), 1)
        next_checkpoint_step = step + checkpoint_interval
    else:
        checkpoint_interval = max(int(np.ceil(total_steps / 10.0)), 1)
        next_checkpoint_step = checkpoint_interval * max(n_checkpoints + 1, 1)

    while step < total_steps:
        loss = []
        entropy = []
        for _ in range(n_model_updates):
            l, lp = update_model(
                model,
                opt,
                experience_replay,
                batch_size=batch_size,
            )
            loss.append(l.detach().cpu().numpy())
            lp = lp.detach().cpu().numpy()
            ent = np.sum(-np.exp(lp) * lp)
            entropy.append(ent)

        desired_return, desired_horizon = choose_commands(experience_replay, n_er_episodes)

        # get all leaves, contain biggest elements, experience_replay got heapified in choose_commands
        leaves_r = np.array([e[2][0].reward for e in experience_replay[len(experience_replay) // 2:]])
        leaves_h = np.array([len(e[2]) for e in experience_replay[len(experience_replay) // 2:]])
        if len(experience_replay) == max_size:
            logger.put('train/leaves/r', leaves_r, step, f'{leaves_r.shape[-1]}d')
            logger.put('train/leaves/h', leaves_h, step, f'{leaves_h.shape[-1]}d')
        hv_est = hypervolume_score(leaves_r, ref_point)
        logger.put('train/hypervolume', hv_est, step, 'scalar')

        returns = []
        horizons = []
        for _ in range(n_step_episodes):
            transitions = run_episode(env, model, desired_return, desired_horizon, max_return)
            step += len(transitions)
            add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
            returns.append(transitions[0].reward)
            horizons.append(len(transitions))

        returns = np.asarray(returns, dtype=np.float32)
        mean_return = np.mean(returns, axis=0)
        std_return = np.std(returns, axis=0)
        total_episodes += n_step_episodes
        logger.put('train/episode', total_episodes, step, 'scalar')
        logger.put('train/loss', np.mean(loss), step, 'scalar')
        logger.put('train/entropy', np.mean(entropy), step, 'scalar')
        logger.put('train/horizon/desired', desired_horizon, step, 'scalar')
        logger.put('train/horizon/distance', np.linalg.norm(np.mean(horizons) - desired_horizon), step, 'scalar')
        for objective in range(len(desired_return)):
            logger.put(f'train/return/{objective}/value', mean_return[objective], step, 'scalar')
            logger.put(f'train/return/{objective}/desired', desired_return[objective], step, 'scalar')
            logger.put(
                f'train/return/{objective}/distance',
                np.linalg.norm(mean_return[objective] - desired_return[objective]),
                step,
                'scalar',
            )
        print(
            f'step {step} \t desired {desired_return}, horizon {desired_horizon:.2f} '
            f'\t return ({mean_return}), ({std_return}) '
            f'\t horizon {np.mean(horizons):.2f} '
            f'\t loss {np.mean(loss):.3E}'
        )
        if step >= next_checkpoint_step or step >= total_steps:
            torch.save(model, checkpoint_dir / f'model_{n_checkpoints + 1}.pt')
            n_checkpoints += 1
            next_checkpoint_step += checkpoint_interval

            e_r, e_dr, e_d = eval(env, model, experience_replay, max_return, gamma=gamma)
            logger.put('eval/return/desired', e_dr, step, f'{len(desired_return)}d')
            logger.put('eval/return/value', e_r, step, f'{len(desired_return)}d')
            logger.put('eval/expected_utility', expected_utility(e_r), step, 'scalar')
            for objective in range(len(desired_return)):
                logger.put(f'eval/return/{objective}/distance', e_d[objective], step, 'scalar')
            save_training_state(
                logger.logdir,
                step=step,
                total_episodes=total_episodes,
                n_checkpoints=n_checkpoints,
                experience_replay=experience_replay,
                optimizer=opt,
            )

    if n_checkpoints == 0:
        torch.save(model, checkpoint_dir / 'model_1.pt')
        n_checkpoints = 1

    final_replay = nlargest((len(experience_replay) + 1) // 2, experience_replay)
    save_training_state(
        logger.logdir,
        step=step,
        total_episodes=total_episodes,
        n_checkpoints=n_checkpoints,
        experience_replay=experience_replay,
        optimizer=opt,
    )
    logger.close()
    write_final_replay_commands(logger.logdir, final_replay, step)
