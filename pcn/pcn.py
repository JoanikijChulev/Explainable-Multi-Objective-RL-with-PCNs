import heapq
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from pygmo import hypervolume

from artifacts import checkpoints_dir, training_state_path
from device_utils import preferred_device
from logger import Logger
from morl_metrics import expected_utility


def crowding_distance(points):
    if len(points) == 0:
        return np.array([], dtype=np.float32)
    if len(points) <= 2:
        return np.full((len(points),), points.shape[-1], dtype=np.float32)
    points = (points - points.min(axis=0)) / (np.ptp(points, axis=0) + 1e-8)
    dim_sorted = np.argsort(points, axis=0)
    point_sorted = np.take_along_axis(points, dim_sorted, axis=0)
    distances = np.abs(point_sorted[:-2] - point_sorted[2:])
    distances = np.pad(distances, ((1,), (0,)), constant_values=1)
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

COSINE_DECAY_FRACTION = 0.8
def cosine_decay_learning_rate(initial_lr, min_lr, step, total_steps):
    if min_lr is None or min_lr >= initial_lr:
        return float(initial_lr)
    decay_steps = max(float(total_steps) * COSINE_DECAY_FRACTION, 1.0)
    progress = np.clip(step / decay_steps, 0.0, 1.0)
    return float(min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + np.cos(np.pi * progress)))


def set_optimizer_learning_rate(opt, learning_rate):
    for param_group in opt.param_groups:
        param_group['lr'] = float(learning_rate)


def get_non_dominated(solutions):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            is_efficient[i] = 1
    return is_efficient


def reward_time_tiebreak_mask(returns, lengths, atol=1e-6, rtol=1e-3):
    returns = np.asarray(returns, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)
    keep = np.ones(len(returns), dtype=bool)
    for i in range(len(returns)):
        if not keep[i]:
            continue
        for j in range(len(returns)):
            if i == j:
                continue
            shorter = lengths[j] < lengths[i]
            if not shorter:
                continue
            scale = np.maximum(np.maximum(np.abs(returns[i]), np.abs(returns[j])), 1.0)
            near_equal = np.all(np.abs(returns[j] - returns[i]) <= (atol + rtol * scale))
            if near_equal:
                keep[i] = False
                break
    return keep


def reward_array(reward):
    return np.asarray(reward, dtype=np.float32)


def compute_hypervolume(q_set, ref):
    nA = len(q_set)
    q_values = np.zeros(nA)
    for i in range(nA):
        points = np.array(q_set[i]) * -1.0
        hv = hypervolume(points)
        q_values[i] = hv.compute(ref * -1)
    return q_values


def reward_is_zero(reward, atol=1e-6):
    reward = reward_array(reward)
    return bool(np.all(np.abs(reward) <= atol))


def relabel_episode_returns(transitions, gamma=1.0):
    for i in reversed(range(len(transitions) - 1)):
        transitions[i].reward = reward_array(transitions[i].reward) + gamma * reward_array(transitions[i + 1].reward)
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


def trim_reward_neutral_suffix(transitions):
    if len(transitions) <= 1:
        return transitions
    suffix_return = np.zeros_like(np.asarray(transitions[-1].reward, dtype=np.float32), dtype=np.float32)
    trim_start = len(transitions)
    for index in reversed(range(len(transitions))):
        suffix_return = suffix_return + np.asarray(transitions[index].reward, dtype=np.float32)
        if reward_is_zero(suffix_return):
            trim_start = index
    if 0 < trim_start < len(transitions):
        return transitions[:trim_start]
    return transitions


def nlargest(n, experience_replay, threshold=.2):
    returns = np.array([e[2][0].reward for e in experience_replay])
    distances = crowding_distance(returns)
    sma = np.argwhere(distances <= threshold).flatten()

    nd_i = get_non_dominated(returns)
    nd = returns[nd_i]
    l2 = np.min(np.linalg.norm(returns[:, None, :] - nd[None, :, :], axis=-1), axis=-1) * -1
    nd_i = np.nonzero(nd_i)[0]
    _, unique_i = np.unique(nd, axis=0, return_index=True)
    unique_i = nd_i[unique_i]
    duplicates = np.ones(len(l2), dtype=bool)
    duplicates[unique_i] = False
    l2[duplicates] -= 1e-5
    l2[sma] *= 2

    sorted_i = np.argsort(l2)
    selected_indexes = sorted_i[-n:]
    largest = [experience_replay[i] for i in selected_indexes]
    for i in range(len(l2)):
        experience_replay[i] = (l2[i], experience_replay[i][1], experience_replay[i][2])
    heapq.heapify(experience_replay)
    return largest


def add_episode(transitions, experience_replay, gamma=1.0, max_size=100, step=0):
    transitions = trim_reward_neutral_suffix(transitions)
    relabel_episode_returns(transitions, gamma=gamma)
    if len(experience_replay) == max_size:
        heapq.heappushpop(experience_replay, (1, step, transitions))
    else:
        heapq.heappush(experience_replay, (1, step, transitions))


def choose_action(model, env, obs, desired_return):
    obs = np.asarray([obs])
    desired_return = np.asarray([desired_return], dtype=np.float32)
    log_probs = model(
        torch.as_tensor(obs).to(device),
        torch.as_tensor(desired_return).to(device),
    )
    log_probs = log_probs.detach().cpu().numpy()[0]
    action_mask = action_mask_for_env(env)
    probs, _ = action_probabilities(log_probs, action_mask)
    action = np.random.choice(np.arange(len(log_probs)), p=probs)
    return int(action)


def run_episode(env, model, desired_return, max_return):
    transitions = []
    obs = reset_env(env)
    done = False
    remaining_return = np.asarray(desired_return, dtype=np.float32).copy()
    while not done:
        action = choose_action(model, env, obs, remaining_return)
        n_obs, reward, done, _ = step_env(env, action)

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


def command_sampling_pool(
    experience_replay,
    n_episodes,
    time_tiebreak_commands=True,
):
    episodes = nlargest(n_episodes, experience_replay)
    candidate_returns = np.asarray([e[2][0].reward for e in episodes], dtype=np.float32)
    lengths = np.asarray([len(e[2]) for e in episodes], dtype=np.float32)
    mask = get_non_dominated(candidate_returns)
    base_returns = candidate_returns[mask]
    lengths = lengths[mask]
    if time_tiebreak_commands:
        mask = reward_time_tiebreak_mask(base_returns, lengths)
        base_returns = base_returns[mask]
    return candidate_returns, base_returns


def choose_commands(
    experience_replay,
    n_episodes,
    time_tiebreak_commands=True,
):
    candidate_returns, base_returns = command_sampling_pool(
        experience_replay,
        n_episodes,
        time_tiebreak_commands=time_tiebreak_commands,
    )
    r_i = np.random.randint(0, len(base_returns))
    desired_return = base_returns[r_i].copy()
    objective_std = np.std(candidate_returns, axis=0)

    objective_index = np.random.randint(0, len(desired_return))
    delta = np.random.uniform(high=objective_std[objective_index])
    desired_return[objective_index] += delta
    return np.float32(desired_return)


def update_model(
    model,
    opt,
    experience_replay,
    batch_size,
):
    batch = []
    for _ in range(batch_size):
        episode = experience_replay[np.random.randint(0, len(experience_replay))][2]
        timestep = np.random.randint(0, len(episode))
        state = episode[timestep].observation
        action = episode[timestep].action
        reward_to_go = np.float32(episode[timestep].reward)
        batch.append((state, action, reward_to_go))

    obs, actions, desired_return = zip(*batch)
    obs = np.asarray(obs)
    desired_return = np.asarray(desired_return, dtype=np.float32)
    log_prob = model(
        torch.as_tensor(obs).to(device),
        torch.as_tensor(desired_return).to(device),
    )

    opt.zero_grad(set_to_none=True)
    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    loss = F.nll_loss(log_prob, actions)
    loss.backward()
    opt.step()
    return loss, log_prob


def eval(env, model, experience_replay, max_return, gamma=1.0, n=10):
    episodes = nlargest(n, experience_replay)
    returns = np.asarray([e[2][0].reward for e in episodes], dtype=np.float32)
    n_eval = len(returns)
    e_returns = np.full((n, returns.shape[-1]), np.nan, dtype=np.float32)
    desired_returns = np.full((n, returns.shape[-1]), np.nan, dtype=np.float32)
    desired_returns[:n_eval] = returns
    actual_returns = []
    for i in range(n_eval):
        transitions = run_episode(env, model, returns[i], max_return)
        for j in reversed(range(len(transitions) - 1)):
            transitions[j].reward += gamma * transitions[j + 1].reward
        actual_returns.append(transitions[0].reward)

    actual_returns = np.asarray(actual_returns, dtype=np.float32)
    e_returns[:n_eval] = actual_returns
    distances = np.linalg.norm(returns - actual_returns, axis=0)
    return e_returns, desired_returns, distances


def train(env,
          model,
          learning_rate=1e-2,
          min_learning_rate=None,
          batch_size=1024,
          total_steps=1e7,
          n_model_updates=100,
          n_step_episodes=10,
          n_er_episodes=500,
          gamma=1.0,
          max_return=250.0,
          max_size=500,
          time_tiebreak_commands=True,
          ref_point=np.array([0, 0]),
          logdir='runs/',
          resume_state=None):
    step = 0
    total_episodes = n_er_episodes
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
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
            f'resuming training at step {step} with {len(experience_replay)} replay episodes'
        )
    else:
        for _ in range(n_er_episodes):
            transitions = []
            obs = reset_env(env)
            done = False
            while not done:
                action = sample_random_action(env)
                n_obs, reward, done, _ = step_env(env, action)
                transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, done))
                obs = n_obs
                step += 1
            add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)

    if resume_state is not None:
        checkpoint_interval = max(int(np.ceil(max(total_steps - step, 1) / 10.0)), 1)
        next_checkpoint_step = step + checkpoint_interval
    else:
        checkpoint_interval = max(int(np.ceil(total_steps / 10.0)), 1)
        next_checkpoint_step = checkpoint_interval * max(n_checkpoints + 1, 1)

    while step < total_steps:
        current_learning_rate = cosine_decay_learning_rate(
            learning_rate,
            min_learning_rate,
            step,
            total_steps,
        )
        set_optimizer_learning_rate(opt, current_learning_rate)
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

        desired_return = choose_commands(
            experience_replay,
            n_er_episodes,
            time_tiebreak_commands=time_tiebreak_commands,
        )

        leaves_r = np.asarray([e[2][0].reward for e in experience_replay[len(experience_replay) // 2:]])
        leaves_h = np.asarray([len(e[2]) for e in experience_replay[len(experience_replay) // 2:]])
        try:
            if len(experience_replay) == max_size:
                logger.put('train/leaves/r', leaves_r, step, f'{leaves_r.shape[-1]}d')
                logger.put('train/leaves/h', leaves_h, step, f'{leaves_h.shape[-1]}d')
            hv = hypervolume(leaves_r * -1)
            hv_est = hv.compute(ref_point * -1)
            logger.put('train/hypervolume', hv_est, step, 'scalar')
        except ValueError:
            pass

        returns = []
        lengths = []
        for _ in range(n_step_episodes):
            transitions = run_episode(env, model, desired_return, max_return)
            step += len(transitions)
            add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
            returns.append(transitions[0].reward)
            lengths.append(len(transitions))

        returns = np.asarray(returns, dtype=np.float32)
        mean_return = np.mean(returns, axis=0)
        std_return = np.std(returns, axis=0)
        total_episodes += n_step_episodes
        logger.put('train/episode_length', np.mean(lengths), step, 'scalar')
        logger.put('train/episode', total_episodes, step, 'scalar')
        logger.put('train/loss', np.mean(loss), step, 'scalar')
        logger.put('train/entropy', np.mean(entropy), step, 'scalar')
        logger.put('train/learning_rate', current_learning_rate, step, 'scalar')
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
            f'step {step} \t desired {desired_return} '
            f'\t return ({mean_return}), ({std_return}) '
            f'\t length {np.mean(lengths):.2f} '
            f'\t lr {current_learning_rate:.2E} \t loss {np.mean(loss):.3E}'
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

    save_training_state(
        logger.logdir,
        step=step,
        total_episodes=total_episodes,
        n_checkpoints=n_checkpoints,
        experience_replay=experience_replay,
        optimizer=opt,
    )
    logger.close()
