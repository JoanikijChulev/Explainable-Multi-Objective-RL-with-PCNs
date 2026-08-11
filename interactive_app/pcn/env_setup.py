from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import TimeLimit


class ActionMaskProxy(gym.Wrapper):

    def get_action_mask(self):
        mask_fn = getattr(self.env, 'get_action_mask', None)
        if not callable(mask_fn):
            get_wrapper_attr = getattr(self.env, 'get_wrapper_attr', None)
            if callable(get_wrapper_attr):
                try:
                    mask_fn = get_wrapper_attr('get_action_mask')
                except AttributeError:
                    mask_fn = None
        if callable(mask_fn):
            return np.asarray(mask_fn(), dtype=bool)
        return np.ones(int(self.action_space.n), dtype=bool)


class MultiOneHotEnv(gym.ObservationWrapper):

    def __init__(self, env):
        super(MultiOneHotEnv, self).__init__(env)
        base_env = env.unwrapped
        self._size = base_env.size
        self._dimensions = base_env.dimensions
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._size * self._dimensions,),
            dtype=np.float32,
        )

    def observation(self, observation):
        multi_one_hot = np.zeros(self._size * self._dimensions, dtype=np.float32)
        for index, value in enumerate(observation):
            multi_one_hot[index * self._size + value] = 1.0
        return multi_one_hot

    def get_action_mask(self):
        base_env = self.unwrapped
        position = np.asarray(base_env.pos, dtype=np.int32)
        mask = np.zeros(int(self.action_space.n), dtype=bool)
        for index, delta in enumerate(base_env.actions):
            candidate = position + np.asarray(delta, dtype=np.int32)
            mask[index] = bool(np.all((0 <= candidate) & (candidate < self._size)))
        return mask


class DeepSeaTreasureIndexObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super(DeepSeaTreasureIndexObservation, self).__init__(env)
        base_env = env.unwrapped
        self._shape = tuple(base_env.sea_map.shape)
        self.observation_space = gym.spaces.Discrete(int(np.prod(self._shape)))

    def observation(self, observation):
        row, col = np.asarray(observation, dtype=np.int32)
        return int(np.ravel_multi_index((row, col), self._shape))

    def get_action_mask(self):
        mask_fn = getattr(self.env, 'get_action_mask', None)
        if callable(mask_fn):
            return np.asarray(mask_fn(), dtype=bool)
        return np.ones(int(self.action_space.n), dtype=bool)


class DeepSeaTreasureLegacyActionOrder(gym.ActionWrapper):

    # Preserve the historical action semantics used by the local bundled env:
    # 0=up, 1=right, 2=down, 3=left. MO-Gymnasium uses 0=up, 1=down, 2=left, 3=right.
    ACTION_MAPPING = np.asarray([0, 3, 1, 2], dtype=np.int64)

    def action(self, action):
        action = int(action)
        return int(self.ACTION_MAPPING[action])

    def get_action_mask(self):
        base_env = self.unwrapped
        mask = np.zeros(int(self.action_space.n), dtype=bool)
        for wrapped_action, base_action in enumerate(self.ACTION_MAPPING):
            next_state = base_env.current_state + base_env.dir[int(base_action)]
            mask[wrapped_action] = bool(base_env._is_valid_state(next_state))
        return mask


class FourRoomObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super(FourRoomObservation, self).__init__(env)
        base_env = env.unwrapped
        self._height = base_env.height
        self._width = base_env.width
        self._n_shapes = len(base_env.shape_ids)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._height + self._width + self._n_shapes,),
            dtype=np.float32,
        )

    def observation(self, observation):
        row, col = np.asarray(observation[:2], dtype=np.int32)
        encoded = np.zeros(self._height + self._width + self._n_shapes, dtype=np.float32)
        encoded[row] = 1.0
        encoded[self._height + col] = 1.0
        encoded[self._height + self._width:] = np.asarray(observation[2:], dtype=np.float32)
        return encoded

    def get_action_mask(self):
        base_env = self.unwrapped
        (row, col), _ = base_env.state
        candidates = (
            (row, col - 1),
            (row - 1, col),
            (row, col + 1),
            (row + 1, col),
        )
        mask = np.zeros(int(self.action_space.n), dtype=bool)
        for index, (candidate_row, candidate_col) in enumerate(candidates):
            valid = (
                0 <= candidate_row < self._height
                and 0 <= candidate_col < self._width
                and (candidate_row, candidate_col) not in base_env.occupied
            )
            mask[index] = valid
        return mask


class ResourceGatheringIndexObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super(ResourceGatheringIndexObservation, self).__init__(env)
        base_env = env.unwrapped
        self._size = base_env.size
        self._resource_flag_states = 4
        self.observation_space = gym.spaces.Discrete(self._size * self._size * self._resource_flag_states)

    def observation(self, observation):
        row, col, has_gold, has_gem = np.asarray(observation, dtype=np.int32)
        position_index = row * self._size + col
        inventory_index = (has_gold << 1) + has_gem
        return int(position_index * self._resource_flag_states + inventory_index)

    def get_action_mask(self):
        base_env = self.unwrapped
        mask = np.zeros(int(self.action_space.n), dtype=bool)
        for action, delta in base_env.dir.items():
            candidate = base_env.current_pos + delta
            mask[int(action)] = bool(base_env.is_valid_state(candidate))
        return mask


class BreakableBottlesIndexObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super(BreakableBottlesIndexObservation, self).__init__(env)
        base_env = env.unwrapped
        self.observation_space = gym.spaces.Discrete(base_env.num_observations)

    def observation(self, observation):
        state_index = np.asarray(self.unwrapped.get_obs_idx(observation), dtype=np.int64)
        return int(state_index.reshape(-1)[0])

    def get_action_mask(self):
        base_env = self.unwrapped
        mask = np.zeros(int(self.action_space.n), dtype=bool)
        mask[base_env.LEFT] = bool(base_env.location > 0)
        mask[base_env.RIGHT] = bool(base_env.location < base_env.size - 1)
        carrying_capacity = base_env.bottles_carrying < 2
        pickup_from_source = base_env.location == 0 and carrying_capacity
        pickup_from_ground = (
            base_env.unbreakable_bottles
            and carrying_capacity
            and base_env.location in range(1, base_env.size - 1)
            and base_env.bottles_dropped[base_env.location - 1] > 0
        )
        mask[base_env.PICKUP] = bool(pickup_from_source or pickup_from_ground)
        return mask


class MountainCarSeedPolicy:
    """Energy-pumping seeding policy for MountainCar replay initialization.

    Accelerating in the direction of the current velocity reliably reaches the
    goal, which a uniform random policy essentially never does within the time
    limit. Each episode randomizes how often the pump is replaced by no-ops or
    random actions so the seeded returns spread along the time/effort
    trade-off instead of collapsing onto a single demonstration.
    """

    def __init__(self, random_episode_prob=0.5, max_noop_prob=0.6, max_epsilon=0.3):
        self.random_episode_prob = random_episode_prob
        self.max_noop_prob = max_noop_prob
        self.max_epsilon = max_epsilon
        self.reset()

    def reset(self):
        self._random_episode = np.random.rand() < self.random_episode_prob
        self._noop_prob = np.random.uniform(0.0, self.max_noop_prob)
        self._epsilon = np.random.uniform(0.0, self.max_epsilon)

    def __call__(self, env, obs):
        n_actions = int(env.action_space.n)
        if self._random_episode or np.random.rand() < self._epsilon:
            return int(np.random.randint(0, n_actions))
        if np.random.rand() < self._noop_prob:
            return 1
        velocity = float(np.asarray(obs, dtype=np.float32)[1])
        return 0 if velocity < 0 else 2


class LunarLanderSeedPolicy:
    """Heuristic seeding policy for (MO) LunarLander replay initialization.

    Uses Gymnasium's built-in PD landing controller, which reliably lands the
    craft -- something a uniform random policy almost never does. Without it the
    seed buffer contains only crashes, so the fuel objectives are the only axis
    PCN can exploit and it collapses onto firing no engines and crashing (the
    same failure mode as random-seeded MountainCar). Each episode randomizes how
    often the fuel-dominant main engine is suppressed (trading landing quality
    for lower fuel) plus random/no-op dilution, so the seeded returns spread
    along the landing-vs-fuel trade-off instead of collapsing onto one
    demonstration.
    """

    def __init__(self, random_episode_prob=0.2, max_main_suppress_prob=0.15, max_epsilon=0.1):
        from gymnasium.envs.box2d.lunar_lander import heuristic

        self._heuristic = heuristic
        self.random_episode_prob = random_episode_prob
        self.max_main_suppress_prob = max_main_suppress_prob
        self.max_epsilon = max_epsilon
        self.reset()

    def reset(self):
        self._random_episode = np.random.rand() < self.random_episode_prob
        self._main_suppress_prob = np.random.uniform(0.0, self.max_main_suppress_prob)
        self._epsilon = np.random.uniform(0.0, self.max_epsilon)

    def __call__(self, env, obs):
        n_actions = int(env.action_space.n)
        if self._random_episode or np.random.rand() < self._epsilon:
            return int(np.random.randint(0, n_actions))
        action = int(self._heuristic(env, np.asarray(obs, dtype=np.float32)))
        # Action 2 is the main engine, which dominates fuel cost; skipping it
        # spreads the fuel objective across the seeded front.
        if action == 2 and np.random.rand() < self._main_suppress_prob:
            return 0
        return action


class MinecartActionMask(gym.Wrapper):

    def get_action_mask(self):
        from mo_gymnasium.envs.minecart.minecart import ACT_ACCEL, ACT_BRAKE, ACT_LEFT, ACT_MINE, ACT_NONE, ACT_RIGHT
        from mo_gymnasium.envs.minecart.minecart import EPS_SPEED, MAX_SPEED

        base_env = self.unwrapped
        mask = np.ones(int(self.action_space.n), dtype=bool)
        mask[ACT_ACCEL] = bool(base_env.cart.speed < MAX_SPEED - EPS_SPEED)
        mask[ACT_BRAKE] = bool(base_env.cart.speed > EPS_SPEED)

        cart_free = base_env.capacity - np.sum(base_env.cart.content)
        mine_action_valid = False
        if base_env.cart.speed < EPS_SPEED and cart_free > 1e-8:
            closest_mine = min(base_env.mines, key=lambda mine: mine.distance(base_env.cart))
            mine_action_valid = bool(closest_mine.mineable(base_env.cart))
        mask[ACT_MINE] = mine_action_valid
        mask[ACT_LEFT] = True
        mask[ACT_RIGHT] = True
        mask[ACT_NONE] = True
        return mask


def scaled_command_tensor(desired_return, scaling_factor):
    desired_return = desired_return.float()
    return desired_return * scaling_factor[:, :desired_return.shape[-1]]


class DiscreteCommandModel(nn.Module):

    def __init__(self, n_states, n_actions, n_objectives, scaling_factor, n_hidden=128):
        super(DiscreteCommandModel, self).__init__()
        self.n_states = n_states
        self.scaling_factor = scaling_factor
        self.s_emb = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid(),
        )
        self.c_emb = nn.Sequential(
            nn.Linear(n_objectives, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
            nn.LogSoftmax(1),
        )

    def encode_state(self, state):
        return F.one_hot(state.long(), num_classes=self.n_states).to(state.device).float()

    def forward(self, state, desired_return):
        commands = scaled_command_tensor(desired_return, self.scaling_factor)
        state_embedding = self.s_emb(self.encode_state(state))
        command_embedding = self.c_emb(commands)
        return self.fc(state_embedding * command_embedding)


class WalkroomModel(nn.Module):

    def __init__(self, n_states, n_actions, n_objectives, scaling_factor, n_hidden=256):
        super(WalkroomModel, self).__init__()
        self.scaling_factor = scaling_factor
        self.s_emb = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid(),
        )
        self.c_emb = nn.Sequential(
            nn.Linear(n_objectives, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
            nn.LogSoftmax(1),
        )

    def forward(self, state, desired_return):
        commands = scaled_command_tensor(desired_return, self.scaling_factor)
        state_embedding = self.s_emb(state.float())
        command_embedding = self.c_emb(commands)
        return self.fc(state_embedding * command_embedding)


class FourRoomModel(nn.Module):

    def __init__(
        self,
        height,
        width,
        n_items,
        n_actions,
        n_objectives,
        scaling_factor,
        position_dim=64,
        inventory_dim=64,
        n_hidden=256,
    ):
        super(FourRoomModel, self).__init__()
        self.height = height
        self.width = width
        self.n_items = n_items
        self.n_objectives = n_objectives
        self.scaling_factor = scaling_factor
        self.position_embedding = nn.Embedding(height * width, position_dim)
        self.inventory_encoder = nn.Sequential(
            nn.Linear(n_items, inventory_dim),
            nn.ReLU(),
            nn.Linear(inventory_dim, inventory_dim),
            nn.ReLU(),
        )
        self.s_emb = nn.Sequential(
            nn.Linear(position_dim + inventory_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid(),
        )
        self.c_emb = nn.Sequential(
            nn.Linear(n_objectives, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
            nn.LogSoftmax(1),
        )

    def decode_state(self, state):
        row = state[:, :self.height].argmax(dim=-1)
        col = state[:, self.height:self.height + self.width].argmax(dim=-1)
        cell_index = row * self.width + col
        inventory = state[:, self.height + self.width:].float()
        return cell_index, inventory

    def forward(self, state, desired_return):
        state = state.float()
        return_features = desired_return.float() * self.scaling_factor[:, :self.n_objectives]

        cell_index, inventory = self.decode_state(state)
        position_embedding = self.position_embedding(cell_index)
        inventory_embedding = self.inventory_encoder(inventory)
        state_embedding = self.s_emb(torch.cat((position_embedding, inventory_embedding), dim=-1))

        command_embedding = self.c_emb(return_features)
        return self.fc(state_embedding * command_embedding)


class MinecartModel(nn.Module):

    def __init__(self, n_actions, scaling_factor, state_dim=7, n_hidden=256):
        super(MinecartModel, self).__init__()
        self.scaling_factor = scaling_factor
        self.n_objectives = scaling_factor.shape[-1]
        self.s_emb = nn.Sequential(
            nn.Linear(state_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid(),
        )
        self.c_emb = nn.Sequential(
            nn.Linear(self.n_objectives, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
            nn.LogSoftmax(1),
        )

    def forward(self, state, desired_return):
        commands = scaled_command_tensor(desired_return, self.scaling_factor)
        state_embedding = self.s_emb(state.float())
        command_embedding = self.c_emb(commands)
        return self.fc(state_embedding * command_embedding)


class FruitTreeModel(DiscreteCommandModel):

    def __init__(self, depth, n_actions, n_objectives, scaling_factor, n_hidden=256):
        self.depth = depth
        n_states = 2 ** (depth + 1) - 1
        super(FruitTreeModel, self).__init__(
            n_states,
            n_actions,
            n_objectives,
            scaling_factor,
            n_hidden=n_hidden,
        )

    def encode_state(self, state):
        state = state.long()
        rows = state[:, 0]
        positions = state[:, 1]
        # Map binary-tree coordinates to the corresponding binary-heap node index.
        state_index = torch.bitwise_left_shift(torch.ones_like(rows), rows) - 1 + positions
        return super().encode_state(state_index)


@dataclass
class ExperimentSetup:
    env_name: str
    env: gym.Env
    n_actions: int
    max_return: np.ndarray
    ref_point: np.ndarray
    model: Optional[nn.Module] = None
    learning_rate: Optional[float] = None
    min_learning_rate: Optional[float] = None
    total_steps: Optional[int] = None
    batch_size: Optional[int] = None
    n_model_updates: Optional[int] = None
    n_er_episodes: Optional[int] = None
    max_size: Optional[int] = None
    er_env: Optional[gym.Env] = None
    er_policy: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def canonical_env_name(env_name):
    if env_name in {'dst', 'deepsea-treasure', 'deep-sea-treasure', 'deepsea-treasure', 'deepsaetreasure', 'deepseatreasure'}:
        return 'dst'
    if env_name in {'collect_two', 'collect-two', 'c2'}:
        return 'collect_two'
    if env_name in {'branch-path', 'branch_path', 'branch-path-v0', 'branch_path_v0', 'bp'}:
        return 'branch-path'
    if env_name in {'reward-line', 'reward_line', 'reward-line-v0', 'reward_line_v0', 'rl'}:
        return 'reward-line'
    if env_name in {'three-tree', 'three_tree', 'three-tree-v0', 'three_tree_v0', 'tt'}:
        return 'three-tree'
    if env_name in {'minecart', 'mc', 'mine-cart', 'mine_cart', 'minecart-v0', 'mine_cart_v0'}:
        return 'minecart'
    if env_name in {
        'mo-mountaincar-timespeed-v0',
        'mo-mountaincar-timespeed',
        'mountaincar-timespeed',
        'timespeed',
        'mo-mountaincar',
        'mountaincar',
        'mountain-car',
        'mountain_car',
        'momc',
    }:
        return 'mo-mountaincar-timespeed-v0'
    if env_name in {'mo-lunar-lander-v2', 'mo-lunar-lander-v3', 'mo-lunar-lander', 'lunar-lander', 'lunar_lander', 'moll'}:
        return 'mo-lunar-lander-v3'
    if env_name in {'mo-reacher-v5', 'mo-reacher', 'reacher', 'reacher-v5', 'mor'}:
        return 'mo-reacher-v5'
    if env_name in {'fruit-tree', 'fruit_tree', 'fruit-tree-v0', 'fruit_tree_v0', 'ft'}:
        return 'fruit-tree-v0'
    if env_name in {'four-room', 'four_room', 'four-room-v0', 'four_room_v0', '4room'}:
        return 'four-room-v0'
    if env_name in {'resource-gathering', 'resource_gathering', 'resource-gathering-v0', 'resource_gathering_v0', 'rsg'}:
        return 'resource-gathering-v0'
    if env_name in {'breakable-bottles', 'breakable_bottles', 'breakable-bottles-v0', 'breakable_bottles_v0', 'bb'}:
        return 'breakable-bottles-v0'
    return env_name


def build_experiment_setup(
    env_name,
    device='cpu',
    fruit_tree_depth=6,
    include_model=True,
    render_mode=None,
):
    env_name = canonical_env_name(env_name)

    if env_name == 'dst':
        try:
            from mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure import CONCAVE_MAP, DeepSeaTreasure
        except ImportError as exc:
            raise ImportError('dst requires the mo-gymnasium package to be installed') from exc

        env = DeepSeaTreasure(render_mode=render_mode, dst_map=CONCAVE_MAP)
        env = DeepSeaTreasureLegacyActionOrder(env)
        env = DeepSeaTreasureIndexObservation(env)
        env = TimeLimit(env, 200)
        env = ActionMaskProxy(env)
        n_actions = 4
        max_treasure = float(np.max(env.unwrapped.sea_map))
        max_return = np.array([max_treasure, -1.0], dtype=np.float32)
        ref_point = np.array([0.0, -200.0], dtype=np.float32)
        model = None
        if include_model:
            scaling_factor = torch.tensor([[0.1, 0.1]], dtype=torch.float32, device=device)
            model = DiscreteCommandModel(
                n_states=env.observation_space.n,
                n_actions=n_actions,
                n_objectives=max_return.shape[0],
                scaling_factor=scaling_factor,
                n_hidden=256,
            ).to(device)
        setup = ExperimentSetup(
            env_name=env_name,
            env=env,
            n_actions=n_actions,
            model=model,
            learning_rate=1e-2,
            min_learning_rate=1e-3,
            total_steps=int(2e5),
            batch_size=512,
            n_model_updates=40,
            n_er_episodes=10000,
            max_size=10000,
            max_return=max_return,
            ref_point=ref_point,
        )

    elif env_name == 'collect_two':
        import envs  # noqa: F401

        env = gym.make('collect_two-v0', render_mode=render_mode)
        env = ActionMaskProxy(env)
        n_actions = int(env.action_space.n)
        reward_dim = int(env.unwrapped.reward_dim)
        max_return = np.ones(reward_dim, dtype=np.float32)
        ref_point = np.zeros(reward_dim, dtype=np.float32)
        model = None
        if include_model:
            scaling_factor = torch.tensor([[1.0] * reward_dim], dtype=torch.float32, device=device)
            model = WalkroomModel(
                n_states=int(np.prod(env.observation_space.shape)),
                n_actions=n_actions,
                n_objectives=reward_dim,
                scaling_factor=scaling_factor,
                n_hidden=512,
            ).to(device)
        setup = ExperimentSetup(
            env_name=env_name,
            env=env,
            n_actions=n_actions,
            model=model,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            total_steps=int(1e6),
            batch_size=512,
            n_model_updates=50,
            n_er_episodes=20000,
            max_size=20000,
            max_return=max_return,
            ref_point=ref_point,
        )

    elif env_name == 'branch-path':
        import envs  # noqa: F401

        env = gym.make('branch-path-v0', render_mode=render_mode)
        env = ActionMaskProxy(env)
        n_actions = int(env.action_space.n)
        reward_dim = int(env.unwrapped.reward_dim)
        max_return = np.array([3.0, 3.0], dtype=np.float32)
        ref_point = np.zeros(reward_dim, dtype=np.float32)
        model = None
        if include_model:
            scaling_factor = torch.tensor([[1.0] * reward_dim], dtype=torch.float32, device=device)
            model = WalkroomModel(
                n_states=int(np.prod(env.observation_space.shape)),
                n_actions=n_actions,
                n_objectives=reward_dim,
                scaling_factor=scaling_factor,
                n_hidden=512,
            ).to(device)
        setup = ExperimentSetup(
            env_name=env_name,
            env=env,
            n_actions=n_actions,
            model=model,
            learning_rate=1e-4,
            min_learning_rate=1e-5,
            total_steps=int(1e6),
            batch_size=256,
            n_model_updates=20,
            n_er_episodes=20000,
            max_size=20000,
            max_return=max_return,
            ref_point=ref_point,
        )

    elif env_name == 'reward-line':
        import envs  # noqa: F401

        env = gym.make('reward-line-v0', render_mode=render_mode)
        env = ActionMaskProxy(env)
        n_actions = int(env.action_space.n)
        reward_dim = int(env.unwrapped.reward_dim)
        max_return = np.ones(reward_dim, dtype=np.float32)
        ref_point = np.zeros(reward_dim, dtype=np.float32)
        model = None
        if include_model:
            scaling_factor = torch.tensor([[1.0] * reward_dim], dtype=torch.float32, device=device)
            model = WalkroomModel(
                n_states=int(np.prod(env.observation_space.shape)),
                n_actions=n_actions,
                n_objectives=reward_dim,
                scaling_factor=scaling_factor,
                n_hidden=256,
            ).to(device)
        setup = ExperimentSetup(
            env_name=env_name,
            env=env,
            n_actions=n_actions,
            model=model,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            total_steps=int(4e5),
            batch_size=256,
            n_model_updates=20,
            n_er_episodes=10000,
            max_size=10000,
            max_return=max_return,
            ref_point=ref_point,
        )

    elif env_name == 'three-tree':
        import envs  # noqa: F401

        env = gym.make('three-tree-v0', render_mode=render_mode)
        env = ActionMaskProxy(env)
        n_actions = int(env.action_space.n)
        base_env = env.unwrapped
        reward_dim = int(base_env.reward_dim)
        all_returns = np.asarray(
            [base_env.return_for_path(path) for path in base_env.leaf_paths()],
            dtype=np.float32,
        )
        max_return = np.max(all_returns, axis=0).astype(np.float32)
        ref_point = np.zeros(reward_dim, dtype=np.float32)
        model = None
        if include_model:
            scaling_factor = torch.tensor([[1.0] * reward_dim], dtype=torch.float32, device=device)
            model = WalkroomModel(
                n_states=int(np.prod(env.observation_space.shape)),
                n_actions=n_actions,
                n_objectives=reward_dim,
                scaling_factor=scaling_factor,
                n_hidden=512,
            ).to(device)
        setup = ExperimentSetup(
            env_name=env_name,
            env=env,
            n_actions=n_actions,
            model=model,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            total_steps=int(5e5),
            batch_size=256,
            n_model_updates=20,
            n_er_episodes=10000,
            max_size=10000,
            max_return=max_return,
            ref_point=ref_point,
        )

    elif env_name.startswith('walkroom'):
        import envs  # noqa: F401

        n_objectives = int(env_name[len('walkroom'):])
        base_env = gym.make(f'Walkroom{n_objectives}D-v0', render_mode=render_mode)
        walkroom_size = base_env.unwrapped.size
        env = MultiOneHotEnv(base_env)
        env = TimeLimit(env, 200)
        env = ActionMaskProxy(env)
        n_actions = n_objectives * 2
        max_return = np.zeros(n_objectives, dtype=np.float32)
        ref_point = np.full(n_objectives, -walkroom_size, dtype=np.float32)
        model = None
        if include_model:
            scaling_factor = torch.tensor([[0.1] * n_objectives], dtype=torch.float32, device=device)
            model = WalkroomModel(
                n_states=walkroom_size * n_objectives,
                n_actions=n_actions,
                n_objectives=n_objectives,
                scaling_factor=scaling_factor,
                n_hidden=256,
            ).to(device)
        setup = ExperimentSetup(
            env_name=env_name,
            env=env,
            n_actions=n_actions,
            model=model,
            learning_rate=3e-3,
            min_learning_rate=3e-4,
            total_steps=int(3 * 100 * n_objectives * (300 + 100 * n_objectives)),
            batch_size=256,
            n_model_updates=30,
            n_er_episodes=1000,
            max_size=1000,
            max_return=max_return,
            ref_point=ref_point,
        )

    elif env_name == 'minecart':
        try:
            import mo_gymnasium as mo_gym
        except ImportError as exc:
            raise ImportError('minecart requires the mo-gymnasium package to be installed') from exc

        env = mo_gym.make('minecart-deterministic-v0', render_mode=render_mode)
        env = MinecartActionMask(env)
        env = TimeLimit(env, 1000)
        env = ActionMaskProxy(env)
        n_actions = 6
        state_dim = int(np.prod(env.observation_space.shape))
        max_return = np.array([1.5, 1.5, -0.0], dtype=np.float32)
        ref_point = np.array([0.0, 0.0, -200.0], dtype=np.float32)
        model = None
        if include_model:
            scaling_factor = torch.tensor([[1.0, 1.0, 0.1]], dtype=torch.float32, device=device)
            model = MinecartModel(
                n_actions=n_actions,
                scaling_factor=scaling_factor,
                state_dim=state_dim,
                n_hidden=512,
            ).to(device)
        setup = ExperimentSetup(
            env_name=env_name,
            env=env,
            n_actions=n_actions,
            model=model,
            learning_rate=1e-3,
            min_learning_rate=1e-5,
            total_steps=int(2.5e7),
            batch_size=256,
            n_model_updates=30,
            n_er_episodes=25000,
            max_size=25000,
            max_return=max_return,
            ref_point=ref_point,
        )

    elif env_name == 'mo-mountaincar-timespeed-v0':
        try:
            import mo_gymnasium as mo_gym
        except ImportError as exc:
            raise ImportError(f'{env_name} requires the mo-gymnasium package to be installed') from exc

        env = mo_gym.make(env_name, render_mode=render_mode)
        env = ActionMaskProxy(env)
        n_actions = int(env.action_space.n)
        state_dim = int(np.prod(env.observation_space.shape))
        reward_dim = int(env.unwrapped.reward_dim)
        max_episode_steps = int(getattr(getattr(env, 'spec', None), 'max_episode_steps', 200))
        max_step_speed_reward = 15.0 * float(env.unwrapped.max_speed)
        max_return = np.array(
            [0.0, max_step_speed_reward * max_episode_steps],
            dtype=np.float32,
        )
        ref_point = np.array([-float(max_episode_steps), 0.0], dtype=np.float32)
        seed_time_limit = 1000
        model = None
        if include_model:
            scaling_factor = torch.tensor([[0.01] * reward_dim], dtype=torch.float32, device=device)
            model = WalkroomModel(
                n_states=state_dim,
                n_actions=n_actions,
                n_objectives=reward_dim,
                scaling_factor=scaling_factor,
                n_hidden=256,
            ).to(device)
        # Replay seeding uses a longer time limit so non-demo exploration can
        # occasionally reach the goal; training rollouts keep the 200-step cap.
        er_env = ActionMaskProxy(
            mo_gym.make(env_name, max_episode_steps=seed_time_limit)
        )
        er_policy = MountainCarSeedPolicy()
        setup = ExperimentSetup(
            env_name=env_name,
            env=env,
            n_actions=n_actions,
            model=model,
            learning_rate=1e-3,
            min_learning_rate=1e-3,
            total_steps=int(3e7),
            batch_size=256,
            n_model_updates=40,
            n_er_episodes=5000,
            max_size=5000,
            max_return=max_return,
            ref_point=ref_point,
            er_env=er_env,
            er_policy=er_policy,
            metadata={
                'library_env_id': env_name,
                'er_seed_policy': 'mountaincar_energy_pumping_mix',
                'er_seed_time_limit': seed_time_limit,
            },
        )

    elif env_name == 'mo-lunar-lander-v3':
        try:
            import mo_gymnasium as mo_gym
        except ImportError as exc:
            raise ImportError('mo-lunar-lander-v3 requires the mo-gymnasium package to be installed') from exc

        env = mo_gym.make('mo-lunar-lander-v3', render_mode=render_mode)
        env = ActionMaskProxy(env)
        n_actions = int(env.action_space.n)
        state_dim = int(np.prod(env.observation_space.shape))
        reward_dim = int(env.unwrapped.reward_dim)
        max_return = np.array([100.0, 300.0, 0.0, 0.0], dtype=np.float32)
        ref_point = np.array([-100.0, -500.0, -1000.0, -1000.0], dtype=np.float32)
        model = None
        if include_model:
            scaling_factor = torch.tensor([[0.01] * reward_dim], dtype=torch.float32, device=device)
            model = WalkroomModel(
                n_states=state_dim,
                n_actions=n_actions,
                n_objectives=reward_dim,
                scaling_factor=scaling_factor,
                n_hidden=512,
            ).to(device)
        # Heuristic-landing demonstrations seed the replay buffer so the front
        # contains successful (objective-0 = +100) landings; the heuristic
        # terminates naturally within the default step limit, so the main env is
        # reused for seeding (no extended-horizon er_env needed).
        er_policy = LunarLanderSeedPolicy()
        setup = ExperimentSetup(
            env_name=env_name,
            env=env,
            n_actions=n_actions,
            model=model,
            learning_rate=1e-3,
            min_learning_rate=1e-3,
            total_steps=int(2.5e7),
            batch_size=256,
            n_model_updates=30,
            n_er_episodes=5000,
            max_size=5000,
            max_return=max_return,
            ref_point=ref_point,
            er_policy=er_policy,
            metadata={
                'library_env_id': 'mo-lunar-lander-v3',
                'deprecated_aliases': ['mo-lunar-lander-v2'],
                'er_seed_policy': 'lunarlander_heuristic_mix',
            },
        )

    elif env_name == 'mo-reacher-v5':
        try:
            import mo_gymnasium as mo_gym
        except ImportError as exc:
            raise ImportError('mo-reacher-v5 requires the mo-gymnasium package to be installed') from exc

        env = mo_gym.make('mo-reacher-v5', render_mode=render_mode)
        env = ActionMaskProxy(env)
        n_actions = int(env.action_space.n)
        state_dim = int(np.prod(env.observation_space.shape))
        reward_dim = int(env.unwrapped.reward_dim)
        max_episode_steps = int(getattr(getattr(env, 'spec', None), 'max_episode_steps', 50))
        max_return = np.full(reward_dim, float(max_episode_steps), dtype=np.float32)
        ref_point = np.full(reward_dim, -float(max_episode_steps), dtype=np.float32)
        model = None
        if include_model:
            scaling_factor = torch.tensor([[0.02] * reward_dim], dtype=torch.float32, device=device)
            model = WalkroomModel(
                n_states=state_dim,
                n_actions=n_actions,
                n_objectives=reward_dim,
                scaling_factor=scaling_factor,
                n_hidden=512,
            ).to(device)
        setup = ExperimentSetup(
            env_name=env_name,
            env=env,
            n_actions=n_actions,
            model=model,
            learning_rate=1e-3,
            min_learning_rate=1e-3,
            total_steps=int(2e7),
            batch_size=256,
            n_model_updates=30,
            n_er_episodes=5000,
            max_size=5000,
            max_return=max_return,
            ref_point=ref_point,
        )

    elif env_name == 'fruit-tree-v0':
        if fruit_tree_depth not in (5, 6, 7):
            raise ValueError('fruit-tree depth must be 5, 6 or 7')

        try:
            import mo_gymnasium as mo_gym
        except ImportError as exc:
            raise ImportError('fruit-tree-v0 requires the mo-gymnasium package to be installed') from exc

        env = mo_gym.make('fruit-tree-v0', depth=fruit_tree_depth, render_mode=render_mode)
        env = ActionMaskProxy(env)
        n_actions = int(env.action_space.n)
        pareto_front = np.asarray(env.unwrapped.pareto_front(1.0), dtype=np.float32)
        max_return = np.max(pareto_front, axis=0).astype(np.float32)
        ref_point = np.zeros_like(max_return, dtype=np.float32)
        reward_dim = int(env.unwrapped.reward_dim)
        model = None
        if include_model:
            scaling_factor = torch.tensor([[0.1] * reward_dim], dtype=torch.float32, device=device)
            model = FruitTreeModel(
                depth=fruit_tree_depth,
                n_actions=n_actions,
                n_objectives=reward_dim,
                scaling_factor=scaling_factor,
                n_hidden=256,
            ).to(device)
        replay_size = max(256, 2 * len(pareto_front))
        setup = ExperimentSetup(
            env_name=env_name,
            env=env,
            n_actions=n_actions,
            model=model,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            total_steps=int(3e5),
            batch_size=256,
            n_model_updates=20,
            n_er_episodes=replay_size,
            max_size=replay_size,
            max_return=max_return,
            ref_point=ref_point,
            metadata={'fruit_tree_depth': fruit_tree_depth},
        )

    elif env_name == 'four-room-v0':
        try:
            import mo_gymnasium as mo_gym
        except ImportError as exc:
            raise ImportError('four-room-v0 requires the mo-gymnasium package to be installed') from exc

        env = mo_gym.make('four-room-v0', render_mode=render_mode)
        env = FourRoomObservation(env)
        env = TimeLimit(env, 150)
        env = ActionMaskProxy(env)
        n_actions = int(env.action_space.n)

        base_env = env.unwrapped
        reward_dim = int(base_env.reward_dim)
        max_return = np.ones(reward_dim, dtype=np.float32)
        for row, col in base_env.shape_ids:
            shape_index = base_env.all_shapes[base_env.maze[row, col]]
            max_return[shape_index] += 1.0
        ref_point = np.zeros_like(max_return, dtype=np.float32)

        model = None
        if include_model:
            scaling_factor = torch.tensor([[0.25] * reward_dim], dtype=torch.float32, device=device)
            model = FourRoomModel(
                height=base_env.height,
                width=base_env.width,
                n_items=len(base_env.shape_ids),
                n_actions=n_actions,
                n_objectives=reward_dim,
                scaling_factor=scaling_factor,
                position_dim=96,
                inventory_dim=96,
                n_hidden=512,
            ).to(device)
        setup = ExperimentSetup(
            env_name=env_name,
            env=env,
            n_actions=n_actions,
            model=model,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            total_steps=int(2e7),
            batch_size=512,
            n_model_updates=20,
            n_er_episodes=70000,
            max_size=70000,
            max_return=max_return,
            ref_point=ref_point,
        )

    elif env_name == 'resource-gathering-v0':
        try:
            import mo_gymnasium as mo_gym
        except ImportError as exc:
            raise ImportError('resource-gathering-v0 requires the mo-gymnasium package to be installed') from exc

        env = mo_gym.make('resource-gathering-v0', render_mode=render_mode)
        env = ResourceGatheringIndexObservation(env)
        env = ActionMaskProxy(env)
        n_actions = int(env.action_space.n)

        base_env = env.unwrapped
        reward_dim = int(base_env.reward_dim)
        max_return = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        ref_point = np.array([-1.1, -0.1, -0.1], dtype=np.float32)

        model = None
        if include_model:
            scaling_factor = torch.tensor([[1.0] * reward_dim], dtype=torch.float32, device=device)
            model = DiscreteCommandModel(
                n_states=env.observation_space.n,
                n_actions=n_actions,
                n_objectives=reward_dim,
                scaling_factor=scaling_factor,
                n_hidden=256,
            ).to(device)
        setup = ExperimentSetup(
            env_name=env_name,
            env=env,
            n_actions=n_actions,
            model=model,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            total_steps=int(1e6),
            batch_size=512,
            n_model_updates=20,
            n_er_episodes=10000,
            max_size=10000,
            max_return=max_return,
            ref_point=ref_point,
        )

    elif env_name == 'breakable-bottles-v0':
        try:
            import mo_gymnasium as mo_gym
        except ImportError as exc:
            raise ImportError('breakable-bottles-v0 requires the mo-gymnasium package to be installed') from exc

        env = mo_gym.make('breakable-bottles-v0', render_mode=render_mode)
        env = BreakableBottlesIndexObservation(env)
        env = ActionMaskProxy(env)
        n_actions = int(env.action_space.n)

        base_env = env.unwrapped
        reward_dim = int(base_env.reward_dim)
        max_episode_steps = int(getattr(getattr(env, 'spec', None), 'max_episode_steps', 100))
        max_return = np.array([-1.0, 2.0 * base_env.bottle_reward, 0.0], dtype=np.float32)
        ref_point = np.array(
            [base_env.time_penalty * max_episode_steps - 1.0, -0.1, -1.1],
            dtype=np.float32,
        )

        model = None
        if include_model:
            scaling_factor = torch.tensor([[0.1] * reward_dim], dtype=torch.float32, device=device)
            model = DiscreteCommandModel(
                n_states=env.observation_space.n,
                n_actions=n_actions,
                n_objectives=reward_dim,
                scaling_factor=scaling_factor,
                n_hidden=256,
            ).to(device)
        setup = ExperimentSetup(
            env_name=env_name,
            env=env,
            n_actions=n_actions,
            model=model,
            learning_rate=1e-2,
            min_learning_rate=1e-4,
            total_steps=int(5e5),
            batch_size=256,
            n_model_updates=100,
            n_er_episodes=10000,
            max_size=10000,
            max_return=max_return,
            ref_point=ref_point,
        )

    else:
        raise ValueError(f'unsupported environment: {env_name}')

    setup.env.nA = setup.n_actions
    if setup.er_env is not None:
        setup.er_env.nA = setup.n_actions
    return setup
