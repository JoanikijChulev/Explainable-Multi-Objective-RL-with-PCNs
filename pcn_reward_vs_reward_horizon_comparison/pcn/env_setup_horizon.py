import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcn.env_setup import (
    ExperimentSetup,
    build_experiment_setup as build_reward_experiment_setup,
    canonical_env_name,
)


def scaled_horizon_command_tensor(desired_return, desired_horizon, scaling_factor):
    desired_return = desired_return.float()
    desired_horizon = desired_horizon.float()
    if desired_horizon.ndim == 1:
        desired_horizon = desired_horizon.unsqueeze(1)
    command = torch.cat((desired_return, desired_horizon), dim=-1)
    return command * scaling_factor[:, :command.shape[-1]]


class HorizonDiscreteCommandModel(nn.Module):

    def __init__(self, n_states, n_actions, n_objectives, scaling_factor, n_hidden=128):
        super(HorizonDiscreteCommandModel, self).__init__()
        self.n_states = n_states
        self.n_objectives = n_objectives
        self.scaling_factor = scaling_factor
        self.s_emb = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid(),
        )
        self.c_emb = nn.Sequential(
            nn.Linear(n_objectives + 1, n_hidden),
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

    def forward(self, state, desired_return, desired_horizon):
        commands = scaled_horizon_command_tensor(
            desired_return,
            desired_horizon,
            self.scaling_factor,
        )
        state_embedding = self.s_emb(self.encode_state(state))
        command_embedding = self.c_emb(commands)
        return self.fc(state_embedding * command_embedding)


class HorizonVectorCommandModel(nn.Module):

    def __init__(self, n_states, n_actions, n_objectives, scaling_factor, n_hidden=256):
        super(HorizonVectorCommandModel, self).__init__()
        self.n_objectives = n_objectives
        self.scaling_factor = scaling_factor
        self.s_emb = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid(),
        )
        self.c_emb = nn.Sequential(
            nn.Linear(n_objectives + 1, n_hidden),
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

    def forward(self, state, desired_return, desired_horizon):
        commands = scaled_horizon_command_tensor(
            desired_return,
            desired_horizon,
            self.scaling_factor,
        )
        state_embedding = self.s_emb(state.float())
        command_embedding = self.c_emb(commands)
        return self.fc(state_embedding * command_embedding)


class HorizonFourRoomModel(nn.Module):

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
        super(HorizonFourRoomModel, self).__init__()
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
            nn.Linear(n_objectives + 1, n_hidden),
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

    def forward(self, state, desired_return, desired_horizon):
        state = state.float()
        commands = scaled_horizon_command_tensor(
            desired_return,
            desired_horizon,
            self.scaling_factor,
        )

        cell_index, inventory = self.decode_state(state)
        position_embedding = self.position_embedding(cell_index)
        inventory_embedding = self.inventory_encoder(inventory)
        state_embedding = self.s_emb(torch.cat((position_embedding, inventory_embedding), dim=-1))
        command_embedding = self.c_emb(commands)
        return self.fc(state_embedding * command_embedding)


class HorizonMinecartModel(nn.Module):

    def __init__(self, n_actions, scaling_factor, state_dim=7, n_hidden=256):
        super(HorizonMinecartModel, self).__init__()
        self.scaling_factor = scaling_factor
        self.n_objectives = scaling_factor.shape[-1] - 1
        self.s_emb = nn.Sequential(
            nn.Linear(state_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid(),
        )
        self.c_emb = nn.Sequential(
            nn.Linear(self.n_objectives + 1, n_hidden),
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

    def forward(self, state, desired_return, desired_horizon):
        commands = scaled_horizon_command_tensor(
            desired_return,
            desired_horizon,
            self.scaling_factor,
        )
        state_embedding = self.s_emb(state.float())
        command_embedding = self.c_emb(commands)
        return self.fc(state_embedding * command_embedding)


class HorizonFruitTreeModel(HorizonDiscreteCommandModel):

    def __init__(self, depth, n_actions, n_objectives, scaling_factor, n_hidden=256):
        self.depth = depth
        n_states = 2 ** (depth + 1) - 1
        super(HorizonFruitTreeModel, self).__init__(
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
        state_index = torch.bitwise_left_shift(torch.ones_like(rows), rows) - 1 + positions
        return super().encode_state(state_index)


def _max_episode_steps(env, fallback=200):
    spec = getattr(env, 'spec', None)
    if spec is not None and getattr(spec, 'max_episode_steps', None) is not None:
        return int(spec.max_episode_steps)

    current = env
    while current is not None:
        value = getattr(current, '_max_episode_steps', None)
        if value is not None:
            return int(value)
        value = getattr(current, 'max_steps', None)
        if value is not None:
            return int(value)
        value = getattr(current, 'max_depth', None)
        if value is not None:
            return int(value)
        current = getattr(current, 'env', None)

    return int(fallback)


def _scaling_factor(reward_scales, horizon_steps, device):
    horizon_scale = 1.0 / max(float(horizon_steps), 1.0)
    return torch.tensor(
        [list(reward_scales) + [horizon_scale]],
        dtype=torch.float32,
        device=device,
    )


def _vector_state_dim(env):
    return int(np.prod(env.observation_space.shape))


def _build_horizon_model(setup, device, fruit_tree_depth):
    env_name = canonical_env_name(setup.env_name)
    env = setup.env
    n_actions = int(setup.n_actions)
    reward_dim = int(setup.max_return.shape[-1])
    horizon_steps = _max_episode_steps(env)

    if env_name == 'dst':
        scaling_factor = _scaling_factor([0.1, 0.1], horizon_steps, device)
        return HorizonDiscreteCommandModel(
            n_states=env.observation_space.n,
            n_actions=n_actions,
            n_objectives=reward_dim,
            scaling_factor=scaling_factor,
            n_hidden=256,
        ).to(device)

    if env_name == 'collect_two':
        scaling_factor = _scaling_factor([1.0] * reward_dim, horizon_steps, device)
        return HorizonVectorCommandModel(
            n_states=_vector_state_dim(env),
            n_actions=n_actions,
            n_objectives=reward_dim,
            scaling_factor=scaling_factor,
            n_hidden=512,
        ).to(device)

    if env_name == 'branch-path':
        scaling_factor = _scaling_factor([1.0] * reward_dim, horizon_steps, device)
        return HorizonVectorCommandModel(
            n_states=_vector_state_dim(env),
            n_actions=n_actions,
            n_objectives=reward_dim,
            scaling_factor=scaling_factor,
            n_hidden=512,
        ).to(device)

    if env_name == 'reward-line':
        scaling_factor = _scaling_factor([1.0] * reward_dim, horizon_steps, device)
        return HorizonVectorCommandModel(
            n_states=_vector_state_dim(env),
            n_actions=n_actions,
            n_objectives=reward_dim,
            scaling_factor=scaling_factor,
            n_hidden=256,
        ).to(device)

    if env_name == 'three-tree':
        scaling_factor = _scaling_factor([1.0] * reward_dim, horizon_steps, device)
        return HorizonVectorCommandModel(
            n_states=_vector_state_dim(env),
            n_actions=n_actions,
            n_objectives=reward_dim,
            scaling_factor=scaling_factor,
            n_hidden=512,
        ).to(device)

    if env_name.startswith('walkroom'):
        scaling_factor = _scaling_factor([0.1] * reward_dim, horizon_steps, device)
        return HorizonVectorCommandModel(
            n_states=_vector_state_dim(env),
            n_actions=n_actions,
            n_objectives=reward_dim,
            scaling_factor=scaling_factor,
            n_hidden=256,
        ).to(device)

    if env_name == 'minecart':
        scaling_factor = _scaling_factor([1.0, 1.0, 0.1], horizon_steps, device)
        return HorizonMinecartModel(
            n_actions=n_actions,
            scaling_factor=scaling_factor,
            state_dim=_vector_state_dim(env),
            n_hidden=1024,
        ).to(device)

    if env_name == 'mo-mountaincar-timespeed-v0':
        scaling_factor = _scaling_factor([0.01] * reward_dim, horizon_steps, device)
        return HorizonVectorCommandModel(
            n_states=_vector_state_dim(env),
            n_actions=n_actions,
            n_objectives=reward_dim,
            scaling_factor=scaling_factor,
            n_hidden=256,
        ).to(device)

    if env_name == 'mo-lunar-lander-v3':
        scaling_factor = _scaling_factor([0.01] * reward_dim, horizon_steps, device)
        return HorizonVectorCommandModel(
            n_states=_vector_state_dim(env),
            n_actions=n_actions,
            n_objectives=reward_dim,
            scaling_factor=scaling_factor,
            n_hidden=512,
        ).to(device)

    if env_name == 'mo-reacher-v5':
        scaling_factor = _scaling_factor([0.02] * reward_dim, horizon_steps, device)
        return HorizonVectorCommandModel(
            n_states=_vector_state_dim(env),
            n_actions=n_actions,
            n_objectives=reward_dim,
            scaling_factor=scaling_factor,
            n_hidden=512,
        ).to(device)

    if env_name == 'fruit-tree-v0':
        scaling_factor = _scaling_factor([0.1] * reward_dim, horizon_steps, device)
        return HorizonFruitTreeModel(
            depth=fruit_tree_depth,
            n_actions=n_actions,
            n_objectives=reward_dim,
            scaling_factor=scaling_factor,
            n_hidden=256,
        ).to(device)

    if env_name == 'four-room-v0':
        base_env = env.unwrapped
        scaling_factor = _scaling_factor([0.25] * reward_dim, horizon_steps, device)
        return HorizonFourRoomModel(
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

    if env_name == 'resource-gathering-v0':
        scaling_factor = _scaling_factor([1.0] * reward_dim, horizon_steps, device)
        return HorizonDiscreteCommandModel(
            n_states=env.observation_space.n,
            n_actions=n_actions,
            n_objectives=reward_dim,
            scaling_factor=scaling_factor,
            n_hidden=256,
        ).to(device)

    if env_name == 'breakable-bottles-v0':
        scaling_factor = _scaling_factor([0.1] * reward_dim, horizon_steps, device)
        return HorizonDiscreteCommandModel(
            n_states=env.observation_space.n,
            n_actions=n_actions,
            n_objectives=reward_dim,
            scaling_factor=scaling_factor,
            n_hidden=256,
        ).to(device)

    raise ValueError(f'unsupported environment for horizon model: {env_name}')


def build_horizon_experiment_setup(
    env_name,
    device='cpu',
    fruit_tree_depth=6,
    include_model=True,
    render_mode=None,
):
    setup = build_reward_experiment_setup(
        env_name,
        device=device,
        fruit_tree_depth=fruit_tree_depth,
        include_model=False,
        render_mode=render_mode,
    )
    if include_model:
        setup.model = _build_horizon_model(setup, device, fruit_tree_depth)
    else:
        setup.model = None

    metadata = dict(setup.metadata)
    metadata.update(
        {
            'pcn_variant': 'horizon',
            'horizon_conditioned': True,
            'command_inputs': ['desired_return', 'desired_horizon'],
        }
    )
    setup.metadata = metadata
    return setup


def build_experiment_setup(
    env_name,
    device='cpu',
    fruit_tree_depth=6,
    include_model=True,
    render_mode=None,
):
    return build_horizon_experiment_setup(
        env_name,
        device=device,
        fruit_tree_depth=fruit_tree_depth,
        include_model=include_model,
        render_mode=render_mode,
    )
