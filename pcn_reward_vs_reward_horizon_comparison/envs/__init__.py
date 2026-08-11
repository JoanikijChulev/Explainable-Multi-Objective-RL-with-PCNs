from gymnasium.envs.registration import register

from envs.branch_path import BranchPathEnv
from envs.ordered_pair_morl_grid import OrderedPairMORLGridEnv
from envs.reward_line import RewardLineEnv
from envs.three_tree import ThreeTreeEnv

# WalkRoom env
register(
        id='Walkroom2D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 20, 'dimensions': 2},
        disable_env_checker=True,
    )

register(
        id='Walkroom3D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 20, 'dimensions': 3},
        disable_env_checker=True,
    )

register(
        id='Walkroom4D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 20, 'dimensions': 4},
        disable_env_checker=True,
    )

register(
        id='Walkroom5D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 20, 'dimensions': 5},
        disable_env_checker=True,
    )

register(
        id='Walkroom6D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 10, 'dimensions': 6},
        disable_env_checker=True,
    )

register(
        id='Walkroom7D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 10, 'dimensions': 7},
        disable_env_checker=True,
    )

register(
        id='Walkroom8D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 10, 'dimensions': 8},
        disable_env_checker=True,
    )

register(
        id='Walkroom9D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 10, 'dimensions': 9},
        disable_env_checker=True,
    )

register(
        id='collect_two-v0',
        entry_point='envs.ordered_pair_morl_grid:OrderedPairMORLGridEnv',
        disable_env_checker=True,
    )

register(
        id='branch-path-v0',
        entry_point='envs.branch_path:BranchPathEnv',
        disable_env_checker=True,
    )

register(
        id='reward-line-v0',
        entry_point='envs.reward_line:RewardLineEnv',
        disable_env_checker=True,
    )

register(
        id='three-tree-v0',
        entry_point='envs.three_tree:ThreeTreeEnv',
        disable_env_checker=True,
    )
