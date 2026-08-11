import itertools

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled


class ThreeTreeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    ACTION_LABELS = ("P", "S", "E")
    RECOVERY_NODES = {"PP", "PPP", "PPE"}

    REWARD_TABLE = {
        "": {
            0: [1.00, 0.10, 0.10],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
        "P": {
            0: [1.10, 0.05, 0.05],
            1: [0.45, 0.80, 0.05],
            2: [0.45, 0.05, 0.80],
        },
        "S": {
            0: [0.80, 0.45, 0.05],
            1: [0.05, 1.10, 0.05],
            2: [0.05, 0.45, 0.80],
        },
        "E": {
            0: [0.80, 0.05, 0.45],
            1: [0.05, 0.80, 0.45],
            2: [0.05, 0.05, 1.10],
        },
        "PP": {
            0: [0.10, 0.00, 0.05],
            1: [1.20, 1.80, 1.20],
            2: [0.05, 0.20, 0.10],
        },
        "PS": {
            0: [0.70, 0.05, 0.10],
            1: [0.05, 0.70, 0.10],
            2: [0.15, 0.15, 1.40],
        },
        "PE": {
            0: [0.70, 0.10, 0.05],
            1: [0.15, 1.40, 0.15],
            2: [0.05, 0.10, 0.70],
        },
        "SP": {
            0: [0.70, 0.05, 0.10],
            1: [0.05, 0.70, 0.10],
            2: [0.15, 0.15, 1.40],
        },
        "SS": {
            0: [1.40, 0.15, 0.15],
            1: [0.10, 0.70, 0.05],
            2: [0.10, 0.05, 0.70],
        },
        "SE": {
            0: [1.40, 0.15, 0.15],
            1: [0.10, 0.70, 0.05],
            2: [0.10, 0.05, 0.70],
        },
        "EP": {
            0: [0.70, 0.10, 0.05],
            1: [0.15, 1.40, 0.15],
            2: [0.05, 0.10, 0.70],
        },
        "ES": {
            0: [1.40, 0.15, 0.15],
            1: [0.10, 0.70, 0.05],
            2: [0.10, 0.05, 0.70],
        },
        "EE": {
            0: [1.40, 0.15, 0.15],
            1: [0.10, 0.70, 0.05],
            2: [0.10, 0.05, 0.70],
        },
        "PPP": {
            0: [0.05, 0.05, 0.05],
            1: [1.20, 1.20, 1.20],
            2: [0.05, 0.05, 0.05],
        },
        "PPE": {
            0: [0.05, 0.05, 0.05],
            1: [1.20, 1.20, 1.20],
            2: [0.05, 0.05, 0.05],
        },
        "PPS": {
            0: [1.00, 0.10, 0.10],
            1: [0.10, 1.00, 0.10],
            2: [0.15, 0.15, 1.30],
        },
        "PSP": {
            0: [1.00, 0.10, 0.10],
            1: [0.10, 1.00, 0.10],
            2: [0.15, 0.15, 1.30],
        },
        "PSS": {
            0: [1.00, 0.10, 0.10],
            1: [0.10, 1.00, 0.10],
            2: [0.15, 0.15, 1.30],
        },
        "PSE": {
            0: [1.30, 0.15, 0.15],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
        "PEP": {
            0: [1.00, 0.10, 0.10],
            1: [0.15, 1.30, 0.15],
            2: [0.10, 0.10, 1.00],
        },
        "PES": {
            0: [1.30, 0.15, 0.15],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
        "PEE": {
            0: [1.00, 0.10, 0.10],
            1: [0.15, 1.30, 0.15],
            2: [0.10, 0.10, 1.00],
        },
        "SPP": {
            0: [1.00, 0.10, 0.10],
            1: [0.10, 1.00, 0.10],
            2: [0.15, 0.15, 1.30],
        },
        "SPS": {
            0: [1.00, 0.10, 0.10],
            1: [0.10, 1.00, 0.10],
            2: [0.15, 0.15, 1.30],
        },
        "SPE": {
            0: [1.30, 0.15, 0.15],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
        "SSP": {
            0: [1.00, 0.10, 0.10],
            1: [0.10, 1.00, 0.10],
            2: [0.15, 0.15, 1.30],
        },
        "SSS": {
            0: [1.30, 0.15, 0.15],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
        "SSE": {
            0: [1.30, 0.15, 0.15],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
        "SEP": {
            0: [1.30, 0.15, 0.15],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
        "SES": {
            0: [1.30, 0.15, 0.15],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
        "SEE": {
            0: [1.30, 0.15, 0.15],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
        "EPP": {
            0: [1.00, 0.10, 0.10],
            1: [0.15, 1.30, 0.15],
            2: [0.10, 0.10, 1.00],
        },
        "EPS": {
            0: [1.30, 0.15, 0.15],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
        "EPE": {
            0: [1.00, 0.10, 0.10],
            1: [0.15, 1.30, 0.15],
            2: [0.10, 0.10, 1.00],
        },
        "ESP": {
            0: [1.30, 0.15, 0.15],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
        "ESS": {
            0: [1.30, 0.15, 0.15],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
        "ESE": {
            0: [1.30, 0.15, 0.15],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
        "EEP": {
            0: [1.00, 0.10, 0.10],
            1: [0.15, 1.30, 0.15],
            2: [0.10, 0.10, 1.00],
        },
        "EES": {
            0: [1.30, 0.15, 0.15],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
        "EEE": {
            0: [1.30, 0.15, 0.15],
            1: [0.10, 1.00, 0.10],
            2: [0.10, 0.10, 1.00],
        },
    }

    def __init__(self, render_mode=None, window_size=720):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        if int(window_size) <= 0:
            raise ValueError("window_size must be positive")

        self.render_mode = render_mode
        self.window_size = int(window_size)
        self.max_depth = 4
        self.reward_dim = 3

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.reward_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.window = None
        self.clock = None
        self._pygame = None

        self._validate_reward_table()
        self.reset()

    @property
    def depth(self):
        return len(self.path)

    @property
    def current_node(self):
        return "".join(self.path)

    @property
    def recovery_flag(self):
        return float(self.current_node in self.RECOVERY_NODES)

    @property
    def pareto_front(self):
        returns = np.asarray([self.return_for_path(path) for path in self.leaf_paths()], dtype=np.float32)
        keep = np.ones(len(returns), dtype=bool)
        for i, value in enumerate(returns):
            dominates = np.all(returns >= value, axis=1) & np.any(returns > value, axis=1)
            if np.any(dominates):
                keep[i] = False
        return returns[keep]

    def leaf_paths(self):
        return ["".join(path) for path in itertools.product(self.ACTION_LABELS, repeat=self.max_depth)]

    def return_for_path(self, path):
        node = ""
        total = np.zeros(self.reward_dim, dtype=np.float32)
        for label in path:
            action = self.ACTION_LABELS.index(label)
            total += np.asarray(self.REWARD_TABLE[node][action], dtype=np.float32)
            node += label
        return total.astype(np.float32)

    def _validate_reward_table(self):
        expected_nodes = {""}
        for depth in (1, 2, 3):
            expected_nodes.update(
                "".join(path) for path in itertools.product(self.ACTION_LABELS, repeat=depth)
            )
        missing = expected_nodes.difference(self.REWARD_TABLE)
        extra = set(self.REWARD_TABLE).difference(expected_nodes)
        if missing or extra:
            raise ValueError(f"invalid reward table nodes: missing={missing}, extra={extra}")
        for node, action_rewards in self.REWARD_TABLE.items():
            if set(action_rewards) != {0, 1, 2}:
                raise ValueError(f"node {node!r} must define rewards for actions 0, 1, 2")
            for reward in action_rewards.values():
                arr = np.asarray(reward, dtype=np.float32)
                if arr.shape != (self.reward_dim,):
                    raise ValueError(f"reward at node {node!r} must have shape ({self.reward_dim},)")

    def _get_obs(self):
        counts = [self.path.count(label) for label in self.ACTION_LABELS]
        return np.array(
            [
                self.depth / self.max_depth,
                counts[0] / self.max_depth,
                counts[1] / self.max_depth,
                counts[2] / self.max_depth,
                self.recovery_flag,
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        return {
            "path": self.current_node,
            "depth": int(self.depth),
            "counts": {label: int(self.path.count(label)) for label in self.ACTION_LABELS},
            "recovery_flag": bool(self.current_node in self.RECOVERY_NODES),
            "total_return": self.total_return.astype(np.float32).tolist(),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.path = []
        self.total_return = np.zeros(self.reward_dim, dtype=np.float32)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        if self.depth >= self.max_depth:
            return (
                self._get_obs(),
                np.zeros(self.reward_dim, dtype=np.float32),
                True,
                False,
                self._get_info(),
            )

        node = self.current_node
        reward = np.asarray(self.REWARD_TABLE[node][action], dtype=np.float32)
        self.path.append(self.ACTION_LABELS[action])
        self.total_return = (self.total_return + reward).astype(np.float32)
        terminated = bool(self.depth >= self.max_depth)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def get_action_mask(self):
        return np.ones(self.action_space.n, dtype=bool)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        if self.render_mode == "human":
            self._render_frame()
            return None
        return None

    def _load_pygame(self):
        if self._pygame is None:
            try:
                import pygame
            except ImportError as exc:
                raise DependencyNotInstalled("pygame is required for ThreeTreeEnv rendering") from exc
            self._pygame = pygame
        return self._pygame

    def _render_frame(self):
        pygame = self._load_pygame()
        canvas_width = self.window_size
        canvas_height = int(round(self.window_size * 0.78))

        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((canvas_width, canvas_height))
            if self.clock is None:
                self.clock = pygame.time.Clock()

        canvas = pygame.Surface((canvas_width, canvas_height))
        canvas.fill((255, 255, 255))

        level_y = [45, 140, 235, 330, 425]
        level_y = [min(y, canvas_height - 35) for y in level_y]
        positions = self._node_positions(canvas_width, level_y)
        edge_color = (190, 190, 190)
        active_edge_color = (45, 90, 220)

        for node, pos in positions.items():
            if len(node) >= self.max_depth:
                continue
            for label in self.ACTION_LABELS:
                child = node + label
                if child in positions:
                    color = active_edge_color if self.current_node.startswith(child) else edge_color
                    pygame.draw.line(canvas, color, pos, positions[child], width=2)

        for node, pos in positions.items():
            radius = 7 if len(node) < self.max_depth else 4
            if node == self.current_node:
                color = (45, 90, 220)
                radius = 12
            elif node in self.RECOVERY_NODES:
                color = (235, 205, 80)
            else:
                color = (120, 120, 120) if len(node) < self.max_depth else (180, 180, 180)
            pygame.draw.circle(canvas, color, pos, radius)

        self._draw_return_bars(canvas, pygame, canvas_width, canvas_height)

        if self.render_mode == "human":
            pygame.event.pump()
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(
            np.asarray(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        ).copy()

    def _node_positions(self, canvas_width, level_y):
        positions = {"": (canvas_width // 2, level_y[0])}
        for depth in range(1, self.max_depth + 1):
            nodes = ["".join(path) for path in itertools.product(self.ACTION_LABELS, repeat=depth)]
            margin = 32
            span = max(canvas_width - margin * 2, 1)
            denom = max(len(nodes) - 1, 1)
            for index, node in enumerate(nodes):
                x = int(round(margin + span * index / denom))
                positions[node] = (x, level_y[depth])
        return positions

    def _draw_return_bars(self, canvas, pygame, canvas_width, canvas_height):
        colors = [(210, 60, 60), (60, 150, 80), (65, 105, 220)]
        labels = self.ACTION_LABELS
        bar_left = 24
        bar_top = max(12, canvas_height - 36)
        bar_width = max(80, canvas_width - 48)
        segment_width = bar_width // self.reward_dim
        for i, value in enumerate(self.total_return):
            x = bar_left + i * segment_width
            width = int(round(segment_width * min(float(value) / 5.0, 1.0)))
            pygame.draw.rect(canvas, (230, 230, 230), pygame.Rect(x, bar_top, segment_width - 8, 14))
            pygame.draw.rect(canvas, colors[i], pygame.Rect(x, bar_top, max(width - 8, 0), 14))
            pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(x, bar_top, segment_width - 8, 14), width=1)
            self._draw_letter(canvas, pygame, labels[i], (x + 2, bar_top - 17), colors[i])

    def _draw_letter(self, canvas, pygame, text, pos, color):
        try:
            if not pygame.font.get_init():
                pygame.font.init()
            font = pygame.font.Font(None, 20)
            surface = font.render(text, True, color)
            canvas.blit(surface, pos)
        except pygame.error:
            pass

    def close(self):
        if self._pygame is not None:
            if self.window is not None:
                self._pygame.display.quit()
            self._pygame.quit()
        self.window = None
        self.clock = None
        self._pygame = None
