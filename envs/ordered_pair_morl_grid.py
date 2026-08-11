import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled


class OrderedPairMORLGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    LAYOUT = (
        "...A...",
        ".##.##.",
        ".##.##.",
        "B..S..C",
        ".##.##.",
        ".##.##.",
        "...D...",
    )
    OBJECTIVE_LABELS = ("A", "B", "C", "D")
    ACTION_TO_DELTA = {
        0: np.array([0, 1], dtype=np.int64),   # right
        1: np.array([-1, 0], dtype=np.int64),  # up
        2: np.array([0, -1], dtype=np.int64),  # left
        3: np.array([1, 0], dtype=np.int64),   # down
    }

    def __init__(self, max_steps=50, render_mode=None, window_size=512):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        if int(window_size) <= 0:
            raise ValueError("window_size must be positive")

        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.window_size = int(window_size)

        self.grid = np.asarray([list(row) for row in self.LAYOUT])
        self.rows, self.cols = self.grid.shape
        self.size = self.rows
        self.reward_dim = len(self.OBJECTIVE_LABELS)

        self.start_pos = self._single_position("S")
        self.objective_positions = {
            label: self._single_position(label) for label in self.OBJECTIVE_LABELS
        }
        self.objective_to_index = {
            label: index for index, label in enumerate(self.OBJECTIVE_LABELS)
        }
        self.position_to_objective = {
            tuple(position.tolist()): label
            for label, position in self.objective_positions.items()
        }
        self.walls = {
            tuple(position.tolist())
            for position in np.argwhere(self.grid == "#")
        }

        self.action_space = spaces.Discrete(len(self.ACTION_TO_DELTA))
        self.observation_space = spaces.MultiDiscrete(
            [self.rows, self.cols, 2, 2, 2, 2]
        )
        self.reward_space = spaces.Box(
            low=np.zeros(self.reward_dim, dtype=np.float32),
            high=np.ones(self.reward_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.window = None
        self.clock = None
        self._pygame = None

        self.position = self.start_pos.copy()
        self.collected = np.zeros(self.reward_dim, dtype=bool)
        self.collection_order = []
        self.steps = 0

    def _single_position(self, marker):
        positions = np.argwhere(self.grid == marker)
        if len(positions) != 1:
            raise ValueError(f"layout must contain exactly one {marker!r} cell")
        return positions[0].astype(np.int64)

    def _get_obs(self):
        return np.asarray(
            [
                int(self.position[0]),
                int(self.position[1]),
                *[int(value) for value in self.collected],
            ],
            dtype=np.int64,
        )

    def _get_info(self):
        return {
            "position": (int(self.position[0]), int(self.position[1])),
            "collected": [int(value) for value in self.collected],
            "num_collected": int(np.count_nonzero(self.collected)),
            "collection_order": list(self.collection_order),
            "steps": int(self.steps),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.start_pos.copy()
        self.collected = np.zeros(self.reward_dim, dtype=bool)
        self.collection_order = []
        self.steps = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        candidate = self.position + self.ACTION_TO_DELTA[action]
        if self._is_valid_position(candidate):
            self.position = candidate

        self.steps += 1
        reward = np.zeros(self.reward_dim, dtype=np.float32)

        objective = self.position_to_objective.get(tuple(self.position.tolist()))
        if objective is not None:
            objective_index = self.objective_to_index[objective]
            if not self.collected[objective_index]:
                previous_count = int(np.count_nonzero(self.collected))
                self.collected[objective_index] = True
                self.collection_order.append(objective)
                reward[objective_index] = np.float32(1.0 if previous_count == 0 else 0.8)

        terminated = int(np.count_nonzero(self.collected)) >= 2
        truncated = bool(not terminated and self.steps >= self.max_steps)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, bool(terminated), truncated, self._get_info()

    def _is_valid_position(self, position):
        row, col = int(position[0]), int(position[1])
        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
            and (row, col) not in self.walls
        )

    def get_action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        for action, delta in self.ACTION_TO_DELTA.items():
            mask[action] = self._is_valid_position(self.position + delta)
        return mask

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
                raise DependencyNotInstalled(
                    "pygame is required for OrderedPairMORLGridEnv rendering"
                ) from exc
            self._pygame = pygame
        return self._pygame

    def _render_frame(self):
        pygame = self._load_pygame()
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
            if self.clock is None:
                self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.rows

        for row in range(self.rows):
            for col in range(self.cols):
                rect = self._cell_rect(pygame, row, col, pix_square_size)
                color = (50, 50, 50) if (row, col) in self.walls else (255, 255, 255)
                pygame.draw.rect(canvas, color, rect)

        for label in self.OBJECTIVE_LABELS:
            self._draw_objective(canvas, pygame, label, pix_square_size)

        agent_center = self._cell_center(self.position[0], self.position[1], pix_square_size)
        pygame.draw.circle(
            canvas,
            (45, 90, 220),
            agent_center,
            max(4, int(pix_square_size * 0.28)),
        )

        for x in range(self.cols + 1):
            pixel_x = int(round(x * pix_square_size))
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (pixel_x, 0),
                (pixel_x, self.window_size),
                width=1,
            )
        for y in range(self.rows + 1):
            pixel_y = int(round(y * pix_square_size))
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, pixel_y),
                (self.window_size, pixel_y),
                width=1,
            )

        if self.render_mode == "human":
            pygame.event.pump()
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(
            np.asarray(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        ).copy()

    def _cell_rect(self, pygame, row, col, pix_square_size):
        left = int(round(col * pix_square_size))
        top = int(round(row * pix_square_size))
        right = int(round((col + 1) * pix_square_size))
        bottom = int(round((row + 1) * pix_square_size))
        return pygame.Rect(left, top, right - left, bottom - top)

    def _cell_center(self, row, col, pix_square_size):
        return (
            int(round((int(col) + 0.5) * pix_square_size)),
            int(round((int(row) + 0.5) * pix_square_size)),
        )

    def _draw_objective(self, canvas, pygame, label, pix_square_size):
        colors = {
            "A": ((220, 40, 40), (245, 185, 185)),
            "B": ((30, 150, 70), (185, 230, 195)),
            "C": ((235, 135, 35), (250, 215, 170)),
            "D": ((135, 70, 180), (220, 190, 235)),
        }
        position = self.objective_positions[label]
        objective_index = self.objective_to_index[label]
        color = colors[label][1 if self.collected[objective_index] else 0]
        row, col = int(position[0]), int(position[1])
        center_x, center_y = self._cell_center(row, col, pix_square_size)
        radius = max(4, int(pix_square_size * 0.28))

        if label == "A":
            half = radius
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(center_x - half, center_y - half, half * 2, half * 2),
            )
        elif label == "B":
            points = [
                (center_x, center_y - radius),
                (center_x - radius, center_y + radius),
                (center_x + radius, center_y + radius),
            ]
            pygame.draw.polygon(canvas, color, points)
        elif label == "C":
            points = [
                (center_x, center_y - radius),
                (center_x + radius, center_y),
                (center_x, center_y + radius),
                (center_x - radius, center_y),
            ]
            pygame.draw.polygon(canvas, color, points)
        else:
            points = []
            for index in range(6):
                angle = math.pi / 6.0 + index * math.pi / 3.0
                points.append(
                    (
                        int(round(center_x + math.cos(angle) * radius)),
                        int(round(center_y + math.sin(angle) * radius)),
                    )
                )
            pygame.draw.polygon(canvas, color, points)

    def close(self):
        if self._pygame is not None:
            if self.window is not None:
                self._pygame.display.quit()
            self._pygame.quit()
        self.window = None
        self.clock = None
        self._pygame = None
