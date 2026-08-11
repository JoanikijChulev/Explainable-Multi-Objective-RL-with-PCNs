import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled


class BranchPathEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    LAYOUT = (
        "A.A.A.B",
        "###.###",
        "###S###",
        "###.###",
        "B.B.B.A",
    )
    COLLECTIBLES = (
        ("A", (0, 0)),
        ("A", (0, 2)),
        ("A", (0, 4)),
        ("B", (0, 6)),
        ("B", (4, 0)),
        ("B", (4, 2)),
        ("B", (4, 4)),
        ("A", (4, 6)),
    )
    ACTION_TO_DELTA = {
        0: np.array([0, 1], dtype=np.int64),   # right
        1: np.array([-1, 0], dtype=np.int64),  # up
        2: np.array([0, -1], dtype=np.int64),  # left
        3: np.array([1, 0], dtype=np.int64),   # down
    }

    def __init__(self, max_steps=13, render_mode=None, window_size=700):
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
        self.start_pos = self._single_position("S")
        self.walls = {tuple(pos.tolist()) for pos in np.argwhere(self.grid == "#")}

        self.reward_dim = 2
        self.objective_types = tuple(objective_type for objective_type, _ in self.COLLECTIBLES)
        self.objective_positions = tuple(
            np.asarray(position, dtype=np.int64) for _, position in self.COLLECTIBLES
        )
        self.position_to_objective = {
            tuple(position.tolist()): index
            for index, position in enumerate(self.objective_positions)
        }

        self.action_space = spaces.Discrete(len(self.ACTION_TO_DELTA))
        self.observation_space = spaces.MultiDiscrete([self.rows, self.cols] + [2] * 8)
        self.reward_space = spaces.Box(
            low=np.zeros(self.reward_dim, dtype=np.float32),
            high=np.ones(self.reward_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.window = None
        self.clock = None
        self._pygame = None

        self.position = self.start_pos.copy()
        self.collected = np.zeros(len(self.COLLECTIBLES), dtype=bool)
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
        total_a = sum(
            int(collected)
            for collected, objective_type in zip(self.collected, self.objective_types)
            if objective_type == "A"
        )
        total_b = sum(
            int(collected)
            for collected, objective_type in zip(self.collected, self.objective_types)
            if objective_type == "B"
        )
        return {
            "position": (int(self.position[0]), int(self.position[1])),
            "collected": [int(value) for value in self.collected],
            "num_collected": int(np.count_nonzero(self.collected)),
            "collection_order": [int(index) for index in self.collection_order],
            "steps": int(self.steps),
            "total_A_collected": int(total_a),
            "total_B_collected": int(total_b),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.start_pos.copy()
        self.collected = np.zeros(len(self.COLLECTIBLES), dtype=bool)
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

        objective_index = self.position_to_objective.get(tuple(self.position.tolist()))
        if objective_index is not None and not self.collected[objective_index]:
            self.collected[objective_index] = True
            self.collection_order.append(objective_index)
            objective_type = self.objective_types[objective_index]
            reward[0 if objective_type == "A" else 1] = np.float32(1.0)

        terminated = bool(np.all(self.collected))
        truncated = bool(not terminated and self.steps >= self.max_steps)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

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
                    "pygame is required for BranchPathEnv rendering"
                ) from exc
            self._pygame = pygame
        return self._pygame

    def _render_frame(self):
        pygame = self._load_pygame()
        pix_square_size = self.window_size / self.cols
        canvas_width = int(round(self.cols * pix_square_size))
        canvas_height = int(round(self.rows * pix_square_size))

        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((canvas_width, canvas_height))
            if self.clock is None:
                self.clock = pygame.time.Clock()

        canvas = pygame.Surface((canvas_width, canvas_height))
        canvas.fill((255, 255, 255))

        for row in range(self.rows):
            for col in range(self.cols):
                rect = self._cell_rect(pygame, row, col, pix_square_size)
                color = (50, 50, 50) if (row, col) in self.walls else (255, 255, 255)
                pygame.draw.rect(canvas, color, rect)

        for objective_index, objective_type in enumerate(self.objective_types):
            self._draw_objective(canvas, pygame, objective_index, objective_type, pix_square_size)

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
                (pixel_x, canvas_height),
                width=1,
            )
        for y in range(self.rows + 1):
            pixel_y = int(round(y * pix_square_size))
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, pixel_y),
                (canvas_width, pixel_y),
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

    def _draw_objective(self, canvas, pygame, objective_index, objective_type, pix_square_size):
        position = self.objective_positions[objective_index]
        row, col = int(position[0]), int(position[1])
        center_x, center_y = self._cell_center(row, col, pix_square_size)
        radius = max(4, int(pix_square_size * 0.28))

        if objective_type == "A":
            color = (245, 185, 185) if self.collected[objective_index] else (220, 40, 40)
            half = radius
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(center_x - half, center_y - half, half * 2, half * 2),
            )
        else:
            color = (185, 230, 195) if self.collected[objective_index] else (30, 150, 70)
            points = [
                (center_x, center_y - radius),
                (center_x - radius, center_y + radius),
                (center_x + radius, center_y + radius),
            ]
            pygame.draw.polygon(canvas, color, points)

    def close(self):
        if self._pygame is not None:
            if self.window is not None:
                self._pygame.display.quit()
            self._pygame.quit()
        self.window = None
        self.clock = None
        self._pygame = None
