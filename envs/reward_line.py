import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled


class RewardLineEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    ACTION_TO_DELTA = {
        0: np.array([0, 1], dtype=np.int64),   # right
        1: np.array([-1, 0], dtype=np.int64),  # up
        2: np.array([0, -1], dtype=np.int64),  # left
        3: np.array([1, 0], dtype=np.int64),   # down
    }

    def __init__(self, rows=5, cols=11, max_steps=20, render_mode=None, window_size=660):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        if int(rows) < 2:
            raise ValueError("rows must be at least 2")
        if int(cols) < 2:
            raise ValueError("cols must be at least 2")
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        if int(window_size) <= 0:
            raise ValueError("window_size must be positive")

        self.rows = int(rows)
        self.cols = int(cols)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.window_size = int(window_size)

        self.start_pos = np.array([self.rows - 1, self.cols // 2], dtype=np.int64)
        self.reward_dim = 2

        self.action_space = spaces.Discrete(len(self.ACTION_TO_DELTA))
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([self.rows - 1, self.cols - 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.reward_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.window = None
        self.clock = None
        self._pygame = None

        self.position = self.start_pos.copy()
        self.steps = 0

    def _get_obs(self):
        return self.position.astype(np.float32)

    def _terminal_reward_for_col(self, col):
        u = np.float32(float(col) / float(self.cols - 1))
        return np.array([np.float32(1.0) - u, u], dtype=np.float32)

    def _get_info(self):
        is_terminal = bool(self.position[0] == 0)
        terminal_col = int(self.position[1]) if is_terminal else None
        terminal_u = (
            float(terminal_col) / float(self.cols - 1)
            if terminal_col is not None
            else None
        )
        pareto_reward = (
            self._terminal_reward_for_col(terminal_col)
            if terminal_col is not None
            else np.zeros(self.reward_dim, dtype=np.float32)
        )
        return {
            "position": (int(self.position[0]), int(self.position[1])),
            "steps": int(self.steps),
            "terminal_col": terminal_col,
            "terminal_u": terminal_u,
            "is_terminal": is_terminal,
            "pareto_reward": pareto_reward.astype(np.float32).tolist(),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.start_pos.copy()
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
        terminated = bool(self.position[0] == 0)
        truncated = bool(not terminated and self.steps >= self.max_steps)
        reward = (
            self._terminal_reward_for_col(int(self.position[1]))
            if terminated
            else np.zeros(self.reward_dim, dtype=np.float32)
        )

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _is_valid_position(self, position):
        row, col = int(position[0]), int(position[1])
        return 0 <= row < self.rows and 0 <= col < self.cols

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
                    "pygame is required for RewardLineEnv rendering"
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
                color = self._terminal_cell_color(col) if row == 0 else (255, 255, 255)
                pygame.draw.rect(canvas, color, rect)

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

    def _terminal_cell_color(self, col):
        u = float(col) / float(self.cols - 1)
        red = int(round(230 * (1.0 - u) + 235 * u))
        green = int(round(70 * (1.0 - u) + 205 * u))
        blue = int(round(70 * (1.0 - u) + 70 * u))
        return red, green, blue

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

    def close(self):
        if self._pygame is not None:
            if self.window is not None:
                self._pygame.display.quit()
            self._pygame.quit()
        self.window = None
        self.clock = None
        self._pygame = None
