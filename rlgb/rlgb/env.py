"""Gym-style RL environment over rlgb — fully wired by TOML, zero hardcoding.

The action set, observation type, frame budget per action, and the reward /
done / info functions are all declared in the ``[env]`` table. Reward logic
plugs in as a dotted path ``"my_pkg.rewards:pokemon_levels"`` — the function
receives the live GameBoy (full memory access) and returns a float.

Works standalone; if ``gymnasium`` is installed the spaces attributes become
real gym spaces, so it drops straight into vectorized RL stacks.

CC BY-NC-ND 4.0 license. Built by BuiltNotTaught.
"""
from __future__ import annotations

import numpy as np

from .config import load_callable
from .gameboy import GameBoy

__all__ = ["GameBoyEnv"]


class GameBoyEnv:
    def __init__(self, rom_path: str | None = None, config: str | None = None,
                 **overrides):
        self.gb = GameBoy(rom_path, config, **overrides)
        env_cfg = self.gb.config["env"]

        self.actions: list[str] = list(env_cfg["actions"])
        self.obs_mode: str = env_cfg["obs"]
        self.frames_per_action: int = int(env_cfg["frames_per_action"])
        self.press_frames: int = int(env_cfg["press_frames"])
        self.max_steps: int = int(env_cfg["max_steps"])

        self._reward_fn = load_callable(env_cfg["reward"])
        self._done_fn = load_callable(env_cfg["done"])
        self._info_fn = load_callable(env_cfg["info"])

        self._initial_state = self.gb.save_state()
        self._steps = 0

        try:  # optional gymnasium integration
            import gymnasium as gym
            self.action_space = gym.spaces.Discrete(len(self.actions))
            shape = (144, 160, 3) if self.obs_mode == "rgb" else (144, 160)
            self.observation_space = gym.spaces.Box(0, 255, shape, np.uint8)
        except ImportError:
            self.action_space = len(self.actions)
            self.observation_space = None

    # -------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        if self.obs_mode == "rgb":
            return self.gb.screen_rgb
        if self.obs_mode == "gray":
            return self.gb.screen_gray
        return self.gb.screen.copy()

    def reset(self, *, seed=None, options=None):
        self.gb.load_state(self._initial_state)
        self._steps = 0
        info = self._info_fn(self.gb) if self._info_fn else {}
        return self._obs(), info

    def step(self, action: int):
        name = self.actions[int(action)]
        press = min(self.press_frames, self.frames_per_action)
        if name:
            self.gb.set_buttons(name)
            self.gb.tick(press)
            self.gb.release()
            self.gb.tick(self.frames_per_action - press)
        else:
            self.gb.tick(self.frames_per_action)

        self._steps += 1
        reward = float(self._reward_fn(self.gb)) if self._reward_fn else 0.0
        terminated = bool(self._done_fn(self.gb)) if self._done_fn else False
        truncated = bool(self.max_steps and self._steps >= self.max_steps)
        info = self._info_fn(self.gb) if self._info_fn else {}
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        return self.gb.screen_rgb

    def close(self):
        self.gb.close()
