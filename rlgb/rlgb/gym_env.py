"""Gymnasium-registered wrapper: ``gym.make("rlgb/GameBoy-v0", rom_path=...)``.

Thin ``gymnasium.Env`` subclass over :class:`rlgb.GameBoyEnv` so rlgb drops
into any framework that speaks the gymnasium API (SB3, CleanRL, RLlib, ...)
with zero glue code. All GameBoyEnv/TOML knobs pass straight through::

    import gymnasium as gym
    import rlgb  # noqa: F401  (import registers the env id)

    env = gym.make("rlgb/GameBoy-v0", rom_path="pokemon_red.gb",
                   env={"frames_per_action": 24,
                        "reward": "my_pkg.rewards:fn"})

CC BY-NC-ND 4.0 license. Built by BuiltNotTaught.
"""
from __future__ import annotations

import gymnasium as gym

from .env import GameBoyEnv

__all__ = ["RlgbGymEnv"]


class RlgbGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, rom_path: str | None = None, config: str | None = None,
                 render_mode: str | None = "rgb_array", **overrides):
        self._env = GameBoyEnv(rom_path, config, **overrides)
        self.render_mode = render_mode
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    @property
    def gb(self):
        """The underlying rlgb.GameBoy — full memory/register/state access."""
        return self._env.gb

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


def register():
    """Idempotently register rlgb/GameBoy-v0 with gymnasium."""
    if "rlgb/GameBoy-v0" not in gym.registry:
        gym.register(id="rlgb/GameBoy-v0",
                     entry_point="rlgb.gym_env:RlgbGymEnv")


register()
