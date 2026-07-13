"""Vectorized multi-core env — N emulators stepping in parallel threads.

The C core releases the GIL for the whole ``gb_run_frames`` call (plain
ctypes CDLL), so N Python threads run N emulators on N cores concurrently;
only the cheap Python glue (reward fns, obs stacking) serializes. No
processes, no pickling, zero-copy state access stays available.

Follows the gymnasium vector-env convention: ``step`` auto-resets finished
envs and returns the fresh obs; the final obs/info of the ended episode is
in ``info["final_obs"]`` / ``info["final_info"]``.

CC BY-NC-ND 4.0 license. Built by BuiltNotTaught.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .env import GameBoyEnv

__all__ = ["VecGameBoyEnv"]


class VecGameBoyEnv:
    """``VecGameBoyEnv(16, "game.gb")`` — 16 parallel environments.

    ``workers`` defaults to one thread per env (capped at cpu count);
    constructor kwargs are forwarded to every ``GameBoyEnv``.
    """

    def __init__(self, n: int, rom_path: str | None = None,
                 config: str | None = None, workers: int | None = None,
                 **overrides):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.envs = [GameBoyEnv(rom_path, config, **overrides)
                     for _ in range(n)]
        if workers is None:
            workers = min(n, os.cpu_count() or 1)
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="rlgb-vec")
        # one task per worker, each stepping a fixed slice of lanes: n
        # submissions per step would cost more than the emulation itself
        self._slices = [range(i, n, workers) for i in range(min(workers, n))]
        e = self.envs[0]
        self.actions = e.actions
        self.action_space = e.action_space
        self.observation_space = e.observation_space

    # -------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs0, info0 = self.envs[0].reset()
        self._obs = np.empty((self.n,) + obs0.shape, dtype=obs0.dtype)
        self._rew = np.empty(self.n, dtype=np.float32)
        self._term = np.empty(self.n, dtype=bool)
        self._trunc = np.empty(self.n, dtype=bool)
        self._infos = [None] * self.n
        self._obs[0] = obs0
        self._infos[0] = info0

        def _reset_slice(lanes):
            for i in lanes:
                if i:
                    self._obs[i], self._infos[i] = self.envs[i].reset()
        list(self._pool.map(_reset_slice, self._slices))
        return self._obs.copy(), list(self._infos)

    def _step_one(self, i: int, action: int):
        obs, reward, terminated, truncated, info = self.envs[i].step(action)
        if terminated or truncated:                 # auto-reset, gym-style
            info = dict(info or {})
            info["final_obs"] = obs
            info["final_info"] = info.get("final_info")
            obs, _ = self.envs[i].reset()
        return obs, reward, terminated, truncated, info

    def _step_slice(self, lanes, actions):
        for i in lanes:
            (self._obs[i], self._rew[i], self._term[i], self._trunc[i],
             self._infos[i]) = self._step_one(i, int(actions[i]))

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (self.n,):
            raise ValueError(f"actions must have shape ({self.n},)")
        list(self._pool.map(self._step_slice, self._slices,
                            [actions] * len(self._slices)))
        return (self._obs.copy(), self._rew.copy(), self._term.copy(),
                self._trunc.copy(), list(self._infos))

    def close(self):
        self._pool.shutdown(wait=True)
        for e in self.envs:
            e.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
