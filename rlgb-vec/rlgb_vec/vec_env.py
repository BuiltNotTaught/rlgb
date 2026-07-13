"""
GBVecEnv — SB3-compatible VecEnv wrapping the async emu-vec machinery.

Drop-in replacement for SubprocVecEnv.  The public API matches SB3's
VecEnv contract so RecurrentPPO / DummyVecEnv wrappers all work.

Usage
-----
    from rlgb_vec import GBVecEnv, GBVecConfig
    cfg = GBVecConfig(rom_path="...", n_workers=4, envs_per_worker=3)
    vec = GBVecEnv(cfg)
    obs = vec.reset()
    obs, rews, dones, infos = vec.step(actions)
    vec.close()

Architecture
------------
  - n_workers  subprocess workers, each owning envs_per_worker rl-emu instances
  - n_envs = n_workers * envs_per_worker total environments
  - ShmPool holds one SharedMemory block per env (owned by THIS process)
  - Workers attach by name; GPU worker also reads from the same blocks
  - policy_fn / lstm_state injected after construction via set_policy()
"""

import multiprocessing as mp
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from rlgb_vec.config import GBVecConfig, TELEMETRY_FIELDS
from rlgb_vec.shm import ShmPool, ShmSlot, FLAG_OBS_READY, FLAG_ACT_READY
from rlgb_vec.gpu_worker import GPUWorker
from rlgb_vec.worker import worker_main


class GBVecEnv(VecEnv):
    """
    Async VecEnv bridge.

    env_fns    : ignored (present for API compat with SB3 — envs are created
                 internally by worker processes)
    cfg        : GBVecConfig
    policy_fn  : set via set_policy() before the first step.  Must be callable
                 with signature:
                     policy_fn(obs_dict, lstm_state, env_indices) → Tensor
    """

    metadata = {}

    def __init__(self, cfg: GBVecConfig, env_fns=None):
        self.cfg = cfg
        n = cfg.n_envs

        # observation / action spaces
        fs, h, w = cfg.screen_shape
        rs = cfg.ram_size
        observation_space = spaces.Dict({
            "screen": spaces.Box(0, 255, shape=(fs, h, w), dtype=np.uint8),
            "ram":    spaces.Box(0, 255, shape=(rs,),       dtype=np.uint8),
        })
        action_space = spaces.Discrete(cfg.n_actions)

        super().__init__(n, observation_space, action_space)

        # shared memory pool (owned here, workers attach by name)
        self._pool   = ShmPool(cfg)
        self._slots  = [ShmSlot(cfg, name) for name in self._pool.names]

        # GPU worker (no policy yet — inject via set_policy)
        self._gpu: Optional[GPUWorker] = None
        self._lstm_state: Optional[Tuple] = None

        # per-env episode tracking needed by VecEnv contract
        self._episode_rewards = np.zeros(n, dtype=np.float32)
        self._episode_lengths = np.zeros(n, dtype=np.int32)

        # latest obs from shm
        self._last_obs: Optional[Dict[str, np.ndarray]] = None

        # start worker processes
        self._workers: List[mp.Process] = []
        self._stop_events: List[mp.Event] = []
        self._ready_events: List[mp.Event] = []
        self._start_workers()

    # ── worker management ──────────────────────────────────────────────────────

    def _start_workers(self):
        cfg = self.cfg
        ctx = mp.get_context("spawn")

        for w in range(cfg.n_workers):
            start_env = w * cfg.envs_per_worker
            shm_names = self._pool.names[start_env : start_env + cfg.envs_per_worker]

            if cfg.init_states is not None:
                if len(cfg.init_states) == cfg.n_workers:
                    states = [cfg.init_states[w]] * cfg.envs_per_worker
                else:
                    states = cfg.init_states[start_env : start_env + cfg.envs_per_worker]
            else:
                states = None

            ready = ctx.Event()
            stop  = ctx.Event()
            p = ctx.Process(
                target=worker_main,
                args=(cfg, shm_names, states, ready, stop),
                daemon=True,
                name=f"emu-vec-worker-{w}",
            )
            p.start()
            self._workers.append(p)
            self._stop_events.append(stop)
            self._ready_events.append(ready)

        # wait for all workers to load ROMs and write first obs
        for ev in self._ready_events:
            ev.wait(timeout=120)

    # ── policy injection ───────────────────────────────────────────────────────

    def set_policy(self, policy_fn: Callable, lstm_state: Optional[Tuple] = None):
        """
        Inject the inference callable.  Must be called before step().

        policy_fn(obs_dict, lstm_state, env_indices) → Tensor(batch, n_actions)
        """
        self._gpu = GPUWorker(
            cfg=self.cfg,
            slots=self._slots,
            policy_fn=policy_fn,
            lstm_state=lstm_state,
        )
        self._lstm_state = lstm_state

    # ── VecEnv API ─────────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, np.ndarray]:
        """Block until all envs have written their first obs, return batch."""
        # workers already wrote FLAG_OBS_READY on startup; just collect
        for slot in self._slots:
            deadline = time.perf_counter() + 30.0
            while slot.get_flag() != FLAG_OBS_READY:
                if time.perf_counter() > deadline:
                    raise TimeoutError("Env worker did not produce first obs within 30s")
                time.sleep(1e-4)

        self._last_obs = self._collect_obs()
        return self._last_obs

    def step_async(self, actions: np.ndarray):
        """Write actions for all envs and set FLAG_ACT_READY."""
        for i, slot in enumerate(self._slots):
            slot.write_action(int(actions[i]))
            slot.set_flag(FLAG_ACT_READY)

    def step_wait(self) -> Tuple[Dict, np.ndarray, np.ndarray, List[Dict]]:
        """
        Wait for env workers to step and produce new obs.
        Also runs GPU inference to generate next-step actions
        (pre-loaded so workers can proceed while SB3 updates gradients).

        Returns: obs, rewards, dones, infos
        """
        n = self.cfg.n_envs
        rewards = np.zeros(n, dtype=np.float32)
        dones   = np.zeros(n, dtype=bool)

        # wait for all FLAG_OBS_READY
        deadline = time.perf_counter() + 60.0
        pending  = list(range(n))
        while pending:
            still_pending = []
            for i in pending:
                if self._slots[i].get_flag() == FLAG_OBS_READY:
                    r, d         = self._slots[i].read_result()
                    rewards[i]   = r
                    dones[i]     = d
                else:
                    still_pending.append(i)
            pending = still_pending
            if pending and time.perf_counter() > deadline:
                raise TimeoutError(f"Envs {pending} did not step within 60s")
            if pending:
                time.sleep(50e-6)

        obs   = self._collect_obs()
        infos = [{"env_idx": i} for i in range(n)]

        # per-env live telemetry (badges, level, exploration, position)
        for i in range(n):
            telem = self._slots[i].read_telemetry()
            infos[i]["telemetry"] = {
                f: float(telem[j]) for j, f in enumerate(TELEMETRY_FIELDS)
            }

        # accumulate episode stats
        self._episode_rewards += rewards
        self._episode_lengths += 1
        for i in range(n):
            if dones[i]:
                infos[i]["episode"] = {
                    "r": float(self._episode_rewards[i]),
                    "l": int(self._episode_lengths[i]),
                }
                self._episode_rewards[i] = 0.0
                self._episode_lengths[i] = 0

        self._last_obs = obs
        return obs, rewards, dones, infos

    def step(self, actions: np.ndarray):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        for ev in self._stop_events:
            ev.set()
        for w in self._workers:
            w.join(timeout=10)
            if w.is_alive():
                w.kill()
        for slot in self._slots:
            slot.close()
        self._pool.close()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _collect_obs(self) -> Dict[str, np.ndarray]:
        n = self.cfg.n_envs
        fs, h, w = self.cfg.screen_shape
        rs       = self.cfg.ram_size
        screens  = np.zeros((n, fs, h, w), dtype=np.uint8)
        rams     = np.zeros((n, rs),        dtype=np.uint8)
        for i, slot in enumerate(self._slots):
            slot.read_obs_into(screens[i], rams[i])
        return {"screen": screens, "ram": rams}

    # ── VecEnv abstract stubs ─────────────────────────────────────────────────

    def env_method(self, method_name, *method_args, **method_kwargs):
        raise NotImplementedError("env_method not supported in GBVecEnv")

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":   # SB3 >= 2.x queries this at init
            n = self.num_envs if indices is None else len(indices)
            return [None] * n
        raise NotImplementedError("get_attr not supported in GBVecEnv")

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("set_attr not supported in GBVecEnv")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def get_images(self):
        raise NotImplementedError
