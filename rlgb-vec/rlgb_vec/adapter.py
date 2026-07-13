"""Plug-and-play SB3 wiring.

The one-liner path — no policy wiring, no callbacks required:

    from rlgb_vec import make_env
    from stable_baselines3 import PPO

    vec   = make_env("/roms/pokemon_red.gb", n_envs=12)   # ready SB3 VecEnv
    model = PPO("MultiInputPolicy", vec)                   # Dict obs → MultiInput*
    model.learn(1_000_000)
    vec.close()

`GBVecEnv` is a standard SB3 `VecEnv`: SB3 runs the policy forward pass on the
GPU and hands actions to `vec.step(...)`, so the GPU is already the inference
engine — nothing extra to wire. `make_env` just picks a sensible worker split
for the env count.

Recurrent policies work the same way (needs `sb3-contrib`):

    from sb3_contrib import RecurrentPPO
    model = RecurrentPPO("MultiInputLstmPolicy", make_env("/roms/game.gb"))

`build_config` is the pure (torch-free) config builder, so this module imports
without torch/SB3 installed — handy for tests and introspection.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from rlgb_vec.config import GBVecConfig


def _split(n_envs: int, envs_per_worker: Optional[int]) -> Tuple[int, int]:
    """Choose (n_workers, envs_per_worker) whose product == n_envs."""
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    if envs_per_worker is None:
        # prefer 3 per worker (the tuned default), then 4/2/1
        envs_per_worker = next((k for k in (3, 4, 2, 1) if n_envs % k == 0), 1)
    if n_envs % envs_per_worker != 0:
        raise ValueError(f"n_envs={n_envs} not divisible by envs_per_worker={envs_per_worker}")
    return n_envs // envs_per_worker, envs_per_worker


def build_config(
    rom_path: str,
    n_workers: int = 4,
    envs_per_worker: int = 3,
    init_state: Optional[bytes] = None,
    init_states: Optional[List[bytes]] = None,
    device: str = "auto",
    frame_skip: int = 4,
    frame_stack: int = 4,
    screen_h: int = 84,
    screen_w: int = 84,
    max_episode_steps: int = 20_480,
    max_batch_wait_ms: float = 2.0,
    **overrides,
) -> GBVecConfig:
    """Build a GBVecConfig from flat kwargs. Pure — no torch/SB3 import."""
    if init_states is None and init_state is not None:
        init_states = [init_state] * n_workers
    return GBVecConfig(
        rom_path=rom_path,
        n_workers=n_workers,
        envs_per_worker=envs_per_worker,
        init_states=init_states,
        device=device,
        frame_skip=frame_skip,
        screen_shape=(frame_stack, screen_h, screen_w),
        max_episode_steps=max_episode_steps,
        max_batch_wait_ms=max_batch_wait_ms,
        **overrides,
    )


def make_gb_vec_env(
    rom_path: str,
    n_workers: int = 4,
    envs_per_worker: int = 3,
    **kw,
):
    """Build a ready-to-train `GBVecEnv` (SB3 VecEnv). `kw` → build_config."""
    from rlgb_vec.vec_env import GBVecEnv   # lazy: needs torch/SB3/gymnasium
    cfg = build_config(rom_path, n_workers=n_workers, envs_per_worker=envs_per_worker, **kw)
    return GBVecEnv(cfg)


def make_env(rom_path: str, n_envs: int = 12, envs_per_worker: Optional[int] = None, **kw):
    """Friendliest entry point: one ROM, one env count → a ready SB3 VecEnv.

        vec = make_env("/roms/game.gb", n_envs=12)
    """
    n_workers, epw = _split(n_envs, envs_per_worker)
    return make_gb_vec_env(rom_path, n_workers=n_workers, envs_per_worker=epw, **kw)


# ── optional / experimental ─────────────────────────────────────────────────
# The standard SB3 loop above needs NONE of the below. wire_policy() drives the
# separate async GPUWorker inference path (gpu_worker.py); it is experimental
# and off by default. Kept for explicit opt-in only.

def wire_policy(model, vec):
    """[experimental] Attach an SB3 RecurrentPPO policy to the async GPUWorker.
    Not required for normal training — GBVecEnv already works as a plain VecEnv."""
    import torch
    policy = model.policy
    policy.set_training_mode(False)
    dev = next(policy.parameters()).device
    hidden = model.policy.lstm_actor.hidden_size
    layers = model.policy.lstm_actor.num_layers
    hx = torch.zeros(layers, vec.num_envs, hidden, device=dev)
    cx = torch.zeros(layers, vec.num_envs, hidden, device=dev)
    lstm_state = (hx, cx)

    @torch.no_grad()
    def policy_fn(obs_dict, lstm_st, env_indices):
        batch_n = obs_dict["screen"].shape[0]
        starts = torch.zeros(batch_n, dtype=torch.float32, device=dev)
        lstm_batch = (lstm_st[0][:, env_indices, :], lstm_st[1][:, env_indices, :])
        actions, _, _, new_lstm = policy.forward(obs_dict, lstm_batch, starts, deterministic=True)
        lstm_st[0][:, env_indices, :] = new_lstm[0]
        lstm_st[1][:, env_indices, :] = new_lstm[1]
        return actions

    vec.set_policy(policy_fn, lstm_state)


class SB3TrainingCallback:
    """No-op SB3 callback. Safe to pass, but not needed — GBVecEnv trains as a
    plain VecEnv. Present for backward compatibility with older examples."""

    def __init__(self, vec=None):
        self._vec = vec

    def init_callback(self, model):
        self.model = model

    def on_step(self):
        return True

    def on_training_start(self, locals_, globals_):
        pass

    def on_training_end(self):
        pass

    def on_rollout_start(self):
        pass

    def on_rollout_end(self):
        pass
