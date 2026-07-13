"""
GPU inference worker (runs in the main process on the CUDA device).

Design
------
  - Scans all n_envs flags every MAX_BATCH_WAIT_MS milliseconds.
  - Accumulates env indices whose flag == FLAG_OBS_READY into a batch.
  - Once wait expires (or all envs are ready), runs one batched forward pass.
  - Writes one action per env, sets flag = FLAG_ACT_READY.

Two CUDA streams
  - stream_xfer : host → device memory copies  (pinned → GPU tensor)
  - stream_infer: forward pass

Ordering:
  1. stream_xfer copies obs from pinned host buffer to GPU tensor
  2. stream_infer waits on stream_xfer (event.record / event.wait)
  3. forward pass runs on stream_infer
  4. logits synchronised back to CPU (stream_infer.synchronize)
  5. argmax on CPU → write to shm

cuDNN benchmark mode is set once at import time.
GTX 1080 (SM 6.1): no tensor cores, no CUDA graphs — FP32 only.
Pinned memory: page-locked allocation for async DMA (no extra kernel copy).

The GPU worker is driven synchronously from GBVecEnv.step_wait().
It is NOT a thread or process — it runs in-line so we avoid GIL issues
with PyTorch CUDA and keep the policy object in main process memory.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from rlgb_vec.config import GBVecConfig
from rlgb_vec.shm import ShmSlot, FLAG_OBS_READY, FLAG_ACT_READY

# set cuDNN benchmark once at import — never per-step
torch.backends.cudnn.benchmark = True


class GPUWorker:
    """
    Drives batched inference for all env slots.

    Parameters
    ----------
    cfg         : GBVecConfig
    slots       : flat list of ShmSlot, len == n_envs
    policy_fn   : callable(obs_dict) → Tensor(n, 8) logits or action probs
                  obs_dict keys: "screen" (n, 4, 84, 84) float32, "ram" (n, 18) float32
    lstm_state  : (hx, cx) tensors on GPU, updated in-place each call (may be None if MLP)
    """

    def __init__(
        self,
        cfg: GBVecConfig,
        slots: List[ShmSlot],
        policy_fn: Callable,
        lstm_state: Optional[Tuple] = None,
    ):
        self.cfg    = cfg
        self.slots  = slots
        self.n_envs = cfg.n_envs
        self._policy_fn   = policy_fn
        self._lstm_state  = lstm_state

        want = cfg.device
        if want == "auto" or (want.startswith("cuda") and not torch.cuda.is_available()):
            want = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(want)
        self._dev = dev

        # two CUDA streams (fall back to CPU-stream stub if no CUDA)
        if dev.type == "cuda":
            self._stream_xfer  = torch.cuda.Stream(device=dev)
            self._stream_infer = torch.cuda.Stream(device=dev)
            self._xfer_event   = torch.cuda.Event()
        else:
            self._stream_xfer  = None
            self._stream_infer = None
            self._xfer_event   = None

        n  = self.n_envs
        fs, h, w = cfg.screen_shape
        rs       = cfg.ram_size

        # pre-allocated pinned host buffers
        if dev.type == "cuda":
            self._pin_screen = torch.zeros(n, fs, h, w, dtype=torch.uint8).pin_memory()
            self._pin_ram    = torch.zeros(n, rs,       dtype=torch.uint8).pin_memory()
        else:
            self._pin_screen = torch.zeros(n, fs, h, w, dtype=torch.uint8)
            self._pin_ram    = torch.zeros(n, rs,       dtype=torch.uint8)

        # pre-allocated GPU tensor pool
        self._gpu_screen = torch.zeros(n, fs, h, w, dtype=torch.float32, device=dev)
        self._gpu_ram    = torch.zeros(n, rs,       dtype=torch.float32, device=dev)

        # numpy views into pinned buffers (zero-copy fill from shm)
        self._np_screen = self._pin_screen.numpy()   # (n, fs, h, w)
        self._np_ram    = self._pin_ram.numpy()       # (n, rs)

        # per-env LSTM episode done mask (for hidden state reset)
        self._episode_done = np.zeros(n, dtype=bool)

    # ── main entry: collect a batch, infer, dispatch ──────────────────────────

    def step(self, episode_dones: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Collect obs from all FLAG_OBS_READY envs, run inference, write actions.
        Returns array(n_envs,) of int actions (-1 for envs not yet ready).
        Blocks until all n_envs have received actions.
        """
        if episode_dones is not None:
            self._episode_done = episode_dones

        n      = self.n_envs
        ready  = np.zeros(n, dtype=bool)
        actions = np.full(n, -1, dtype=np.int32)

        deadline = time.perf_counter() + self.cfg.max_batch_wait_ms * 1e-3

        # accumulate until all ready or deadline passes
        while not np.all(ready):
            for i, slot in enumerate(self.slots):
                if not ready[i] and slot.get_flag() == FLAG_OBS_READY:
                    ready[i] = True
            if time.perf_counter() >= deadline:
                break

        if not np.any(ready):
            return actions

        batch_idx = np.where(ready)[0]
        batch_n   = len(batch_idx)

        # fill pinned host buffers (zero-copy from shm numpy arrays)
        for bi, ei in enumerate(batch_idx):
            screen_view = self._np_screen[bi]
            ram_view    = self._np_ram[bi]
            self.slots[ei].read_obs_into(screen_view, ram_view)

        # transfer + inference on CUDA streams
        if self._dev.type == "cuda":
            with torch.cuda.stream(self._stream_xfer):
                self._gpu_screen[:batch_n].copy_(
                    self._pin_screen[:batch_n].to(dtype=torch.float32).div_(255.0),
                    non_blocking=True,
                )
                self._gpu_ram[:batch_n].copy_(
                    self._pin_ram[:batch_n].to(dtype=torch.float32),
                    non_blocking=True,
                )
                self._xfer_event.record(self._stream_xfer)

            with torch.cuda.stream(self._stream_infer):
                self._stream_infer.wait_event(self._xfer_event)
                obs_dict = {
                    "screen": self._gpu_screen[:batch_n],
                    "ram":    self._gpu_ram[:batch_n],
                }
                logits = self._policy_fn(obs_dict, self._lstm_state, batch_idx)
                self._stream_infer.synchronize()
        else:
            # CPU fallback (testing / no GPU)
            obs_dict = {
                "screen": self._pin_screen[:batch_n].float().div_(255.0),
                "ram":    self._pin_ram[:batch_n].float(),
            }
            logits = self._policy_fn(obs_dict, self._lstm_state, batch_idx)

        # argmax on CPU → write actions
        if isinstance(logits, torch.Tensor):
            acts_cpu = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
        else:
            acts_cpu = np.array(logits, dtype=np.int32)

        for bi, ei in enumerate(batch_idx):
            actions[ei] = int(acts_cpu[bi])
            self.slots[ei].write_action(actions[ei])
            self.slots[ei].set_flag(FLAG_ACT_READY)

        return actions

    def reset_lstm(self, env_indices: Optional[np.ndarray] = None):
        """Zero out LSTM hidden state for done envs (or all if None)."""
        if self._lstm_state is None:
            return
        hx, cx = self._lstm_state
        if env_indices is None:
            hx.zero_()
            cx.zero_()
        else:
            hx[:, env_indices, :] = 0.0
            cx[:, env_indices, :] = 0.0
