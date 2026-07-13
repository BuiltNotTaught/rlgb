"""
Shared memory layout for emu-vec async IPC.

One contiguous shared memory block per env slot.

Layout (bytes, env index i):
    [0]              flag          uint8   0=idle 1=obs_ready 2=act_ready
    [1..obs_bytes]   obs           uint8   screen_bytes + ram_bytes flat
    [obs_bytes+1]    action        uint8   Discrete index 0-7
    [obs_bytes+2..] reward+done   float32+uint8 (reward=4B, done=1B)
    [.. telemetry]   telemetry     float32[len(TELEMETRY_FIELDS)]  live stats

All offsets computed from GBVecConfig so they are consistent across
the writer (env worker) and the reader (GPU worker / VecEnv).

Flag protocol
-------------
  Env worker   → writes obs → sets flag = FLAG_OBS_READY
  GPU worker   → reads obs, runs inference, writes action → sets flag = FLAG_ACT_READY
  Env worker   → reads action, steps emulator → back to FLAG_IDLE

The GPU worker never writes to the obs region.
The env worker never writes to the action byte while flag == FLAG_ACT_READY.
No mutex required: single-writer per region, flag is the memory fence.
"""

import ctypes
from multiprocessing.shared_memory import SharedMemory
from typing import List, Tuple

import numpy as np

from rlgb_vec.config import GBVecConfig

# Flag values
FLAG_IDLE      = np.uint8(0)
FLAG_OBS_READY = np.uint8(1)
FLAG_ACT_READY = np.uint8(2)

# Byte widths
_REWARD_BYTES = 4   # float32
_DONE_BYTES   = 1   # uint8  (0 or 1)


def _slot_size(cfg: GBVecConfig) -> int:
    """Total bytes for one env slot in shared memory."""
    return (
        1                      # flag
        + cfg.obs_total_bytes  # obs (screen + ram)
        + 1                    # action
        + _REWARD_BYTES        # reward
        + _DONE_BYTES          # done
        + cfg.telemetry_bytes  # live per-env telemetry (float32[])
    )


def _offsets(cfg: GBVecConfig) -> dict:
    """Byte offsets within a single slot."""
    flag   = 0
    obs    = 1
    action = 1 + cfg.obs_total_bytes
    reward = action + 1
    done   = reward + _REWARD_BYTES
    telem  = done + _DONE_BYTES
    return {"flag": flag, "obs": obs, "action": action,
            "reward": reward, "done": done, "telemetry": telem}


class ShmPool:
    """
    Owner of the shared memory blocks.  Created once by GBVecEnv.__init__,
    then passed (by name) to worker processes.

    Each env gets its own SharedMemory block of size _slot_size(cfg).
    Workers attach by name.
    """

    def __init__(self, cfg: GBVecConfig):
        self.cfg   = cfg
        self._size = _slot_size(cfg)
        self._offs = _offsets(cfg)
        self._shms: List[SharedMemory] = []
        self.names: List[str] = []

        for i in range(cfg.n_envs):
            shm = SharedMemory(create=True, size=self._size)
            # zero-initialise
            buf = np.frombuffer(shm.buf, dtype=np.uint8)
            buf[:] = 0
            self._shms.append(shm)
            self.names.append(shm.name)

    def close(self):
        for shm in self._shms:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


class ShmSlot:
    """
    Lightweight view into one env's shared memory slot.
    Created by worker processes (attach by name).
    """

    def __init__(self, cfg: GBVecConfig, shm_name: str):
        self.cfg  = cfg
        self._shm = SharedMemory(name=shm_name, create=False)
        self._buf = np.frombuffer(self._shm.buf, dtype=np.uint8)
        self._offs = _offsets(cfg)
        self._size = _slot_size(cfg)

    # ── flag ──────────────────────────────────────────────────────────────────

    def get_flag(self) -> int:
        return int(self._buf[self._offs["flag"]])

    def set_flag(self, value: int):
        self._buf[self._offs["flag"]] = value

    # ── obs ───────────────────────────────────────────────────────────────────

    def write_obs(self, screen: np.ndarray, ram: np.ndarray):
        """Copy screen + ram into the obs region."""
        start = self._offs["obs"]
        sb    = self.cfg.obs_screen_bytes
        rb    = self.cfg.ram_size
        self._buf[start          : start + sb]      = screen.ravel()
        self._buf[start + sb     : start + sb + rb] = ram.ravel()

    def read_obs_into(self, screen_out: np.ndarray, ram_out: np.ndarray):
        """Zero-copy read: fill caller's pre-allocated arrays."""
        start = self._offs["obs"]
        sb    = self.cfg.obs_screen_bytes
        rb    = self.cfg.ram_size
        screen_out.ravel()[:] = self._buf[start      : start + sb]
        ram_out.ravel()[:]    = self._buf[start + sb  : start + sb + rb]

    # ── action ────────────────────────────────────────────────────────────────

    def write_action(self, action: int):
        self._buf[self._offs["action"]] = action

    def read_action(self) -> int:
        return int(self._buf[self._offs["action"]])

    # ── reward / done ─────────────────────────────────────────────────────────

    def write_result(self, reward: float, done: bool):
        off = self._offs["reward"]
        # float32 into 4 bytes
        ctypes.memmove(
            ctypes.addressof(ctypes.c_uint8.from_buffer(self._shm.buf, off)),
            ctypes.cast(ctypes.pointer(ctypes.c_float(reward)), ctypes.c_void_p),
            4,
        )
        self._buf[self._offs["done"]] = int(done)

    def read_result(self) -> Tuple[float, bool]:
        off = self._offs["reward"]
        raw = bytes(self._shm.buf[off : off + 4])
        reward = ctypes.cast(ctypes.c_char_p(raw), ctypes.POINTER(ctypes.c_float)).contents.value
        done   = bool(self._buf[self._offs["done"]])
        return reward, done

    # ── telemetry ───────────────────────────────────────────────────────────────

    def write_telemetry(self, values: np.ndarray):
        """Store the per-env telemetry vector (float32, len == TELEMETRY_FIELDS)."""
        off = self._offs["telemetry"]
        nb  = self.cfg.telemetry_bytes
        self._buf[off : off + nb] = np.ascontiguousarray(
            values, dtype=np.float32).view(np.uint8)

    def read_telemetry(self) -> np.ndarray:
        """Return a copy of the per-env telemetry vector (float32)."""
        off = self._offs["telemetry"]
        nb  = self.cfg.telemetry_bytes
        return self._buf[off : off + nb].copy().view(np.float32)

    def close(self):
        try:
            self._buf = None   # release numpy view before closing shm
            self._shm.close()
        except Exception:
            pass
