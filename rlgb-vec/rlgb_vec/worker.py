"""
Env worker process.

One process owns envs_per_worker rl-emu instances and drives the IPC loop:

    while alive:
        wait until flag == FLAG_ACT_READY
        action = slot.read_action()
        step emulator
        reward, done = compute_reward()
        if done: reset emulator
        write obs, reward, done into slot
        flag = FLAG_OBS_READY

The GPU worker (gpu_worker.py) does the complementary half:
        wait until flag == FLAG_OBS_READY
        read obs, run inference, write action
        flag = FLAG_ACT_READY

Poll interval: 50 µs.  At 12k FPS each emulator step takes ~83 µs, so
worst-case extra latency from polling is ~0.6% — acceptable.
"""

import ctypes
import pathlib
import signal
import time
from typing import List, Optional

import numpy as np

from rlgb_vec.config import GBVecConfig, TELEMETRY_FIELDS
from rlgb_vec.shm import ShmSlot, FLAG_OBS_READY, FLAG_ACT_READY
from rlgb_vec.gb.frame import FrameDecoder, FrameStack
from rlgb_vec.gb.ram_obs import RAMObsReader, _WRAM_BASE
from rlgb_vec.gb.blackout import BlackoutDetector
from rlgb_vec.gb.save_state import make_state_buf, save_state, load_state

# WRAM byte offsets relative to 0xC000 base
# (addresses per the datacrystal Pokémon Red/Blue RAM map)
_BADGES     = 0xD356 - _WRAM_BASE
_MAP_ID     = 0xD35E - _WRAM_BASE
_PLAYER_X   = 0xD362 - _WRAM_BASE
_PLAYER_Y   = 0xD361 - _WRAM_BASE
_PARTY_LVL  = 0xD18C - _WRAM_BASE
_PARTY_SIZE = 0xD163 - _WRAM_BASE
_HOF_FLAG   = 0xD5A0 - _WRAM_BASE   # Hall of Fame flag
# per-party-slot arrays (6 slots, stride 0x2C)
_PARTY_LVLS = [a - _WRAM_BASE for a in (0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268)]
_PARTY_HP   = [a - _WRAM_BASE for a in (0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248)]
_PARTY_MAXHP= [a - _WRAM_BASE for a in (0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269)]
# event-flag block (story progress)
_EVENTS_LO  = 0xD747 - _WRAM_BASE
_EVENTS_HI  = 0xD886 - _WRAM_BASE

_POLL_SLEEP = 50e-6   # 50 µs between flag-scan loops


class _EmuEnv:
    """One rl-emu instance plus all per-env state."""

    def __init__(
        self,
        lib,
        cfg: GBVecConfig,
        shm_name: str,
        init_state: Optional[bytes],
    ):
        self._lib  = lib
        self._cfg  = cfg
        self._slot = ShmSlot(cfg, shm_name)

        # load ROM bytes once — emu_create(rom_buf, rom_len)
        rom_bytes = pathlib.Path(cfg.rom_path).read_bytes()
        self._rom_buf = ctypes.create_string_buffer(rom_bytes)
        self._emu = lib.emu_create(self._rom_buf, len(rom_bytes))
        if not self._emu:
            raise RuntimeError(f"emu_create failed for {cfg.rom_path}")

        # pre-allocated state buffer
        self._state_buf = make_state_buf(lib)

        # if a curriculum state was provided, load it now; else step one frame
        # and snapshot so reset() always restores to the same starting point
        if init_state is not None:
            ctypes.memmove(self._state_buf, init_state, len(init_state))
            lib.emu_load_state(self._emu, self._state_buf)
        else:
            lib.emu_step_frame(self._emu)
            lib.emu_save_state(self._emu, self._state_buf)

        self._init_state: bytes = bytes(self._state_buf)   # locked snapshot

        # helper objects
        self._fdec    = FrameDecoder(cfg)
        self._fstack  = FrameStack(cfg)
        self._ram_obs = RAMObsReader()
        self._bkout   = BlackoutDetector(cfg)

        # per-episode reward tracking
        self._prev_badges  = 0
        self._prev_map_id  = -1
        self._prev_level   = 0
        self._visited_tiles: set = set()
        self._ep_maps: set       = set()
        self._map_visit_log: list = []
        self._episode_steps      = 0
        self._wram               = np.zeros(0x2000, dtype=np.uint8)

        self._reset()

    # ── WRAM helpers ──────────────────────────────────────────────────────────

    def _refresh_wram(self):
        buf = (ctypes.c_uint8 * 0x2000)()
        self._lib.emu_export_wram(self._emu, buf)
        self._wram = np.frombuffer(buf, dtype=np.uint8).copy()

    def _wb(self, offset: int) -> int:
        return int(self._wram[offset])

    # ── reset ─────────────────────────────────────────────────────────────────

    def _reset(self):
        load_state(self._lib, self._emu, self._state_buf, self._init_state)

        self._refresh_wram()
        self._prev_badges  = self._wb(_BADGES)
        self._prev_map_id  = self._wb(_MAP_ID)
        self._prev_level   = self._wb(_PARTY_LVL)
        self._visited_tiles   = set()
        self._ep_maps         = {self._prev_map_id}
        self._map_visit_log   = [(0, self._prev_map_id)]
        self._episode_steps   = 0
        # per-episode reward-component accumulators (for the live monitor)
        self._rc = {"badge": 0.0, "explore": 0.0, "level": 0.0, "step": 0.0}
        self._rc_prev_total = 0.0
        self._bkout.reset()

        self._fstack.reset()
        for _ in range(self._cfg.screen_shape[0]):
            self._fstack.push(self._fdec.read(self._lib, self._emu))

    # ── reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        cfg = self._cfg
        self._refresh_wram()
        rc  = self._rc                       # per-episode component accumulators
        rc["step"] += cfg.penalty_per_step

        badges = self._wb(_BADGES)
        if badges != self._prev_badges:
            gained = bin(badges & ~self._prev_badges).count("1")
            rc["badge"] += gained * cfg.reward_badge
            if badges == 0xFF:
                rc["badge"] += cfg.reward_champion
            self._prev_badges = badges

        map_id = self._wb(_MAP_ID)
        x      = self._wb(_PLAYER_X)
        y      = self._wb(_PLAYER_Y)

        if map_id != self._prev_map_id:
            rc["explore"] += cfg.reward_new_map
            self._prev_map_id = map_id
            self._ep_maps.add(map_id)
            self._map_visit_log.append((self._episode_steps, map_id))
            cutoff = self._episode_steps - cfg.map_revisit_window
            self._map_visit_log = [e for e in self._map_visit_log if e[0] >= cutoff]
            visits = sum(1 for _, mid in self._map_visit_log if mid == map_id)
            if visits > cfg.map_revisit_limit:
                rc["explore"] += cfg.penalty_map_revisit

        tile = (map_id, x, y)
        if tile not in self._visited_tiles:
            self._visited_tiles.add(tile)
            rc["explore"] += cfg.reward_new_tile

        level = self._wb(_PARTY_LVL)
        if level > self._prev_level:
            rc["level"] += (level - self._prev_level) * cfg.reward_level_up
            self._prev_level = level

        # this step's reward = change in the component totals; simplest to
        # recompute from the accumulators by tracking the previous sum
        total = rc["badge"] + rc["explore"] + rc["level"] + rc["step"]
        step_reward = total - getattr(self, "_rc_prev_total", 0.0)
        self._rc_prev_total = total
        return float(step_reward)

    # ── telemetry ───────────────────────────────────────────────────────────────

    def _read_hp(self, off: int) -> int:
        return 256 * int(self._wram[off]) + int(self._wram[off + 1])

    def _hp_fraction(self) -> float:
        cur = sum(self._read_hp(o) for o in _PARTY_HP)
        mx  = sum(self._read_hp(o) for o in _PARTY_MAXHP)
        return cur / mx if mx > 0 else 0.0

    def _level_sum(self) -> int:
        n = min(self._wb(_PARTY_SIZE), 6)
        return sum(int(self._wram[_PARTY_LVLS[i]]) for i in range(n))

    def _events(self) -> int:
        block = self._wram[_EVENTS_LO:_EVENTS_HI]
        return int(np.unpackbits(block).sum())

    def _telemetry(self) -> np.ndarray:
        """Per-env live stats in TELEMETRY_FIELDS order (float32).
        Reads self._wram, refreshed by the preceding _compute_reward()."""
        rc = self._rc
        stats = {
            "badges":         bin(self._prev_badges).count("1"),
            "level_sum":      self._level_sum(),
            "party_size":     min(self._wb(_PARTY_SIZE), 6),
            "hp_frac":        self._hp_fraction(),
            "events":         self._events(),
            "maps_explored":  len(self._ep_maps),
            "tiles_explored": len(self._visited_tiles),
            "r_badge":        rc["badge"],
            "r_explore":      rc["explore"],
            "r_level":        rc["level"],
            "r_step":         rc["step"],
        }
        return np.array([stats[f] for f in TELEMETRY_FIELDS], dtype=np.float32)

    # ── done ─────────────────────────────────────────────────────────────────

    def _check_done(self, frame: np.ndarray) -> bool:
        if self._episode_steps >= self._cfg.max_episode_steps:
            return True
        if self._wb(_HOF_FLAG) & 0x01:
            return True
        return self._bkout.check(frame)

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: int):
        from rlgb_vec.bindings import ACTIONS
        self._lib.emu_set_joypad(self._emu, ACTIONS[action])
        self._lib.emu_step_frames(self._emu, self._cfg.frame_skip)
        self._episode_steps += 1

        frame = self._fdec.read(self._lib, self._emu)
        self._fstack.push(frame)

        reward = self._compute_reward()
        done   = self._check_done(frame)

        # write obs + result + telemetry before (possibly) resetting, so the
        # telemetry reflects this episode's final state on a done step
        ram_obs = self._ram_obs.read(self._lib, self._emu)
        self._slot.write_obs(self._fstack.stack, ram_obs)
        self._slot.write_result(reward, done)
        self._slot.write_telemetry(self._telemetry())

        if done:
            self._reset()

    def write_initial_obs(self):
        """Write first obs right after construction so GPU worker sees it."""
        ram_obs = self._ram_obs.read(self._lib, self._emu)
        self._slot.write_obs(self._fstack.stack, ram_obs)
        self._slot.write_result(0.0, False)
        self._slot.write_telemetry(self._telemetry())

    def close(self):
        self._slot.close()
        try:
            self._lib.emu_destroy(self._emu)
        except Exception:
            pass


# ── Worker entry point ─────────────────────────────────────────────────────────

def worker_main(
    cfg: GBVecConfig,
    shm_names: List[str],
    init_states: Optional[List[bytes]],
    ready_event,
    stop_event,
):
    """
    Entry point for a worker process.

    shm_names    — one name per env slot (len == envs_per_worker)
    init_states  — curriculum state bytes per slot, or None for default boot
    ready_event  — mp.Event set when all envs loaded and first obs written
    stop_event   — mp.Event set by parent to shut down
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    from rlgb_vec.bindings import get_lib
    lib = get_lib()

    envs: List[_EmuEnv] = []
    for i, name in enumerate(shm_names):
        st = init_states[i] if init_states is not None else None
        e  = _EmuEnv(lib, cfg, name, st)
        e.write_initial_obs()
        e._slot.set_flag(FLAG_OBS_READY)
        envs.append(e)

    ready_event.set()

    while not stop_event.is_set():
        acted = False
        for env in envs:
            if env._slot.get_flag() == FLAG_ACT_READY:
                action = env._slot.read_action()
                env.step(action)
                env._slot.set_flag(FLAG_OBS_READY)
                acted = True
        if not acted:
            time.sleep(_POLL_SLEEP)

    for env in envs:
        env.close()
