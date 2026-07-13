"""
RAM observation vector via bulk emu_read_range.

18 bytes matching pokeai/env.py _OBS_ADDRS exactly.
Uses emu_export_wram for zero-copy WRAM snapshot, then index-selects.
"""

import ctypes
from typing import Any

import numpy as np

from rlgb_vec.config import GBVecConfig

# WRAM spans C000-DFFF (8 KB)
_WRAM_BASE = 0xC000
_WRAM_SIZE = 0x2000

# Must match pokeai/ram_map.py / env.py _OBS_ADDRS
_OBS_ADDRS = [
    0xD35E,  # 0  map_id
    0xD362,  # 1  player_x
    0xD361,  # 2  player_y
    0xD356,  # 3  badges
    0xD163,  # 4  party_size
    0xD164,  # 5  lead_species
    0xD16C,  # 6  hp_hi
    0xD16D,  # 7  hp_lo
    0xD18D,  # 8  maxhp_hi
    0xD18E,  # 9  maxhp_lo
    0xD18C,  # 10 level
    0xD16F,  # 11 status
    0xD057,  # 12 in_battle
    0xD05A,  # 13 battle_type
    0xCFE5,  # 14 enemy_species
    0xCFF3,  # 15 enemy_level
    0xD31D,  # 16 item_count
    0xD13B,  # 17 steps
]
_OBS_OFFSETS = np.array([a - _WRAM_BASE for a in _OBS_ADDRS], dtype=np.int32)


class RAMObsReader:
    """
    Reads the 18-byte RAM obs vector from a running rl-emu instance.
    Holds a persistent ctypes buffer to avoid per-step allocation.
    """

    def __init__(self):
        self._wram_c = (ctypes.c_uint8 * _WRAM_SIZE)()

    def read(self, lib: Any, emu: Any) -> np.ndarray:
        """Returns a freshly-copied (18,) uint8 array."""
        lib.emu_export_wram(emu, self._wram_c)
        arr = np.frombuffer(self._wram_c, dtype=np.uint8)
        return arr[_OBS_OFFSETS].copy()

    def read_addr(self, lib: Any, emu: Any, addr: int) -> int:
        """Read a single WRAM address, reusing the cached buffer."""
        lib.emu_export_wram(emu, self._wram_c)
        return int(self._wram_c[addr - _WRAM_BASE])
