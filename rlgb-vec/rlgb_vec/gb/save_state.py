"""
Save-state helpers that match the real rl-emu API.

emu_state_size() → c_size_t (no args)
emu_save_state(emu, buf)  — writes into pre-allocated buffer
emu_load_state(emu, buf)  — reads from pre-allocated buffer

We keep a bytes snapshot per env and restore on reset.
"""

import ctypes
from typing import Any


def make_state_buf(lib: Any) -> ctypes.Array:
    """Allocate a state buffer sized to match the emulator's EmuState."""
    size = int(lib.emu_state_size())
    return ctypes.create_string_buffer(size)


def save_state(lib: Any, emu: Any, buf: ctypes.Array) -> bytes:
    """Write current emulator state into buf and return a bytes snapshot."""
    lib.emu_save_state(emu, buf)
    return bytes(buf)


def load_state(lib: Any, emu: Any, buf: ctypes.Array, state: bytes) -> None:
    """Restore emulator state from a bytes snapshot via the pre-allocated buf."""
    ctypes.memmove(buf, state, len(state))
    lib.emu_load_state(emu, buf)
