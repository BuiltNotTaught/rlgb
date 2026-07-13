"""Binding layer: presents the async-worker's `emu_*` calls over the vendored
rlgb core (`libgb.so`, `gb_*` API).

The worker/frame/ram_obs code is written against a small `emu_*` surface
(create, destroy, state size/save/load, step, joypad, wram export, ranged
read). This module implements that surface on rlgb's `gb_*` symbols so the
rest of the package is emulator-agnostic — swapping the vendored core is the
only change needed to target a sibling emulator.

The vendored emulator lives at ``<repo>/vendor/rlgb`` and is put on sys.path
here so both the barebone (``pip install -e .``) and Docker paths import the
same core without an extra install step.
"""
import ctypes
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(os.path.dirname(_HERE), "vendor", "rlgb")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from rlgb import _core  # noqa: E402  (vendored core)

# Discrete(8): down, left, right, up, a, b, start, noop  → rlgb button masks
_B = _core.BUTTONS
ACTIONS = [_B["down"], _B["left"], _B["right"], _B["up"],
           _B["a"], _B["b"], _B["start"], 0]

_WRAM_SIZE = 0x2000
_VRAM_SIZE = 0x2000
_IO_SIZE = 0x80

_lib_singleton = None


class _CompatLib:
    """`emu_*` methods delegating to the vendored rlgb `libgb.so`."""

    def __init__(self):
        self._lib = _core.load_library(None)

    def emu_create(self, rom_buf, rom_len):
        g = self._lib.gb_new()
        if not g:
            return None
        rom = bytes(rom_buf[:rom_len]) if not isinstance(rom_buf, bytes) else rom_buf
        if self._lib.gb_load_rom(g, rom, rom_len) != 0:
            self._lib.gb_free(g)
            return None
        self._lib.gb_set_render(g, 0)   # we decode VRAM ourselves
        return g

    def emu_destroy(self, emu):
        self._lib.gb_free(emu)

    def emu_state_size(self):
        return self._lib.gb_state_size()

    def emu_save_state(self, emu, buf):
        self._lib.gb_save_state(emu, buf)

    def emu_load_state(self, emu, buf):
        n = self._lib.gb_state_size()
        payload = bytes(buf[:n]) if not isinstance(buf, bytes) else buf
        if self._lib.gb_load_state(emu, payload, n) != 0:
            raise ValueError("gb_load_state failed (size mismatch)")

    def emu_step_frame(self, emu):
        self._lib.gb_run_frames(emu, 1)

    def emu_step_frames(self, emu, n):
        self._lib.gb_run_frames(emu, n)

    def emu_set_joypad(self, emu, mask):
        self._lib.gb_set_buttons(emu, mask)

    def emu_export_wram(self, emu, buf):
        ctypes.memmove(buf, self._lib.gb_ptr_wram(emu), _WRAM_SIZE)

    def emu_read_range(self, emu, addr, length, buf):
        if addr == 0x8000 and length <= _VRAM_SIZE:        # VRAM
            ctypes.memmove(buf, self._lib.gb_ptr_vram(emu), length)
        elif addr == 0xC000 and length <= _WRAM_SIZE:      # WRAM
            ctypes.memmove(buf, self._lib.gb_ptr_wram(emu), length)
        elif addr == 0xFF00 and length <= _IO_SIZE:        # IO
            ctypes.memmove(buf, self._lib.gb_ptr_io(emu), length)
        else:                                              # generic fallback
            for i in range(length):
                buf[i] = self._lib.gb_read(emu, (addr + i) & 0xFFFF)


def get_lib():
    global _lib_singleton
    if _lib_singleton is None:
        _lib_singleton = _CompatLib()
    return _lib_singleton


_get_lib = get_lib   # backward-compat alias
