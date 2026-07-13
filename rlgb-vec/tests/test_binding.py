"""End-to-end smoke test: the emu_* binding drives the vendored rlgb core,
and the DMG VRAM/RAM decode paths produce sane obs. Runs barebone (numpy +
ctypes only); cv2/torch not required. Skips if the .so or a ROM is absent.
"""
import ctypes
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SO = os.path.join(ROOT, "vendor", "rlgb", "rlgb", "libgb.so")

# ROM discovery (portable): $RLGB_VEC_ROM / $ROM env override, else the vendored
# freely-distributable test ROM. No absolute host paths.
_ROM = (os.environ.get("RLGB_VEC_ROM") or os.environ.get("ROM")
        or os.path.join(ROOT, "vendor", "rlgb", "roms", "cpu_instrs.gb"))
ROM = _ROM if os.path.exists(_ROM) else None

pytestmark = pytest.mark.skipif(
    not os.path.exists(SO), reason="libgb.so not built (run ./build.sh)"
)


def _lib():
    from rlgb_vec.bindings import get_lib
    return get_lib()


def test_create_step_state_roundtrip():
    lib = _lib()
    if ROM is None:
        pytest.skip("no test ROM available")
    with open(ROM, "rb") as f:
        rom = f.read()
    emu = lib.emu_create(rom, len(rom))
    assert emu

    size = lib.emu_state_size()
    assert size > 0
    buf = ctypes.create_string_buffer(size)

    lib.emu_step_frames(emu, 10)
    lib.emu_save_state(emu, buf)
    snap = bytes(buf)

    lib.emu_step_frames(emu, 10)
    lib.emu_load_state(emu, buf)         # restore
    lib.emu_save_state(emu, buf)
    assert bytes(buf) == snap            # deterministic restore

    lib.emu_destroy(emu)


def test_wram_and_range_read():
    lib = _lib()
    if ROM is None:
        pytest.skip("no test ROM available")
    with open(ROM, "rb") as f:
        rom = f.read()
    emu = lib.emu_create(rom, len(rom))
    lib.emu_step_frames(emu, 30)

    wram = (ctypes.c_uint8 * 0x2000)()
    lib.emu_export_wram(emu, wram)
    assert len(bytes(wram)) == 0x2000

    vram = (ctypes.c_uint8 * 0x2000)()
    lib.emu_read_range(emu, 0x8000, 0x2000, vram)
    io = (ctypes.c_uint8 * 0x80)()
    lib.emu_read_range(emu, 0xFF00, 0x80, io)
    # generic fallback path (single byte via gb_read)
    one = (ctypes.c_uint8 * 1)()
    lib.emu_read_range(emu, 0xC000, 1, one)
    assert int(one[0]) == int(wram[0])

    lib.emu_destroy(emu)


def test_joypad_and_frame_decode():
    from rlgb_vec.bindings import ACTIONS
    from rlgb_vec.config import GBVecConfig
    from rlgb_vec.gb.frame import FrameDecoder
    lib = _lib()
    if ROM is None:
        pytest.skip("no test ROM available")
    with open(ROM, "rb") as f:
        rom = f.read()
    emu = lib.emu_create(rom, len(rom))

    for a in ACTIONS:                    # every button mask is accepted
        lib.emu_set_joypad(emu, a)
        lib.emu_step_frames(emu, 1)

    dec = FrameDecoder(GBVecConfig(rom_path=ROM))
    frame = dec.read(lib, emu)
    assert frame.shape == (84, 84)
    assert frame.dtype == np.uint8

    lib.emu_destroy(emu)
