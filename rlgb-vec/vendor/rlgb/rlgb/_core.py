"""ctypes bindings to libgb.so — the rlgb C core.

CC BY-NC-ND 4.0 license. Built by BuiltNotTaught.
"""
from __future__ import annotations

import ctypes
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# CPU register ids (mirror gb.h)
REG_A, REG_F, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L = range(8)
REG_SP, REG_PC, REG_AF, REG_BC, REG_DE, REG_HL, REG_IME, REG_HALTED = range(8, 16)

# Button bits (mirror gb.h)
BUTTONS = {
    "a": 0x01, "b": 0x02, "select": 0x04, "start": 0x08,
    "right": 0x10, "left": 0x20, "up": 0x40, "down": 0x80,
}

SCREEN_W, SCREEN_H = 160, 144


def load_library(path: str | None = None) -> ctypes.CDLL:
    """Load libgb.so and declare every C symbol's signature."""
    lib = ctypes.CDLL(path or os.path.join(_HERE, "libgb.so"))
    p, u8, u16, u32, u64 = (ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint16,
                            ctypes.c_uint32, ctypes.c_uint64)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    sig = {
        "gb_new": ([], p),
        "gb_free": ([p], None),
        "gb_load_rom": ([p, ctypes.c_char_p, u32], ctypes.c_int),
        "gb_reset": ([p], None),
        "gb_step": ([p], ctypes.c_int),
        "gb_run_frame": ([p], u32),
        "gb_run_frames": ([p, u32], u32),
        "gb_cycles": ([p], u64),
        "gb_frames": ([p], u32),
        "gb_set_buttons": ([p, u8], None),
        "gb_get_buttons": ([p], u8),
        "gb_read": ([p, u16], u8),
        "gb_write": ([p, u16, u8], None),
        "gb_ptr_vram": ([p], u8p),
        "gb_ptr_wram": ([p], u8p),
        "gb_ptr_oam": ([p], u8p),
        "gb_ptr_hram": ([p], u8p),
        "gb_ptr_io": ([p], u8p),
        "gb_ptr_cartram": ([p], u8p),
        "gb_ptr_framebuffer": ([p], u8p),
        "gb_cartram_size": ([p], u32),
        "gb_get_reg": ([p, ctypes.c_int], u32),
        "gb_set_reg": ([p, ctypes.c_int, u32], None),
        "gb_set_render": ([p, ctypes.c_int], None),
        "gb_state_size": ([], u32),
        "gb_save_state": ([p, ctypes.c_char_p], None),
        "gb_load_state": ([p, ctypes.c_char_p, u32], ctypes.c_int),
        "gb_serial_read": ([p, ctypes.c_char_p, u32], u32),
        "gb_set_timing": ([p, u16], None),
        "gb_get_timing": ([p], u16),
        "gb_set_ppu_state": ([p, u8, u16, u8, u8], None),
        "gb_get_ppu_state": ([p], u32),
        "gb_set_counters": ([p, u64, u32], None),
        "gb_set_mbc_state": ([p, u16, u8, u8, u8], None),
        "gb_get_mbc_state": ([p], u32),
        "gb_set_rtc_state": ([p, ctypes.c_char_p, ctypes.c_char_p], None),
        "gb_get_rtc_state": ([p, ctypes.c_char_p], None),
        "gb_cart_type": ([p], u8),
        "gb_rom_banks": ([p], u32),
    }
    for name, (argtypes, restype) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return lib
