"""
VRAM decode + frame stack.

Mirrors the logic in pokeai/env.py exactly:
  - BG layer only (LCDC controls map select, tile data mode, scroll)
  - BGP palette: index 0→255, 3→0
  - Downscale 160×144 → 84×84 via cv2.INTER_AREA
  - 4-frame stack: oldest at [0], newest at [-1]

Exposed:
    FrameDecoder   — holds pre-allocated buffers, stateless per call
    FrameStack     — maintains the (FRAME_STACK, H, W) rolling buffer
"""

import ctypes
from typing import Any

import numpy as np

try:
    import cv2  # production path: high-quality INTER_AREA downscale
except ImportError:  # barebone install without OpenCV
    cv2 = None

from rlgb_vec.config import GBVecConfig


def _resize(raw: np.ndarray, w: int, h: int) -> np.ndarray:
    """Downscale (144,160)->(h,w). Uses cv2 INTER_AREA when available, else a
    pure-numpy area/stride fallback so the package runs without OpenCV."""
    if cv2 is not None:
        return cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)
    ys = (np.linspace(0, raw.shape[0], h, endpoint=False)).astype(np.int64)
    xs = (np.linspace(0, raw.shape[1], w, endpoint=False)).astype(np.int64)
    return raw[np.ix_(ys, xs)].astype(np.uint8)

_VRAM_START = 0x8000
_VRAM_SIZE  = 0x2000   # 8 KB
_IO_START   = 0xFF00
_IO_SIZE    = 0x80
_GB_W, _GB_H = 160, 144

_SHADE = np.array([255, 170, 85, 0], dtype=np.uint8)


def _decode_vram_screen(vram: np.ndarray, io: np.ndarray) -> np.ndarray:
    """Pure-numpy BG decode.  Returns (144, 160) uint8."""
    lcdc = int(io[0x40])
    scx  = int(io[0x43])
    scy  = int(io[0x42])
    bgp  = int(io[0x47])

    palette = np.array([_SHADE[(bgp >> (i * 2)) & 0x3] for i in range(4)], dtype=np.uint8)

    map_base   = 0x1C00 if (lcdc & 0x08) else 0x1800
    tile_map   = vram[map_base : map_base + 1024].reshape(32, 32)
    signed_tiles = not bool(lcdc & 0x10)

    tile_data = vram[:0x1800].reshape(384, 16)
    lo = tile_data[:, 0::2]
    hi = tile_data[:, 1::2]
    bits    = np.arange(7, -1, -1, dtype=np.uint8)
    lo_bits = (lo[:, :, np.newaxis] >> bits) & 1
    hi_bits = (hi[:, :, np.newaxis] >> bits) & 1
    indices = (hi_bits << 1) | lo_bits
    pixels  = palette[indices]

    if signed_tiles:
        tile_indices = (tile_map.ravel().astype(np.int8).astype(np.int16) + 256) % 384
    else:
        tile_indices = tile_map.ravel().astype(np.uint16)

    bg_tiles = pixels[tile_indices].reshape(32, 32, 8, 8)
    bg       = bg_tiles.transpose(0, 2, 1, 3).reshape(256, 256)

    rows   = (np.arange(_GB_H) + scy) & 0xFF
    cols   = (np.arange(_GB_W) + scx) & 0xFF
    return bg[np.ix_(rows, cols)]   # (144, 160) uint8


class FrameDecoder:
    """
    Holds pre-allocated ctypes buffers for emu_read_range calls.
    One instance per rl-emu instance (not shared across threads).
    """

    def __init__(self, cfg: GBVecConfig):
        self.cfg      = cfg
        self._vram_c  = (ctypes.c_uint8 * _VRAM_SIZE)()
        self._io_c    = (ctypes.c_uint8 * _IO_SIZE)()
        self._h, self._w = cfg.screen_shape[1], cfg.screen_shape[2]

    def read(self, lib: Any, emu: Any) -> np.ndarray:
        """Return resized (H, W) uint8 frame from current emulator state."""
        lib.emu_read_range(emu, _VRAM_START, _VRAM_SIZE, self._vram_c)
        lib.emu_read_range(emu, _IO_START,   _IO_SIZE,   self._io_c)
        vram = np.frombuffer(self._vram_c, dtype=np.uint8)
        io   = np.frombuffer(self._io_c,   dtype=np.uint8)
        raw  = _decode_vram_screen(vram, io)    # (144, 160)
        return _resize(raw, self._w, self._h)


class FrameStack:
    """
    Rolling buffer: shape (FRAME_STACK, H, W) uint8.
    Oldest frame at [0], newest at [-1].
    """

    def __init__(self, cfg: GBVecConfig):
        fs, h, w = cfg.screen_shape
        self._buf = np.zeros((fs, h, w), dtype=np.uint8)

    def reset(self):
        self._buf[:] = 0

    def push(self, frame: np.ndarray):
        self._buf[:-1] = self._buf[1:]
        self._buf[-1]  = frame

    @property
    def stack(self) -> np.ndarray:
        return self._buf

    def copy(self) -> np.ndarray:
        return self._buf.copy()
