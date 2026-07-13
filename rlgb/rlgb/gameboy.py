"""rlgb.GameBoy — a from-scratch, headless, RL-first Game Boy.

Everything is unlocked: full bus read/write, zero-copy numpy views of every
memory region, CPU register access, instant memcpy save states.

CC BY-NC-ND 4.0 license. Built by BuiltNotTaught.
"""
from __future__ import annotations

import ctypes
import glob
import os

import numpy as np

from . import _core
from .config import load_config, load_callable, resolve_path

__all__ = ["GameBoy", "Memory", "Registers"]


class Memory:
    """Full 64 KiB bus access, exactly as the CPU sees it.

    mem[0xD362]          -> int
    mem[0xC000:0xC100]   -> bytes
    mem[0xD362] = 0x42
    mem[0xC000:0xC004] = b"\\x01\\x02\\x03\\x04"
    """

    def __init__(self, lib, handle):
        self._lib = lib
        self._g = handle

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(0x10000)
            return bytes(self._lib.gb_read(self._g, a) for a in range(start, stop, step))
        return self._lib.gb_read(self._g, key & 0xFFFF)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            start, stop, step = key.indices(0x10000)
            addrs = range(start, stop, step)
            if isinstance(value, int):
                value = [value] * len(addrs)
            for a, v in zip(addrs, value):
                self._lib.gb_write(self._g, a, v & 0xFF)
        else:
            self._lib.gb_write(self._g, key & 0xFFFF, value & 0xFF)

    def __len__(self):
        return 0x10000


class Registers:
    """Direct CPU register read/write (gb.registers.pc, .af, .sp, ...)."""

    _MAP = {
        "a": _core.REG_A, "f": _core.REG_F, "b": _core.REG_B, "c": _core.REG_C,
        "d": _core.REG_D, "e": _core.REG_E, "h": _core.REG_H, "l": _core.REG_L,
        "sp": _core.REG_SP, "pc": _core.REG_PC, "af": _core.REG_AF,
        "bc": _core.REG_BC, "de": _core.REG_DE, "hl": _core.REG_HL,
        "ime": _core.REG_IME, "halted": _core.REG_HALTED,
    }

    def __init__(self, lib, handle):
        object.__setattr__(self, "_lib", lib)
        object.__setattr__(self, "_g", handle)

    def __getattr__(self, name):
        try:
            return self._lib.gb_get_reg(self._g, self._MAP[name])
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        if name not in self._MAP:
            raise AttributeError(name)
        self._lib.gb_set_reg(self._g, self._MAP[name], value)

    def __repr__(self):
        return ("AF={af:04X} BC={bc:04X} DE={de:04X} HL={hl:04X} "
                "SP={sp:04X} PC={pc:04X} IME={ime}").format(
                    **{k: getattr(self, k) for k in
                       ("af", "bc", "de", "hl", "sp", "pc", "ime")})


def _np_view(ptr, size) -> np.ndarray:
    return np.ctypeslib.as_array(ptr, shape=(size,))


class GameBoy:
    """A Game Boy. ``GameBoy("game.gb")`` or ``GameBoy(config="run.toml")``.

    Keyword overrides mirror the TOML structure, e.g.::

        GameBoy("game.gb", emulation={"render": False})
    """

    def __init__(self, rom_path: str | None = None, config: str | None = None,
                 **overrides):
        self.config = load_config(config, overrides or None)
        self._lib = _core.load_library(self.config["paths"]["lib"] or None)
        self._g = self._lib.gb_new()
        if not self._g:
            raise MemoryError("gb_new() failed")

        self.memory = Memory(self._lib, self._g)
        self.registers = Registers(self._lib, self._g)

        rom_path = rom_path or resolve_path(self.config, self.config["rom"]["path"])
        if not rom_path:
            raise ValueError("no ROM: pass rom_path or set [rom].path in the TOML")
        with open(rom_path, "rb") as f:
            rom = f.read()
        rc = self._lib.gb_load_rom(self._g, rom, len(rom))
        if rc != 0:
            raise ValueError(f"gb_load_rom failed ({rc}): bad ROM image?")
        self.rom_path = rom_path

        self.render = bool(self.config["emulation"]["render"])
        self._frame_skip = int(self.config["emulation"]["frame_skip"])

        # zero-copy views into live machine memory (read AND write)
        self.vram = _np_view(self._lib.gb_ptr_vram(self._g), 0x2000)
        self.wram = _np_view(self._lib.gb_ptr_wram(self._g), 0x2000)
        self.oam = _np_view(self._lib.gb_ptr_oam(self._g), 0xA0)
        self.hram = _np_view(self._lib.gb_ptr_hram(self._g), 0x7F)
        self.io = _np_view(self._lib.gb_ptr_io(self._g), 0x80)
        self.cartram = _np_view(self._lib.gb_ptr_cartram(self._g),
                                max(1, self._lib.gb_cartram_size(self._g)))
        self._fb = np.ctypeslib.as_array(
            self._lib.gb_ptr_framebuffer(self._g),
            shape=(_core.SCREEN_H, _core.SCREEN_W))

        pal = self.config["video"]["palette"]
        self._palette = np.asarray(pal, dtype=np.uint8)
        self._gray = np.asarray(self.config["video"]["grayscale_levels"],
                                dtype=np.uint8)

        autoload = self.config["rom"]["autoload"]
        if autoload in ("sav", "both"):
            self.load_sav()
        if autoload in ("state", "both"):
            p = self.latest_state_path()
            if p:
                self.load_state_file(p)

    # ---------------- execution ----------------

    def tick(self, frames: int = 1) -> int:
        """Advance ``frames`` frames (plus configured frame_skip per frame).
        Returns t-cycles executed.

        Only the final frame is rendered — intermediate frames are never
        observed, so their pixel work is skipped (PPU timing still exact)."""
        total = frames * (1 + self._frame_skip)
        if self._render and total > 1:
            self._lib.gb_set_render(self._g, 0)
            t = self._lib.gb_run_frames(self._g, total - 1)
            self._lib.gb_set_render(self._g, 1)
            return t + self._lib.gb_run_frames(self._g, 1)
        return self._lib.gb_run_frames(self._g, total)

    def step(self) -> int:
        """Execute a single CPU instruction. Returns its t-cycles."""
        return self._lib.gb_step(self._g)

    @property
    def cycles(self) -> int:
        return self._lib.gb_cycles(self._g)

    @property
    def frames(self) -> int:
        return self._lib.gb_frames(self._g)

    def reset(self):
        """Hardware reset (battery-backed cart RAM survives, like real HW)."""
        self._lib.gb_reset(self._g)

    # ---------------- video ----------------

    @property
    def render(self) -> bool:
        return self._render

    @render.setter
    def render(self, on: bool):
        self._render = bool(on)
        self._lib.gb_set_render(self._g, int(on))

    @property
    def screen(self) -> np.ndarray:
        """(144, 160) uint8, raw DMG shades 0..3. Zero-copy."""
        return self._fb

    @property
    def screen_gray(self) -> np.ndarray:
        """(144, 160) uint8 grayscale via [video].grayscale_levels."""
        return self._gray[self._fb]

    @property
    def screen_rgb(self) -> np.ndarray:
        """(144, 160, 3) uint8 RGB via [video].palette."""
        return self._palette[self._fb]

    def screenshot(self, path: str):
        from PIL import Image
        Image.fromarray(self.screen_rgb).save(path)

    # ---------------- input ----------------

    def set_buttons(self, *names: str, **flags: bool):
        """Set the exact pressed set: ``set_buttons("a", "right")``."""
        mask = 0
        for n in names:
            mask |= _core.BUTTONS[n.lower()]
        for n, on in flags.items():
            if on:
                mask |= _core.BUTTONS[n.lower()]
        self._lib.gb_set_buttons(self._g, mask)

    def press(self, name: str):
        cur = self._lib.gb_get_buttons(self._g)
        self._lib.gb_set_buttons(self._g, cur | _core.BUTTONS[name.lower()])

    def release(self, name: str | None = None):
        if name is None:
            self._lib.gb_set_buttons(self._g, 0)
        else:
            cur = self._lib.gb_get_buttons(self._g)
            self._lib.gb_set_buttons(self._g, cur & ~_core.BUTTONS[name.lower()])

    def push(self, name: str, hold_frames: int | None = None,
             release_frames: int | None = None):
        """Press, hold for [input].hold_frames, release, settle."""
        hold = self.config["input"]["hold_frames"] if hold_frames is None else hold_frames
        rel = self.config["input"]["release_frames"] if release_frames is None else release_frames
        self.press(name)
        self.tick(hold)
        self.release(name)
        self.tick(rel)

    # ---------------- save states ----------------

    def save_state(self) -> bytes:
        n = self._lib.gb_state_size()
        buf = ctypes.create_string_buffer(n)
        self._lib.gb_save_state(self._g, buf)
        return buf.raw

    def load_state(self, blob: bytes):
        rc = self._lib.gb_load_state(self._g, blob, len(blob))
        if rc != 0:
            raise ValueError("state blob size mismatch (built by another core version?)")

    def save_state_file(self, path: str | None = None) -> str:
        if path is None:
            d = resolve_path(self.config, self.config["paths"]["states"])
            os.makedirs(d, exist_ok=True)
            base = os.path.splitext(os.path.basename(self.rom_path))[0]
            path = os.path.join(d, f"{base}-f{self.frames}.state")
        with open(path, "wb") as f:
            f.write(self.save_state())
        return path

    def load_state_file(self, path: str):
        with open(path, "rb") as f:
            self.load_state(f.read())

    def latest_state_path(self) -> str | None:
        """Most recently modified .state snapshot for this ROM's basename in
        [paths].states, or None if none exist yet."""
        d = resolve_path(self.config, self.config["paths"]["states"])
        base = os.path.splitext(os.path.basename(self.rom_path))[0]
        candidates = glob.glob(os.path.join(d, f"{base}-f*.state"))
        return max(candidates, key=os.path.getmtime) if candidates else None

    # ---------------- .sav (battery-backed cart RAM) ----------------
    # Distinct from save_state()/.state above: a .sav is just the raw cart
    # RAM bytes (the standard format real hardware, flashcarts, and other
    # emulators use), not a full-machine snapshot. Not size-portable across
    # different ROMs' cart RAM sizes; is portable across emulators.

    def sav_path(self, path: str | None = None) -> str:
        """Default .sav path: same directory and basename as the ROM."""
        if path is not None:
            return path
        base, _ = os.path.splitext(self.rom_path)
        return base + ".sav"

    def save_sav(self, path: str | None = None) -> str:
        """Write battery-backed cart RAM to a .sav file. No-op (returns the
        would-be path without writing) if the cart has no battery RAM."""
        path = self.sav_path(path)
        n = self._lib.gb_cartram_size(self._g)
        if n:
            with open(path, "wb") as f:
                f.write(self.cartram[:n].tobytes())
        return path

    def load_sav(self, path: str | None = None) -> bool:
        """Load a .sav file into cart RAM. Returns False (no-op) if the
        file doesn't exist or the cart has no battery RAM."""
        path = self.sav_path(path)
        n = self._lib.gb_cartram_size(self._g)
        if not n or not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = f.read()
        m = min(len(data), n)
        self.cartram[:m] = np.frombuffer(data[:m], dtype=np.uint8)
        return True

    # ---------------- unified save/load (per [rom].save_format) ----------

    def save(self, path: str | None = None) -> dict:
        """Persist according to [rom].save_format ('sav' | 'state' | 'both').
        ``path`` only applies when a single format is selected; with 'both'
        each format writes to its own default location. Returns a dict of
        whichever path(s) were written, keyed by format."""
        fmt = self.config["rom"]["save_format"]
        out = {}
        if fmt in ("sav", "both"):
            out["sav"] = self.save_sav(None if fmt == "both" else path)
        if fmt in ("state", "both"):
            out["state"] = self.save_state_file(None if fmt == "both" else path)
        return out

    def load(self, path: str | None = None) -> dict:
        """Load according to [rom].save_format ('sav' | 'state' | 'both').
        ``path`` only applies when a single format is selected; with 'both'
        each format reads from its own default location (.sav next to the
        ROM, most recent .state in [paths].states). Returns a dict of bools
        keyed by format, True where something was actually loaded."""
        fmt = self.config["rom"]["save_format"]
        out = {}
        if fmt in ("sav", "both"):
            out["sav"] = self.load_sav(None if fmt == "both" else path)
        if fmt in ("state", "both"):
            p = self.latest_state_path() if fmt == "both" else (path or self.latest_state_path())
            if p:
                self.load_state_file(p)
            out["state"] = bool(p)
        return out

    # ---------------- PyBoy state interop (reverse-engineered format) ----

    def load_pyboy_state(self, src) -> int:
        """Load a PyBoy .state (versions 0-17, DMG). Accepts bytes or a path.
        Returns the file's state version."""
        from . import pyboy_state
        if isinstance(src, (bytes, bytearray)):
            data = bytes(src)
        else:
            with open(src, "rb") as f:
                data = f.read()
        return pyboy_state.load_pyboy_state(self, data)

    def save_pyboy_state(self, path: str | None = None) -> bytes:
        """Serialize as a PyBoy-loadable .state (v13). Returns the bytes;
        writes them to ``path`` too if given."""
        from . import pyboy_state
        data = pyboy_state.save_pyboy_state(self)
        if path:
            with open(path, "wb") as f:
                f.write(data)
        return data

    # ---------------- misc ----------------

    def serial(self) -> bytes:
        """Drain bytes the game pushed out the link port."""
        buf = ctypes.create_string_buffer(4096)
        n = self._lib.gb_serial_read(self._g, buf, 4096)
        return buf.raw[:n]

    @property
    def cart_type(self) -> int:
        return self._lib.gb_cart_type(self._g)

    def hook(self, spec: str):
        """Load a 'pkg.mod:fn' plug-in and bind it to this GameBoy."""
        fn = load_callable(spec)
        return (lambda *a, **k: fn(self, *a, **k)) if fn else None

    def close(self):
        if self._g:
            self._lib.gb_free(self._g)
            self._g = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
