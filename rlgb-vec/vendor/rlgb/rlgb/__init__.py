"""rlgb — a from-scratch Game Boy (DMG) emulator built for RL training.

C core + thin Python bindings. Headless-first, thousands of fps per core,
memcpy save states, every byte of the machine readable and writable.
Implemented from public hardware documentation; no emulator code copied.

CC BY-NC-ND 4.0 license. Built by BuiltNotTaught.
"""
from ._core import BUTTONS, SCREEN_H, SCREEN_W
from .config import DEFAULTS, load_config
from .gameboy import GameBoy, Memory, Registers
from .env import GameBoyEnv
from .db import EmuDB, RecordingEnv
from .vec import VecGameBoyEnv

__version__ = "1.0.0"
__all__ = [
    "GameBoy", "GameBoyEnv", "VecGameBoyEnv", "Memory", "Registers",
    "EmuDB", "RecordingEnv",
    "BUTTONS", "SCREEN_W", "SCREEN_H", "DEFAULTS", "load_config",
]

# optional gymnasium integration: importing rlgb registers rlgb/GameBoy-v0
try:
    from . import gym_env as _gym_env  # noqa: F401
except ImportError:
    pass
