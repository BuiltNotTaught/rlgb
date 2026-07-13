"""TOML configuration for rlgb — nothing is hardcoded in the emulator API.

Resolution order (later wins):
    built-in defaults  <  config file (TOML)  <  keyword overrides

CC BY-NC-ND 4.0 license. Built by BuiltNotTaught.
"""
from __future__ import annotations

import copy
import os

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

DEFAULTS: dict = {
    "rom": {
        "path": "",
        # what save()/load() (and auto-load on construction) target:
        #   "sav"   - battery-backed cart RAM only (portable, like real HW)
        #   "state" - full-machine .state snapshot only (this core's own format)
        #   "both"  - both, independently
        #   "none"  - no auto-load on construction (save_format still applies to save()/load())
        "autoload": "sav",
        "save_format": "sav",
    },
    "emulation": {
        "render": True,        # False = skip pixel work entirely (fastest)
        "frame_skip": 0,       # extra frames advanced per tick()
        "sound": False,        # reserved; APU is a register stub either way
    },
    "video": {
        # shade index 0..3 -> RGB. Classic DMG pea-soup by default.
        "palette": [[224, 248, 208], [136, 192, 112], [52, 104, 86], [8, 24, 32]],
        "grayscale_levels": [255, 170, 85, 0],
    },
    "input": {
        "hold_frames": 5,      # GameBoy.push(): frames held down
        "release_frames": 1,   # GameBoy.push(): frames after release
    },
    "paths": {
        "lib": "",             # override libgb.so location ("" = bundled)
        "states": "states",    # default directory for .state files
    },
    "db": {
        "path": "",            # sqlite file for EmuDB ("" = not used)
        "compress": 6,         # zlib level for state blobs, 0-9
    },
    "env": {
        # gym-style wrapper (rlgb.env.GameBoyEnv); "" = no-op action
        "actions": ["", "a", "b", "start", "select", "up", "down", "left", "right"],
        "obs": "shades",       # shades | gray | rgb
        "frames_per_action": 24,
        "press_frames": 8,     # of frames_per_action, how many the button is held
        "reward": "",          # dotted path "pkg.mod:fn"; fn(gb) -> float
        "done": "",            # dotted path "pkg.mod:fn"; fn(gb) -> bool
        "info": "",            # dotted path "pkg.mod:fn"; fn(gb) -> dict
        "max_steps": 0,        # 0 = unlimited
    },
}


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    """Build the effective config dict.

    ``path``: optional TOML file. If None, ``rlgb.toml`` / ``config.toml`` in
    the current directory are picked up automatically when present.
    ``overrides``: nested dict applied last (e.g. from kwargs).
    """
    cfg = copy.deepcopy(DEFAULTS)
    if path is None:
        for candidate in ("rlgb.toml", "config.toml"):
            if os.path.isfile(candidate):
                path = candidate
                break
    if path:
        with open(path, "rb") as f:
            cfg = _deep_merge(cfg, tomllib.load(f))
        cfg["_config_dir"] = os.path.dirname(os.path.abspath(path))
    else:
        cfg["_config_dir"] = os.getcwd()
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def resolve_path(cfg: dict, p: str) -> str:
    """Resolve a possibly-relative config path against the config file dir."""
    if not p or os.path.isabs(p):
        return p
    return os.path.join(cfg.get("_config_dir", os.getcwd()), p)


def load_callable(spec: str):
    """Load 'package.module:function' — the plug-in hook used by env.py."""
    if not spec:
        return None
    mod_name, _, fn_name = spec.partition(":")
    if not fn_name:
        raise ValueError(f"callable spec {spec!r} must look like 'pkg.mod:fn'")
    import importlib
    return getattr(importlib.import_module(mod_name), fn_name)
