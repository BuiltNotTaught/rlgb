"""rlgb-vec: plug-and-play async VecEnv bridge between the vendored rlgb (DMG)
emulator and SB3.

Quickstart
----------
    from rlgb_vec import make_env
    from stable_baselines3 import PPO

    vec   = make_env("/roms/pokemon_red.gb", n_envs=12)
    model = PPO("MultiInputPolicy", vec)
    model.learn(1_000_000)
    vec.close()

Or from the shell:  python -m rlgb_vec.train --rom /roms/pokemon_red.gb
"""

from rlgb_vec.config import GBVecConfig


def __getattr__(name):
    # lazy so `import rlgb_vec` and make_env work without torch at import time
    if name in ("make_env", "make_gb_vec_env", "build_config"):
        import rlgb_vec.adapter as _a
        return getattr(_a, name)
    if name == "GBVecEnv":
        from rlgb_vec.vec_env import GBVecEnv
        return GBVecEnv
    raise AttributeError(f"module 'rlgb_vec' has no attribute {name!r}")


__all__ = ["make_env", "make_gb_vec_env", "build_config", "GBVecEnv", "GBVecConfig"]
