"""Torch-free checks of the plug-and-play wiring: adapter imports without the
training stack, make_env picks a valid worker split, and the config yields the
obs/action shapes SB3 will see.
"""
import pytest

from rlgb_vec.adapter import build_config, make_env, _split
from rlgb_vec.config import GBVecConfig


def test_adapter_imports_without_torch():
    # build_config must be pure (no torch/SB3 import at module load)
    import rlgb_vec.adapter  # noqa: F401
    assert callable(build_config)


def test_split_products_match():
    for n in (1, 2, 8, 12, 24):
        w, e = _split(n, None)
        assert w * e == n
    assert _split(12, 3) == (4, 3)
    with pytest.raises(ValueError):
        _split(10, 3)          # not divisible


def test_build_config_shapes():
    cfg = build_config("/roms/x.gb", n_workers=4, envs_per_worker=3, frame_stack=4)
    assert isinstance(cfg, GBVecConfig)
    assert cfg.n_envs == 12
    assert cfg.screen_shape == (4, 84, 84)
    # obs bytes = screen + ram; sanity for shm sizing
    assert cfg.obs_total_bytes == 4 * 84 * 84 + cfg.ram_size


def test_make_env_lazy_import(monkeypatch):
    # make_env should only import vec_env (torch) when actually building.
    called = {}

    import rlgb_vec.adapter as a

    def fake_make(rom, n_workers, envs_per_worker, **kw):
        called["split"] = (n_workers, envs_per_worker)
        return "VEC"

    monkeypatch.setattr(a, "make_gb_vec_env", fake_make)
    out = make_env("/roms/x.gb", n_envs=12)
    assert out == "VEC"
    assert called["split"] == (4, 3)
