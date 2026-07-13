"""Smoke test: ShmPool create/attach/write/read/close without rl-emu."""

import numpy as np
import pytest

from rlgb_vec.config import GBVecConfig
from rlgb_vec.shm import ShmPool, ShmSlot, FLAG_IDLE, FLAG_OBS_READY, FLAG_ACT_READY


@pytest.fixture
def cfg():
    return GBVecConfig(rom_path="fake.gb", n_workers=2, envs_per_worker=2)


def test_slot_size_nonzero(cfg):
    from rlgb_vec.shm import _slot_size
    assert _slot_size(cfg) > 100


def test_pool_create_and_close(cfg):
    pool = ShmPool(cfg)
    assert len(pool.names) == 4
    pool.close()


def test_obs_roundtrip(cfg):
    pool = ShmPool(cfg)
    slot = ShmSlot(cfg, pool.names[0])

    fs, h, w = cfg.screen_shape
    rs = cfg.ram_size
    screen_in = np.random.randint(0, 256, (fs, h, w), dtype=np.uint8)
    ram_in    = np.random.randint(0, 256, (rs,),       dtype=np.uint8)

    slot.write_obs(screen_in, ram_in)
    screen_out = np.zeros_like(screen_in)
    ram_out    = np.zeros_like(ram_in)
    slot.read_obs_into(screen_out, ram_out)

    assert np.array_equal(screen_in, screen_out)
    assert np.array_equal(ram_in,    ram_out)

    slot.close()
    pool.close()


def test_flag_lifecycle(cfg):
    pool = ShmPool(cfg)
    slot = ShmSlot(cfg, pool.names[0])

    assert slot.get_flag() == FLAG_IDLE
    slot.set_flag(FLAG_OBS_READY)
    assert slot.get_flag() == FLAG_OBS_READY
    slot.set_flag(FLAG_ACT_READY)
    assert slot.get_flag() == FLAG_ACT_READY

    slot.close()
    pool.close()


def test_action_roundtrip(cfg):
    pool = ShmPool(cfg)
    slot = ShmSlot(cfg, pool.names[0])

    for a in range(cfg.n_actions):
        slot.write_action(a)
        assert slot.read_action() == a

    slot.close()
    pool.close()


def test_telemetry_roundtrip(cfg):
    from rlgb_vec.config import TELEMETRY_FIELDS
    pool = ShmPool(cfg)
    slot = ShmSlot(cfg, pool.names[0])

    values = np.arange(len(TELEMETRY_FIELDS), dtype=np.float32) + 0.5
    slot.write_telemetry(values)
    out = slot.read_telemetry()

    assert out.dtype == np.float32
    assert np.array_equal(values, out)

    # telemetry must not clobber the reward/done region
    slot.write_result(1.25, True)
    slot.write_telemetry(values)
    r, d = slot.read_result()
    assert r == 1.25 and d is True
    assert np.array_equal(slot.read_telemetry(), values)

    slot.close()
    pool.close()
