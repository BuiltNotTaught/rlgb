"""SQLite persistence for rlgb — save states, runs, episodes, metrics.

One ``EmuDB`` wraps one ``.db`` file (stdlib sqlite3 + zlib, no new deps).
Stores: named/tagged save states with parent lineage (branching exploration),
runs (config + notes), per-episode trajectories (packed action/reward arrays,
before/after state snapshots), and free-form numeric metrics.

State blobs are zlib-compressed (a 167 KiB state is typically a few KiB).
Timestamps are host-side only — the emulator core stays deterministic.

CC BY-NC-ND 4.0 license. Built by BuiltNotTaught.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import zlib

import numpy as np

__all__ = ["EmuDB", "RecordingEnv"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS roms (
    id         INTEGER PRIMARY KEY,
    sha1       TEXT NOT NULL UNIQUE,
    name       TEXT NOT NULL,
    size       INTEGER NOT NULL,
    cart_type  INTEGER NOT NULL,
    added_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS states (
    id         INTEGER PRIMARY KEY,
    rom_id     INTEGER NOT NULL REFERENCES roms(id),
    name       TEXT,
    tag        TEXT,
    parent_id  INTEGER REFERENCES states(id),
    frame      INTEGER NOT NULL,
    cycles     INTEGER NOT NULL,
    raw_size   INTEGER NOT NULL,
    blob       BLOB NOT NULL,
    note       TEXT,
    created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS states_name
    ON states(rom_id, name) WHERE name IS NOT NULL;
CREATE INDEX IF NOT EXISTS states_tag ON states(rom_id, tag);
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    rom_id      INTEGER NOT NULL REFERENCES roms(id),
    config      TEXT,
    note        TEXT,
    started_at  REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS episodes (
    id             INTEGER PRIMARY KEY,
    run_id         INTEGER NOT NULL REFERENCES runs(id),
    idx            INTEGER NOT NULL,
    steps          INTEGER NOT NULL,
    total_reward   REAL NOT NULL,
    terminated     INTEGER NOT NULL,
    truncated      INTEGER NOT NULL,
    start_state_id INTEGER REFERENCES states(id),
    end_state_id   INTEGER REFERENCES states(id),
    actions        BLOB NOT NULL,
    rewards        BLOB NOT NULL,
    info           TEXT,
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS episodes_run ON episodes(run_id);
CREATE TABLE IF NOT EXISTS metrics (
    id         INTEGER PRIMARY KEY,
    run_id     INTEGER NOT NULL REFERENCES runs(id),
    key        TEXT NOT NULL,
    value      REAL NOT NULL,
    frame      INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS metrics_run ON metrics(run_id, key);
"""


class EmuDB:
    """``EmuDB("run.db")`` — persistence for one or more GameBoy sessions."""

    def __init__(self, path: str, compress: int = 6):
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        self._db.execute("PRAGMA user_version=1")
        self._db.commit()
        self._compress = int(compress)
        self._rom_ids: dict[str, int] = {}      # rom_path -> roms.id

    def close(self):
        self._db.commit()
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---------------- roms ----------------

    def _rom_id(self, gb) -> int:
        rid = self._rom_ids.get(gb.rom_path)
        if rid is not None:
            return rid
        with open(gb.rom_path, "rb") as f:
            data = f.read()
        sha1 = hashlib.sha1(data).hexdigest()
        row = self._db.execute("SELECT id FROM roms WHERE sha1=?", (sha1,)).fetchone()
        if row:
            rid = row["id"]
        else:
            cur = self._db.execute(
                "INSERT INTO roms (sha1, name, size, cart_type, added_at) "
                "VALUES (?,?,?,?,?)",
                (sha1, os.path.basename(gb.rom_path), len(data),
                 int(gb._lib.gb_cart_type(gb._g)), time.time()))
            rid = cur.lastrowid
            self._db.commit()
        self._rom_ids[gb.rom_path] = rid
        return rid

    # ---------------- save states ----------------

    def save_state(self, gb, name: str | None = None, tag: str | None = None,
                   parent: int | None = None, note: str | None = None) -> int:
        """Snapshot ``gb`` into the DB; returns the state id.
        A ``name`` is unique per ROM — saving it again replaces the old row."""
        raw = gb.save_state()
        blob = zlib.compress(raw, self._compress)
        rid = self._rom_id(gb)
        if name is not None:
            self._db.execute(
                "DELETE FROM states WHERE rom_id=? AND name=?", (rid, name))
        cur = self._db.execute(
            "INSERT INTO states (rom_id, name, tag, parent_id, frame, cycles,"
            " raw_size, blob, note, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rid, name, tag, parent, gb.frames, gb.cycles,
             len(raw), blob, note, time.time()))
        self._db.commit()
        return cur.lastrowid

    def _state_row(self, gb, ref):
        if isinstance(ref, str):
            row = self._db.execute(
                "SELECT * FROM states WHERE rom_id=? AND name=?",
                (self._rom_id(gb), ref)).fetchone()
        else:
            row = self._db.execute(
                "SELECT * FROM states WHERE id=?", (int(ref),)).fetchone()
        if not row:
            raise KeyError(f"no such state: {ref!r}")
        return row

    def load_state(self, gb, ref: int | str) -> int:
        """Restore a state into ``gb`` by id or name; returns the state id."""
        row = self._state_row(gb, ref)
        # Bounded decompress: a hostile .db could hold a zip bomb. A real
        # state is exactly gb_state_size(); cap a little above that.
        limit = gb._lib.gb_state_size() + 1024
        raw = zlib.decompressobj().decompress(row["blob"], limit)
        gb.load_state(raw)                     # C core validates size + fields
        return row["id"]

    def states(self, gb=None, tag: str | None = None) -> list[dict]:
        q, args = "SELECT id, rom_id, name, tag, parent_id, frame, cycles," \
                  " raw_size, note, created_at FROM states", []
        cond = []
        if gb is not None:
            cond.append("rom_id=?"); args.append(self._rom_id(gb))
        if tag is not None:
            cond.append("tag=?"); args.append(tag)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        return [dict(r) for r in self._db.execute(q + " ORDER BY id", args)]

    def delete_state(self, gb, ref: int | str):
        self._db.execute("DELETE FROM states WHERE id=?",
                         (self._state_row(gb, ref)["id"],))
        self._db.commit()

    # ---------------- runs / episodes / metrics ----------------

    def begin_run(self, gb, note: str | None = None) -> int:
        cur = self._db.execute(
            "INSERT INTO runs (rom_id, config, note, started_at) VALUES (?,?,?,?)",
            (self._rom_id(gb),
             json.dumps({k: v for k, v in gb.config.items()
                         if not k.startswith("_")}),
             note, time.time()))
        self._db.commit()
        return cur.lastrowid

    def end_run(self, run_id: int):
        self._db.execute("UPDATE runs SET finished_at=? WHERE id=?",
                         (time.time(), run_id))
        self._db.commit()

    def log_metric(self, run_id: int, key: str, value: float,
                   frame: int | None = None):
        self._db.execute(
            "INSERT INTO metrics (run_id, key, value, frame, created_at)"
            " VALUES (?,?,?,?,?)",
            (run_id, key, float(value), frame, time.time()))
        self._db.commit()

    def log_episode(self, run_id: int, actions, rewards,
                    terminated: bool, truncated: bool = False,
                    start_state: int | None = None,
                    end_state: int | None = None,
                    info: dict | None = None) -> int:
        acts = np.asarray(actions, dtype=np.int16)
        rews = np.asarray(rewards, dtype=np.float32)
        idx = self._db.execute(
            "SELECT COUNT(*) FROM episodes WHERE run_id=?",
            (run_id,)).fetchone()[0]
        cur = self._db.execute(
            "INSERT INTO episodes (run_id, idx, steps, total_reward,"
            " terminated, truncated, start_state_id, end_state_id, actions,"
            " rewards, info, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, idx, len(acts), float(rews.sum()), int(terminated),
             int(truncated), start_state, end_state, acts.tobytes(),
             rews.tobytes(), json.dumps(info) if info else None, time.time()))
        self._db.commit()
        return cur.lastrowid

    def episodes(self, run_id: int) -> list[dict]:
        """Episodes of a run, with actions/rewards decoded back to arrays."""
        out = []
        for r in self._db.execute(
                "SELECT * FROM episodes WHERE run_id=? ORDER BY idx", (run_id,)):
            d = dict(r)
            d["actions"] = np.frombuffer(d["actions"], dtype=np.int16)
            d["rewards"] = np.frombuffer(d["rewards"], dtype=np.float32)
            d["info"] = json.loads(d["info"]) if d["info"] else None
            out.append(d)
        return out


class RecordingEnv:
    """Wrap a ``GameBoyEnv`` so every episode lands in an ``EmuDB``.

    Buffers actions/rewards in memory and writes one row per episode (plus
    before/after state snapshots when ``snapshot=True``) — no per-step DB
    work, so the emulator stays fast.
    """

    def __init__(self, env, db: EmuDB, note: str | None = None,
                 snapshot: bool = True):
        self.env = env
        self.db = db
        self.snapshot = snapshot
        self.run_id = db.begin_run(env.gb, note=note)
        self._acts: list[int] = []
        self._rews: list[float] = []
        self._start_state: int | None = None
        self._open = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _flush(self, terminated: bool, truncated: bool, info):
        if not self._open:
            return
        end_state = (self.db.save_state(self.env.gb, tag="episode-end",
                                        parent=self._start_state)
                     if self.snapshot else None)
        self.db.log_episode(self.run_id, self._acts, self._rews, terminated,
                            truncated, self._start_state, end_state,
                            info if isinstance(info, dict) else None)
        self._acts, self._rews = [], []
        self._open = False

    def reset(self, **kw):
        self._flush(False, True, None)          # abandoned episode = truncated
        obs, info = self.env.reset(**kw)
        self._start_state = (self.db.save_state(self.env.gb, tag="episode-start")
                             if self.snapshot else None)
        self._open = True
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._acts.append(int(action))
        self._rews.append(float(reward))
        if terminated or truncated:
            self._flush(terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    def close(self):
        self._flush(False, True, None)
        self.db.end_run(self.run_id)
        self.env.close()
