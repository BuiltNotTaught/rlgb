# Game Boy Emulator for Reinforcement Learning

A from-scratch, headless-first Game Boy (DMG) emulator built for high-throughput
reinforcement-learning training with Stable Baselines 3 (SB3). Written in C with
thin Python bindings.

> **Scope & honesty:** This is a *throughput-first* core, not a reference
> accuracy emulator. It runs Pokémon Red (and similar MBC1/3/5 titles) correctly
> and fast; it is **not** intended to compete with SameBoy/mGBA on sub-instruction
> timing accuracy. See [Accuracy & Scope](#accuracy--scope) before relying on it
> for anything timing-sensitive.

## Emulator

### rlgb
From-scratch Game Boy (DMG) core, implemented from public hardware documentation
(Pan Docs, the gbdev community opcode tables). **No emulator source was copied.**

- **CPU:** SM83, **instruction-stepped** using community-verified per-opcode
  cycle counts (`src/cycles.inc`, generated from `gbdev/gb-opcodes`). Subsystems
  (PPU, timer, RTC) are advanced per instruction in a burst, not tick-by-tick.
- **PPU:** scanline renderer with **mode-boundary-accurate timing** — it walks
  every mode 2 → 3 → 0 transition and line wrap (dots 80 / 252 / 456) and raises
  STAT / LYC / VBlank interrupts at those boundaries. It renders a full scanline
  at mode-3 entry; it is **not** a per-dot pixel FIFO, so mid-scanline raster
  tricks are not reproduced.
- **Memory:** flat, pointer-free `GBState` struct — a save state is a single
  `memcpy` (~167 KiB), fully deterministic.
- **MBC:** MBC1 / MBC2 / MBC3 (+ RTC) / MBC5, battery-backed cart RAM.
- **Performance:** ~3,200 fps single-core, headless (`render=false`), on the
  author's machine. Benchmark on your own hardware — the number is CPU-bound and
  varies with rendering settings.

## SB3 Training Bridge

### rlgb-vec

Plug-and-play async VecEnv for SB3 training.

**Quick Start:**
```python
from rlgb_vec import make_gb_vec_env, build_config

config = build_config('/path/to/game.gb', n_envs=12)
vec_env = make_gb_vec_env(config)

from stable_baselines3 import PPO
model = PPO('CnnPolicy', vec_env)
model.learn(total_timesteps=1_000_000)
```

**Features:**
- Multi-process worker pool with shared-memory IPC (no per-step pickling)
- Optional GPU inference worker for batched policy evaluation
- Frame observation (84×84 luma) + configurable RAM observation window
- Portable across CPU/GPU, Docker/barebone; no hardcoded paths

**Build:**
```bash
cd rlgb-vec
./build.sh                    # barebone (requires gcc, make)
docker build -t rlgb-vec .    # Docker (portable, includes CUDA 12.1)
```

**Install:**
```bash
pip install -e rlgb-vec[train]  # torch, SB3, OpenCV
```

**Test:**
```bash
pytest rlgb-vec/tests/           # binding / shared-memory / wiring tests
```

## Folder Structure

```
emulator/
├── rlgb/                 # Game Boy emulator core
│   ├── src/              # C implementation (gb.c, ppu.c, gb.h, cycles.inc)
│   ├── rlgb/             # Python bindings & utils
│   ├── Makefile          # builds libgb.so
│   └── pyproject.toml
│
├── rlgb-vec/             # SB3 async training bridge for rlgb
│   ├── rlgb_vec/         # Python env + worker code
│   ├── vendor/rlgb/      # vendored emulator (libgb.so built here)
│   ├── build.sh
│   ├── Dockerfile        # multi-stage CUDA 12.1 build
│   └── tests/
│
└── README.md
```

## C Core API

The emulator is a self-contained C library with ctypes Python bindings:

- `emu_create(rom, size)` → create emulator
- `emu_step_frames(emu, n)` → advance N frames
- `emu_save_state(emu, buf)` → flat POD state snapshot
- `emu_load_state(emu, buf)` → restore state
- `emu_set_joypad(emu, mask)` → input
- `emu_export_wram(emu, buf)` → raw WRAM access

See `rlgb/rlgb/_core.py` for the full binding surface.

## Observation Formats

- **Frame:** 84×84 uint8 grayscale (4-shade DMG palette → luma).
- **RAM:** configurable WRAM window (game-specific state); override
  `RAMObsReader` for custom extraction.

## Performance

Single-core, bare metal, headless:

- **rlgb:** ~3,200 fps (`render=false`), author's machine.

Aggregate throughput with the vec bridge scales with core count and depends on
`n_envs`, `frame_skip`, batch size, and whether a GPU inference worker is used —
it is **hardware-dependent, so benchmark your own setup** rather than relying on
a headline number.

## Accuracy & Scope

Read this before using rlgb for anything beyond RL training:

- **Not validated against accuracy test ROMs** (Blargg, mooneye-gb) as of this
  release. Correctness evidence today is: it boots and plays Pokémon Red end to
  end, and its WRAM layout matches the `pret/pokered` disassembly. Timing-test
  results are **not yet published** — do not assume it passes them.
- **Instruction-granular timing.** Subsystems catch up per instruction, not per
  T-cycle. Effects that depend on sub-instruction timing (mid-instruction memory
  timing, precise DMA/PPU interaction) are **not** modeled.
- **Scanline PPU.** Mode transitions and STAT/LYC/VBlank interrupts are timed at
  mode boundaries, but pixels are drawn a whole scanline at a time — mid-scanline
  raster effects are not reproduced.
- **DMG only.** `.gbc` images load (padded like unmapped cartridge reads) but run
  in **DMG mode**: no CGB color, no double-speed, no CGB-exclusive behavior.
- **No audio.** The APU is a register stub; no sound is synthesized.

If your use case needs any of the above, use a reference emulator (SameBoy,
mGBA). If you need to run Pokémon-class games headless at thousands of fps into
SB3, that is exactly what this is for.

## Requirements

**Barebone:** GCC/Clang, GNU Make, Python 3.10+, numpy. Optional: opencv-python
(faster image resize).

**Docker:** Docker/Podman, NVIDIA Container Runtime for GPU. ROMs mounted at
runtime (none are baked in).

## Portability

- No hardcoded paths (ROM discovery via env vars / relative paths).
- `-mtune=generic` in Docker; `-march=native` optional locally.
- Device selection via config (`auto` → CUDA if available, else CPU).

## License

CC BY-NC-ND 4.0 license. See `LICENSE`. No ROM files are included — provide your own or use
emulation-legal test ROMs.

## Contributing

Pull requests welcome. Please preserve:
- Flat POD memory layouts (save-state correctness).
- Deterministic emulation.
- Portable build system (no external binary dependencies).
- Honest capability claims — if you add accuracy, back it with a test-ROM result;
  don't inflate the description ahead of the code.
