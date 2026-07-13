# rlgb — A High-Performance Game Boy Emulator for Reinforcement Learning

A from-scratch, headless-first Game Boy (DMG) emulator built for RL training and research. Written in C with thin Python bindings, delivering ~3,200 FPS on bare metal with zero-copy memory access and deterministic save states.

## Features

- **High Performance**: ~3,200 FPS on bare metal, depending on rendering settings and hardware
- **Headless & RL-First**: Designed for batch training, not games—skip rendering entirely for peak speed
- **Full Memory Access**: Every byte of the Game Boy's 64 KiB bus is readable and writable in Python
- **CPU Register Access**: Direct read/write to all CPU registers (A–L, SP, PC, IME)
- **Instant Save States**: Memcpy save/load—deterministic full-machine snapshots in microseconds
- **TOML Configuration**: All behavior (actions, observations, frame timing, reward logic) declared in config files, not hardcoded
- **Vectorized RL**: `VecGameBoyEnv` runs N emulators in parallel across N CPU cores (threads release the GIL)
- **SQLite Persistence**: `EmuDB` stores states, runs, episodes, and metrics—compressed state blobs, branching exploration
- **Gymnasium Compatible**: Drops straight into Stable Baselines 3 and other RL libraries
- **No Emulator Code Copied**: Implemented from public hardware documentation (SM83 ISA, PPU specs, memory map)

## Installation

### From PyPI (when released)

```bash
pip install rlgb
```

### From Source

```bash
git clone <repo>
cd rlgb
make                    # builds libgb.so (requires C compiler + standard libs)
pip install -e .        # editable install
```

### Optional Dependencies

```bash
pip install rlgb[gym]        # gymnasium integration
pip install rlgb[sb3]        # Stable Baselines 3 support
pip install rlgb[image]      # pillow (for image saving)
```

## Quick Start

### 1. Load a ROM and Run

```python
from rlgb import GameBoy

gb = GameBoy("pokemon_red.gb")
gb.tick(60)           # advance 60 frames
print(gb.screen_rgb)  # 144x160 RGB array
gb.close()
```

### 2. RL Training with Gymnasium

```python
from rlgb import GameBoyEnv

env = GameBoyEnv("pokemon_red.gb", config="config.toml")
obs, info = env.reset()

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### 3. Parallel Training (Vectorized)

```python
from rlgb import VecGameBoyEnv

env = VecGameBoyEnv(n=16, rom_path="pokemon_red.gb")
obs, infos = env.reset()

for step in range(1000):
    actions = [env.action_space.sample() for _ in range(16)]
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

env.close()
```

### 4. Full Memory & Register Access

```python
gb = GameBoy("game.gb")

# Read/write individual bytes
hp = gb.memory[0xC3EB]        # read byte at address
gb.memory[0xC3EB] = 100       # write byte

# Slice access
sprite_data = gb.memory[0x8000:0x9000]  # 4 KiB

# CPU registers
pc_before = gb.registers.pc
gb.registers.pc = 0x100
a_register = gb.registers.a
gb.registers.af = 0x00F0

print(gb.registers)  # AF=00F0 BC=0000 DE=0000 HL=0000 SP=FFFE PC=0100 IME=0
```

### 5. Save & Load States

```python
gb = GameBoy("game.gb")
gb.tick(100)

state = gb.save_state()       # memcpy snapshot (~167 KiB)
gb.tick(500)

gb.load_state(state)          # instant deterministic reload
# now at frame 100 again
```

### 6. Input & Buttons

```python
gb = GameBoy("pokemon_red.gb")

# Single button press
gb.set_buttons("a")           # press A button
gb.tick(8)                    # hold for 8 frames
gb.release()
gb.tick(2)                    # release for 2 frames

# Or use helper
gb.push("start")              # hold_frames=5, release_frames=1 (configurable)
gb.tick(60)
```

### 7. TOML Configuration

Create `config.toml`:

```toml
[rom]
path = "pokemon_red.gb"

[emulation]
render = false                # skip pixel work (fastest)
frame_skip = 0

[env]                         # gym-style wrapper
actions = ["", "a", "b", "start", "select", "up", "down", "left", "right"]
obs = "rgb"                   # shades | gray | rgb
frames_per_action = 24
press_frames = 8
reward = "my_pkg.rewards:level_gain"   # fn(gb: GameBoy) -> float
done = "my_pkg.rewards:map_exit"       # fn(gb: GameBoy) -> bool
max_steps = 0                 # 0 = unlimited
```

Then:

```python
from rlgb import GameBoyEnv

env = GameBoyEnv(config="config.toml")
env.reset()
obs, reward, terminated, truncated, info = env.step(0)
```

### 8. SQLite Persistence

```python
from rlgb import GameBoy, EmuDB

db = EmuDB("runs.db")
gb = GameBoy("game.gb")

# Save a named state
db.save_state(gb, rom_path="game.gb", name="start", tag="initial")

# Later: load it back
state_blob = db.load_state("game.gb", name="start")
gb.load_state(state_blob)

# Track a training run
run_id = db.start_run(gb, config="config.toml", note="test run")
db.log_episode(run_id, 0, actions, rewards, gb)
db.finish_run(run_id)
```

## API Reference

### GameBoy

```python
gb = GameBoy(rom_path="game.gb", config="config.toml", **overrides)
```

- `tick(frames)` — advance N frames
- `save_state()` → bytes — full machine snapshot
- `load_state(bytes)` — restore snapshot
- `memory` — Memory object (bus access, see below)
- `registers` — Registers object (CPU register access, see below)
- `screen` → np.ndarray (144×160 uint8, 0–3 shades)
- `screen_gray` → np.ndarray (144×160 uint8, 0–255)
- `screen_rgb` → np.ndarray (144×160×3 uint8)
- `set_buttons(name)` — press button(s): "a", "b", "start", "select", "up", "down", "left", "right", or ""
- `release()` — release all buttons
- `push(name)` — press for `hold_frames`, release for `release_frames` (configured in TOML)
- `close()` — clean up resources

### Memory

```python
mem = gb.memory

mem[0xC000]                # int (0–255)
mem[0xC000:0xC100]         # bytes
mem[0xC000] = 42
mem[0xC000:0xC004] = b"\x01\x02\x03\x04"
```

- Full 64 KiB address space (0x0000–0xFFFF)
- Any read/write is valid (echoing, I/O, VRAM all mirror as the hardware does)

### Registers

```python
reg = gb.registers

reg.pc, reg.sp            # 16-bit
reg.a, reg.f, reg.b, ...  # 8-bit
reg.af, reg.bc, reg.de, reg.hl  # 16-bit pairs
reg.ime                   # interrupt master enable
reg.halted                # halted state
```

### GameBoyEnv

Gymnasium-compatible RL wrapper. Configuration via TOML or kwargs.

```python
env = GameBoyEnv(rom_path="game.gb", config="config.toml")

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

- `action_space` — Discrete(len(actions))
- `observation_space` — Box with shape (144, 160, 3) for rgb, (144, 160) for gray/shades
- Custom rewards/done logic via TOML callback functions

### VecGameBoyEnv

Vectorized environment—N emulators in parallel.

```python
env = VecGameBoyEnv(n=16, rom_path="game.gb", workers=8)

obs, infos = env.reset()
obs, rewards, terminateds, truncateds, infos = env.step(actions)
```

- `n` — number of parallel environments
- `workers` — thread pool size (default: min(n, cpu_count))
- Returns stacked numpy arrays (except infos, which is a list)
- Auto-resets finished episodes; final obs/info in `info["final_obs"]` / `info["final_info"]`

### EmuDB

SQLite persistence for states, runs, and episodes.

```python
db = EmuDB("runs.db")

# Save state
db.save_state(gb, rom_path="game.gb", name="checkpoint", tag="level_5")

# Load state
blob = db.load_state("game.gb", name="checkpoint")
gb.load_state(blob)

# Track training
run_id = db.start_run(gb, config="config.toml", note="training run")
db.log_episode(run_id, episode_idx, actions, rewards, gb)
db.finish_run(run_id)

# Query metrics
metrics = db.get_metrics(run_id, "episode_reward")
```

## Performance

- **Headless (render=false)**: ~3,200 FPS on bare metal (single core), on the
  author's machine. The figure is CPU-bound and varies with rendering settings
  and hardware — benchmark your own setup.

## Configuration

All behavior is declared in `config.toml` (or passed as kwargs). Built-in defaults ensure everything works out of the box.

### `[rom]`

- `path` — ROM file path (required if not passed to constructor)
- `autoload` — save format to auto-load on init: "sav", "state", "both", or "none"
- `save_format` — format for `save()` / `load()`: "sav" or "state"

### `[emulation]`

- `render` — boolean (default: true). False skips all pixel work for max speed.
- `frame_skip` — extra frames per `tick()` (default: 0)
- `sound` — boolean (default: false). APU is stubbed; reserved for future use.

### `[video]`

- `palette` — 4×3 RGB array for shades 0–3 (default: classic DMG green)
- `grayscale_levels` — 4 intensity values 0–255 for gray mode

### `[input]`

- `hold_frames` — frames button is held in `push()` (default: 5)
- `release_frames` — frames after release in `push()` (default: 1)

### `[env]` (Gymnasium wrapper)

- `actions` — list of button names; "" = no-op
- `obs` — observation mode: "shades", "gray", or "rgb"
- `frames_per_action` — frames advanced per step (default: 24)
- `press_frames` — of those frames, how many the button is held (default: 8)
- `reward` — dotted path to reward function: `"my_pkg.rewards:fn"`
- `done` — dotted path to done function
- `info` — dotted path to info function
- `max_steps` — episode length (0 = unlimited)

### `[db]`

- `path` — sqlite file for EmuDB ("" = not used)
- `compress` — zlib level for state blobs, 0–9 (default: 6)

## Use Cases

1. **Behavior Cloning**: Record expert trajectories with `EmuDB.log_episode()`, train imitation policies
2. **Reinforcement Learning**: Plug into Stable Baselines 3 for DQN, PPO, A3C, etc.
3. **ROM Hacking & Research**: Full memory read/write, register access, deterministic states
4. **Speedrunning Analysis**: Extract game state, frame-perfect input research
5. **Dataset Generation**: Generate large RL datasets at ~3,200 fps per machine

## License

CC BY-NC-ND 4.0 license. Copyright (c) 2026 BuiltNotTaught.

## Building from Source

### Requirements

- C compiler (gcc, clang)
- Python ≥ 3.10
- numpy

### Build

```bash
make                # builds rlgb/libgb.so
pip install -e .    # editable install
```

### Optimization

```bash
CC=clang make                    # use clang
make ARCH="-march=znver3"        # target specific CPU
```

## Notes

- No ROM files included. Provide your own or use emulation-legal test ROMs
- This is a **from-scratch** emulator—no code copied from other projects
- The C core is deterministic; Python is the only source of non-determinism (timestamp logging)
- Supports battery-backed cart RAM (MBC1/2/3/5). `.gbc` images load but run in **DMG mode only** — no CGB color, double-speed, or CGB-exclusive behavior
- APU (sound) is a register stub; not implemented yet
