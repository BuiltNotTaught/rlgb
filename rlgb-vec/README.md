# rlgb-vec

Plug-and-play, GPU-async VecEnv that lets **SB3** train on the vendored **rlgb**
emulator. DMG (original Game Boy); the worker bakes in the Pokémon-Red WRAM map for reward/telemetry.

Runs **barebone** (local venv) or **Docker** (CUDA GPU). The binding, shared-memory
IPC and obs-decode layers run with just numpy+ctypes; the training stack (torch +
stable-baselines3 + sb3-contrib) is the `[train]` extra.

## Plug & play
```python
from rlgb_vec import make_env
from stable_baselines3 import PPO

vec   = make_env("/roms/game.gb", n_envs=12)   # ready SB3 VecEnv, one line
model = PPO("MultiInputPolicy", vec)             # Dict obs -> MultiInput policy
model.learn(1_000_000)
vec.close()
```
That is the whole wiring — `GBVecEnv` is a standard SB3 `VecEnv`, so SB3's own
policy forward pass is the GPU inference; there is nothing else to hook up.
Recurrent policies: `from sb3_contrib import RecurrentPPO; RecurrentPPO("MultiInputLstmPolicy", vec)`.

Or straight from the shell:
```bash
python -m rlgb_vec.train --rom /roms/game.gb --n-envs 12         # PPO
python -m rlgb_vec.train --rom /roms/game.gb --recurrent         # RecurrentPPO
```

## Barebone setup
```bash
./build.sh                      # compile vendor/rlgb -> libgb.so
python -m venv .venv && . .venv/bin/activate
pip install -e '.[train]'       # full stack  (or: pip install -e .  for core only)
python -m pytest tests -q       # smoke tests (no GPU/ROM needed)
```

## Docker (CUDA GPU)
```bash
docker build -t rlgb-vec .
docker run --gpus all -v /path/to/roms:/roms rlgb-vec \
    python3 -m rlgb_vec.train --rom /roms/game.gb
```
ROMs are never baked into the image — mount them at runtime.

## Layout
- `rlgb_vec/`        — bridge: `make_env`/`adapter`, async `GBVecEnv`, GPU worker, shm IPC, `bindings.py` (over rlgb's C API), `train.py`
- `vendor/rlgb/` — vendored emulator source, compiled to `vendor/rlgb/rlgb/libgb.so`
- `Dockerfile` · `build.sh` · `tests/` (shm + binding + wiring)
