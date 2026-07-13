# rlgb-vec — async GPU VecEnv bridge for the vendored rlgb (DMG) emulator.
# CUDA runtime base (GTX 1080 / compute 6.1 and up, per emu-vec spec).
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (layer cache). CUDA 12.1 torch wheels + training stack.
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch \
 && pip3 install --no-cache-dir "stable-baselines3>=2.0" "opencv-python-headless>=4.7" "numpy>=1.24" pytest

# Vendored emulator + bridge package.
COPY vendor/ ./vendor/
COPY rlgb_vec/ ./rlgb_vec/
COPY pyproject.toml build.sh ./
COPY tests/ ./tests/

# Build the emulator core (libgb.so) and install the bridge.
RUN ARCH="-mtune=generic" bash build.sh && pip3 install --no-cache-dir -e .

# ROMs are provided at runtime (mount a volume); none are baked into the image.
#   docker run --gpus all -v /path/to/roms:/roms rlgb-vec python3 -m rlgb_vec.train --rom /roms/game.gb
CMD ["python3", "-c", "import rlgb_vec; print('rlgb-vec ready')"]
