"""All emu-vec tunables in one place."""
from dataclasses import dataclass, field
from typing import List, Optional

# Per-env live-telemetry channels shipped back through shared memory alongside
# (reward, done) so the training process can log a full live monitor without
# env_method. Order is the on-wire order; each field is one float32. Keep in
# sync with worker.py's _telemetry() and EmuMonitorCallback.
#
# Two groups: game-progress state (log max = best lane + mean = typical) and
# the per-episode reward-component breakdown (log mean = what's driving reward).
TELEMETRY_FIELDS = (
    # --- game progress ---
    "badges",          # gym badges earned (popcount of 0xD356)
    "level_sum",       # summed party Pokémon levels
    "party_size",      # number of Pokémon in party (0xD163)
    "hp_frac",         # party HP fraction 0..1
    "events",          # event flags set (story progress, 0xD747..0xD886)
    "maps_explored",   # distinct map ids seen this episode
    "tiles_explored",  # distinct (map,x,y) tiles seen this episode (exploration)
    # --- per-episode reward breakdown ---
    "r_badge",         # cumulative reward from badges
    "r_explore",       # cumulative reward from new maps/tiles (net of revisit)
    "r_level",         # cumulative reward from level-ups
    "r_step",          # cumulative per-step penalty
)


@dataclass
class GBVecConfig:
    # ROM / curriculum — no default, always pass explicitly
    rom_path: str = ""
    init_states: Optional[List[bytes]] = None   # one per worker; None → fresh boot

    # Parallelism
    n_workers: int = 4
    envs_per_worker: int = 3          # total envs = n_workers * envs_per_worker

    # Shared memory sizing
    screen_shape: tuple = (4, 84, 84) # FRAME_STACK × H × W  uint8
    ram_size: int = 18                # bytes in RAM obs vector
    n_actions: int = 8                # Discrete(8)

    # GPU batching
    max_batch_wait_ms: float = 2.0    # accumulate ready envs up to this long
    device: str = "auto"

    # Frame skip (must match GBEnv / env.py)
    frame_skip: int = 4

    # Blackout detection (mirrors env.py)
    blackout_threshold: int = 10
    blackout_steps: int = 60

    # Max episode steps (mirrors env.py)
    max_episode_steps: int = 20_480

    # Reward shaping — mirrors pokeai/config.py, all overridable
    reward_badge:        float =  5.0
    reward_new_map:      float =  0.1
    reward_level_up:     float =  0.05
    reward_champion:     float = 20.0
    penalty_per_step:    float = -0.001
    reward_new_tile:     float =  0.02
    penalty_map_revisit: float = -0.5
    map_revisit_limit:   int   =  3
    map_revisit_window:  int   =  300

    @property
    def n_envs(self) -> int:
        return self.n_workers * self.envs_per_worker

    @property
    def obs_screen_bytes(self) -> int:
        s = self.screen_shape
        return s[0] * s[1] * s[2]       # uint8

    @property
    def obs_total_bytes(self) -> int:
        return self.obs_screen_bytes + self.ram_size

    @property
    def telemetry_bytes(self) -> int:
        return len(TELEMETRY_FIELDS) * 4      # one float32 per field
