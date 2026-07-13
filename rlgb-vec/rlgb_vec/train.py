"""Runnable trainer — the plug-and-play entry point.

    python -m rlgb_vec.train --rom /roms/pokemon_red.gb
    python -m rlgb_vec.train --rom /roms/game.gb --n-envs 24 --steps 5e6 --recurrent

Needs the training stack:  pip install -e '.[train]'
"""
import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train an SB3 policy on the vendored emulator.")
    ap.add_argument("--rom", required=True, help="path to the ROM")
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--envs-per-worker", type=int, default=None)
    ap.add_argument("--steps", type=float, default=1e6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--recurrent", action="store_true", help="RecurrentPPO (needs sb3-contrib)")
    ap.add_argument("--init-state", default=None, help="optional .state for curriculum reset")
    args = ap.parse_args(argv)

    from rlgb_vec.adapter import make_env

    init = open(args.init_state, "rb").read() if args.init_state else None
    vec = make_env(
        args.rom,
        n_envs=args.n_envs,
        envs_per_worker=args.envs_per_worker,
        device=args.device,
        init_state=init,
    )
    try:
        if args.recurrent:
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO("MultiInputLstmPolicy", vec, verbose=1, device=args.device)
        else:
            from stable_baselines3 import PPO
            model = PPO("MultiInputPolicy", vec, verbose=1, device=args.device)
        model.learn(total_timesteps=int(args.steps))
    finally:
        vec.close()


if __name__ == "__main__":
    main()
