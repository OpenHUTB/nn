import sys
import os
import argparse
import importlib.util
import traceback
import numpy as np


def load_simulator_module(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"simulator.py not found: {path}")
    spec = importlib.util.spec_from_file_location("simulator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Run a short smoke test of the Simulator")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatability")
    parser.add_argument("--render", action="store_true", help="Enable human rendering (pygame)")
    parser.add_argument("--sim-folder", type=str, default=None, help="Simulator folder (overrides default)")
    parser.add_argument("--log-interval", type=int, default=50, help="How often to print status")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_path = os.path.join(script_dir, "simulator.py")
    module = load_simulator_module(sim_path)
    Simulator = getattr(module, "Simulator")

    sim_folder = args.sim_folder or os.path.join(script_dir, "simulators", "arm_simulation")
    os.makedirs(sim_folder, exist_ok=True)

    print("=" * 50)
    print("Mechanical arm simulator smoke test")
    print(f"simulator folder: {sim_folder}")

    # seed for reproducible samples
    np.random.seed(args.seed)

    try:
        env = Simulator.get(simulator_folder=sim_folder, render_mode=("human" if args.render else None))
        print("Environment created")
        print(f"nq={env.model.nq} nu={env.model.nu} nv={env.model.nv}")

        obs, info = env.reset(seed=args.seed)
        print(f"initial obs shape: {obs.shape}")

        for step in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if step % args.log_interval == 0:
                print(f"Step {step:4d} | reward={reward:.3f} terminated={terminated} truncated={truncated}")

            if terminated or truncated:
                print(f"Terminated at step {step}, resetting environment")
                obs, info = env.reset()

        print("Simulation finished")

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        try:
            if 'env' in locals():
                env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
