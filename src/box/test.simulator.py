import argparse
import importlib
import importlib.util
import os
import sys
import traceback


def check_runtime_dependencies() -> bool:
    """Validate minimal runtime dependencies before importing the simulator."""
    required_modules = ("numpy", "gymnasium", "mujoco", "yaml")
    missing = []

    for name in required_modules:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)

    if not missing:
        return True

    print("Missing dependencies:")
    for name in missing:
        print(f"  - {name}")
    print("\nInstall them with:")
    print("  pip install -r requirements.txt")
    return False


def load_simulator_module(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"simulator.py not found: {path}")

    spec = importlib.util.spec_from_file_location("simulator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a smoke test of the arm Simulator")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatability")
    parser.add_argument("--render", action="store_true", help="Enable human rendering (pygame)")
    parser.add_argument("--sim-folder", type=str, default=None, help="Simulator folder (override default)")
    parser.add_argument("--log-interval", type=int, default=50, help="How often to print status")
    args = parser.parse_args()

    if not check_runtime_dependencies():
        return 2

    import numpy as np

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_path = os.path.join(script_dir, "simulator.py")
    module = load_simulator_module(sim_path)
    Simulator = getattr(module, "Simulator")

    sim_folder = args.sim_folder or os.path.join(script_dir, "simulators", "arm_simulation")
    os.makedirs(sim_folder, exist_ok=True)

    print("=" * 50)
    print("Mechanical arm simulator smoke test")
    print(f"simulator folder: {sim_folder}")

    np.random.seed(args.seed)

    env = None
    try:
        env = Simulator.get(simulator_folder=sim_folder, render_mode=("human" if args.render else None))
        print("Environment created")
        print(f"nq={env.model.nq} nu={env.model.nu} nv={env.model.nv}")

        obs, _ = env.reset(seed=args.seed)
        print(f"initial obs shape: {obs.shape}")

        for step in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)

            if step % args.log_interval == 0:
                print(f"Step {step:4d} | reward={reward:.3f} terminated={terminated} truncated={truncated}")

            if terminated or truncated:
                print(f"Episode ended at step {step}, resetting environment")
                obs, _ = env.reset()

        print("Simulation finished")
        return 0

    except KeyboardInterrupt:
        print("Interrupted by user")
        return 130
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
