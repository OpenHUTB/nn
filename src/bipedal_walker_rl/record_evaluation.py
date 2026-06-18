import argparse
import os
import csv
from stable_baselines3 import PPO
from env_utils import make_env

def parse_args():
    parser = argparse.ArgumentParser(description="Record evaluation episodes for a trained BipedalWalker PPO model")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the saved PPO model file (.zip or directory)"
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "hardcore"],
        default="normal",
        help="Environment mode to run evaluation in"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_reports",
        help="Directory where evaluation reports and videos will be stored"
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record rendered video for the evaluation episodes"
    )
    parser.add_argument(
        "--video-folder",
        default="evaluation_reports/videos",
        help="Video output folder when recording is enabled"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions during evaluation"
    )
    return parser.parse_args()


def resolve_model_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "data")):
        return path
    if os.path.isfile(path + ".zip"):
        return path + ".zip"
    raise FileNotFoundError(f"Model not found at '{path}' or '{path}.zip'.")


def get_env_name(mode: str) -> str:
    return "BipedalWalkerHardcore-v3" if mode == "hardcore" else "BipedalWalker-v3"


def evaluate_and_record(model_path: str, env_mode: str, episodes: int, output_dir: str, record_video: bool, video_folder: str, deterministic: bool):
    os.makedirs(output_dir, exist_ok=True)
    if record_video:
        os.makedirs(video_folder, exist_ok=True)

    model_path = resolve_model_path(model_path)
    model = PPO.load(model_path)

    env = make_env(
        env_name=get_env_name(env_mode),
        hardcore=(env_mode == "hardcore"),
        render_mode="rgb_array" if record_video else "human",
        record_video=record_video,
        video_folder=video_folder,
        use_monitor=False,
        norm_obs=False,
        norm_reward=False,
    )

    episode_results = []
    obs = env.reset()
    for episode in range(1, episodes + 1):
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0] if isinstance(reward, (list, tuple)) else reward
            step += 1

        episode_results.append({
            "episode": episode,
            "reward": float(total_reward),
            "steps": step,
        })
        obs = env.reset()

    env.close()

    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["episode", "reward", "steps"])
        writer.writeheader()
        writer.writerows(episode_results)

    print(f"Saved evaluation summary to: {csv_path}")
    return episode_results


def main():
    args = parse_args()
    episode_results = evaluate_and_record(
        model_path=args.model_path,
        env_mode=args.mode,
        episodes=args.episodes,
        output_dir=args.output_dir,
        record_video=args.record_video,
        video_folder=args.video_folder,
        deterministic=args.deterministic,
    )

    for result in episode_results:
        print(f"Episode {result['episode']}: reward={result['reward']:.2f}, steps={result['steps']}")


if __name__ == "__main__":
    main()
