import argparse
import os
import csv
from typing import List

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from env_utils import make_env


DEFAULT_OUTPUT_DIR = "comparison_reports"
DEFAULT_VIDEO_FOLDER = "comparison_reports/videos"


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate models and draw boxplots of episode rewards")
    parser.add_argument(
        "--model-paths",
        nargs="+",
        required=True,
        help="Paths to trained PPO models to evaluate (directories or .zip files)",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for each model path, in the same order",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "hardcore"],
        default="normal",
        help="Environment mode for evaluation",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes per trial",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of repeated trials (each trial runs eval-episodes episodes)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for boxplot and CSV",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record videos during evaluations",
    )
    parser.add_argument(
        "--video-folder",
        default=DEFAULT_VIDEO_FOLDER,
        help="Folder to save recorded videos",
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


def collect_rewards_for_model(model_path: str, mode: str, eval_episodes: int, trials: int, record_video: bool, video_folder: str):
    model_path = resolve_model_path(model_path)
    model = PPO.load(model_path)

    all_rewards = []
    for t in range(trials):
        env = make_env(
            env_name=get_env_name(mode),
            hardcore=(mode == "hardcore"),
            render_mode="rgb_array" if record_video else None,
            record_video=record_video,
            video_folder=video_folder,
            use_monitor=False,
            norm_obs=False,
            norm_reward=False,
        )

        episode_rewards, _ = evaluate_policy(model, env, n_eval_episodes=eval_episodes, return_episode_rewards=True)
        # episode_rewards is a list of floats
        all_rewards.extend(list(episode_rewards))
        env.close()

    return all_rewards


def write_csv(output_dir: str, rows: List[List]):
    csv_path = os.path.join(output_dir, "batch_boxplot_results.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label", "model_path", "reward"])
        for row in rows:
            label, model_path, rewards = row
            for r in rewards:
                writer.writerow([label, model_path, f"{r:.6f}"])
    print(f"Saved CSV: {csv_path}")
    return csv_path


def plot_boxplot(output_dir: str, labels: List[str], rewards_lists: List[List[float]]):
    plt.figure(figsize=(max(6, len(labels) * 1.5), 6))
    plt.boxplot(rewards_lists, labels=labels, showmeans=True)
    plt.ylabel("Episode Reward")
    plt.title("Model Reward Distribution (boxplot)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "batch_boxplot.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved boxplot: {out_path}")
    return out_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.record_video:
        os.makedirs(args.video_folder, exist_ok=True)

    labels = args.labels if args.labels else [os.path.basename(p) for p in args.model_paths]
    if len(labels) != len(args.model_paths):
        raise ValueError("Number of labels must match number of model paths if provided")

    rows = []
    rewards_lists = []
    for label, model_path in zip(labels, args.model_paths):
        print(f"Collecting rewards for model {label} ({model_path})...")
        rewards = collect_rewards_for_model(model_path, args.mode, args.eval_episodes, args.trials, args.record_video, args.video_folder)
        rows.append([label, model_path, rewards])
        rewards_lists.append(rewards)

    write_csv(args.output_dir, rows)
    plot_boxplot(args.output_dir, labels, rewards_lists)


if __name__ == "__main__":
    main()
