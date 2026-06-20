import argparse
import csv
import os
from typing import List

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from env_utils import make_env


DEFAULT_OUTPUT_DIR = "comparison_reports"
DEFAULT_VIDEO_FOLDER = "comparison_reports/videos"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple BipedalWalker PPO models")
    parser.add_argument(
        "--model-paths",
        nargs="+",
        required=True,
        help="Paths to trained PPO models to compare (directories or .zip files)",
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
        help="Number of evaluation episodes for each model",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for comparison reports",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record videos for each evaluation model",
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


def evaluate_model(model_path: str, mode: str, eval_episodes: int, record_video: bool, video_folder: str):
    model_path = resolve_model_path(model_path)
    print(f"Evaluating model '{model_path}' on {mode} mode for {eval_episodes} episodes...")
    model = PPO.load(model_path)

    os.makedirs(video_folder, exist_ok=True) if record_video else None
    env = make_env(
        env_name=get_env_name(mode),
        hardcore=(mode == "hardcore"),
        render_mode="rgb_array" if record_video else "human",
        record_video=record_video,
        video_folder=video_folder,
        use_monitor=False,
        norm_obs=False,
        norm_reward=False,
    )

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=eval_episodes,
        return_episode_rewards=False,
    )
    env.close()

    print(f"Result for {model_path}: mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}")
    return mean_reward, std_reward


def write_csv(output_dir: str, rows: List[List]):
    csv_path = os.path.join(output_dir, "comparison_results.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label", "model_path", "mode", "eval_episodes", "mean_reward", "std_reward"])
        writer.writerows(rows)
    print(f"Saved comparison CSV: {csv_path}")
    return csv_path


def write_markdown(output_dir: str, rows: List[List]):
    md_path = os.path.join(output_dir, "comparison_report.md")
    with open(md_path, mode="w", encoding="utf-8") as md_file:
        md_file.write("# Bipedal Walker Model Comparison Report\n\n")
        md_file.write("## Summary\n\n")
        md_file.write("| Label | Model Path | Mode | Episodes | Mean Reward | Std Reward |\n")
        md_file.write("|---|---|---|---|---|---|\n")
        for row in rows:
            md_file.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]:.2f} | {row[5]:.2f} |\n")
        md_file.write("\n")
        md_file.write("## Notes\n\n")
        md_file.write("- Each model is evaluated independently for the same number of episodes.\n")
        md_file.write("- If video recording is enabled, each model's videos are saved under the configured video folder.\n")
    print(f"Saved comparison Markdown report: {md_path}")
    return md_path


def plot_results(output_dir: str, rows: List[List]):
    labels = [row[0] for row in rows]
    mean_rewards = [row[4] for row in rows]
    std_rewards = [row[5] for row in rows]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, mean_rewards, yerr=std_rewards, capsize=8, color="#4c72b0")
    plt.xlabel("Model")
    plt.ylabel("Mean Reward")
    plt.title("Bipedal Walker Model Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "comparison_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved comparison plot: {plot_path}")
    return plot_path


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.record_video:
        os.makedirs(args.video_folder, exist_ok=True)

    labels = args.labels if args.labels else [os.path.basename(path) for path in args.model_paths]
    if len(labels) != len(args.model_paths):
        raise ValueError("Number of labels must match number of model paths if labels are provided.")

    rows = []
    for label, model_path in zip(labels, args.model_paths):
        mean_reward, std_reward = evaluate_model(
            model_path=model_path,
            mode=args.mode,
            eval_episodes=args.eval_episodes,
            record_video=args.record_video,
            video_folder=args.video_folder,
        )
        rows.append([label, model_path, args.mode, args.eval_episodes, mean_reward, std_reward])

    write_csv(args.output_dir, rows)
    write_markdown(args.output_dir, rows)
    plot_results(args.output_dir, rows)
    print("Model comparison complete.")


if __name__ == "__main__":
    main()
