import argparse
import os
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env_utils import make_env


DEFAULT_NORMAL_MODEL = "models/ppo_bipedalwalker"
DEFAULT_HARDCORE_MODEL = "models/ppo_bipedalwalker_hardcore"
DEFAULT_OUTPUT_DIR = "reports"
DEFAULT_VIDEO_FOLDER = "reports/videos"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark BipedalWalker PPO models")
    parser.add_argument(
        "--normal-model-path",
        default=DEFAULT_NORMAL_MODEL,
        help="Path to the trained normal BipedalWalker model (without .zip extension)",
    )
    parser.add_argument(
        "--hardcore-model-path",
        default=DEFAULT_HARDCORE_MODEL,
        help="Path to the trained hardcore BipedalWalker model (without .zip extension)",
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
        help="Directory where benchmark reports will be saved",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record video during benchmark evaluation",
    )
    parser.add_argument(
        "--video-folder",
        default=DEFAULT_VIDEO_FOLDER,
        help="Folder for benchmark video recordings",
    )
    return parser.parse_args()


def _resolve_model_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isfile(path + ".zip"):
        return path + ".zip"
    raise FileNotFoundError(
        f"Model not found at '{path}' or '{path}.zip'. Please train the model first or provide the correct path."
    )


def evaluate_model(model_path: str, mode: str, eval_episodes: int, record_video: bool, video_folder: str):
    print(f"Evaluating model '{model_path}' on {mode} mode for {eval_episodes} episodes...")
    model_path = _resolve_model_path(model_path)
    model = PPO.load(model_path)

    env_name = "BipedalWalkerHardcore-v3" if mode == "hardcore" else "BipedalWalker-v3"
    render_mode = "rgb_array" if record_video else "human"
    env = make_env(
        env_name=env_name,
        hardcore=(mode == "hardcore"),
        render_mode=render_mode,
        record_video=record_video,
        video_folder=video_folder,
        use_monitor=False,
        norm_obs=False,
        norm_reward=False,
    )

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes, return_episode_rewards=False)
    env.close()

    print(f"Result for {mode}: mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}")
    return mean_reward, std_reward


def write_csv(output_dir: str, rows):
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["mode", "model_path", "eval_episodes", "mean_reward", "std_reward"])
        writer.writerows(rows)
    print(f"Saved benchmark CSV: {csv_path}")
    return csv_path


def write_markdown(output_dir: str, rows):
    md_path = os.path.join(output_dir, "benchmark_report.md")
    with open(md_path, mode="w", encoding="utf-8") as f:
        f.write("# Bipedal Walker Benchmark Report\n\n")
        f.write("## Summary\n\n")
        f.write("| Mode | Model Path | Episodes | Mean Reward | Std Reward |\n")
        f.write("|---|---|---|---|---|\n")
        for row in rows:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.2f} | {row[4]:.2f} |\n")
        f.write("\n")
        f.write("## Notes\n\n")
        f.write("- The benchmark evaluates each model for the specified number of episodes.\n")
        f.write("- If `record_video` is enabled, evaluation videos are saved under the video folder.\n")
    print(f"Saved benchmark Markdown report: {md_path}")
    return md_path


def run_benchmark(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.record_video:
        os.makedirs(args.video_folder, exist_ok=True)

    results = []
    normal_result = evaluate_model(
        args.normal_model_path,
        mode="normal",
        eval_episodes=args.eval_episodes,
        record_video=args.record_video,
        video_folder=args.video_folder,
    )
    results.append(["normal", args.normal_model_path, args.eval_episodes, normal_result[0], normal_result[1]])

    hardcore_result = evaluate_model(
        args.hardcore_model_path,
        mode="hardcore",
        eval_episodes=args.eval_episodes,
        record_video=args.record_video,
        video_folder=args.video_folder,
    )
    results.append(["hardcore", args.hardcore_model_path, args.eval_episodes, hardcore_result[0], hardcore_result[1]])

    write_csv(args.output_dir, results)
    write_markdown(args.output_dir, results)

    print("Benchmark complete.")


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
