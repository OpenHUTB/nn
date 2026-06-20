import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env_utils import make_env

MODEL_DIR = "models"
LOGS_DIR = "logs"
VIDEO_DIR = "videos"


def parse_args():
    parser = argparse.ArgumentParser(description="Bipedal Walker PPO runner")

    parser.add_argument(
        "--task",
        choices=["train", "eval"],
        required=True,
        help="Task to run: train a PPO model or evaluate a saved model",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "hardcore"],
        default="normal",
        help="Environment mode",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--model-name",
        default="ppo_bipedalwalker",
        help="Model save name for training (without extension)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a saved model for evaluation",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record a video during training or evaluation",
    )
    parser.add_argument(
        "--video-folder",
        default=VIDEO_DIR,
        help="Video folder for recording output",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for PPO training",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps to run for each environment update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for PPO",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    return parser.parse_args()


def get_env_name(mode: str) -> str:
    return "BipedalWalkerHardcore-v3" if mode == "hardcore" else "BipedalWalker-v3"


def train(args):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(args.video_folder, exist_ok=True)

    env = make_env(
        env_name=get_env_name(args.mode),
        hardcore=(args.mode == "hardcore"),
        render_mode="rgb_array" if args.record_video else None,
        record_video=args.record_video,
        video_folder=args.video_folder,
        use_monitor=True,
        logs_dir=LOGS_DIR,
        norm_obs=True,
        norm_reward=True,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
    )

    print(f"Starting training: mode={args.mode}, timesteps={args.timesteps}")
    model.learn(total_timesteps=args.timesteps)

    model_path = os.path.join(MODEL_DIR, f"{args.model_name}")
    model.save(model_path)
    print(f"Saved model to: {model_path}")

    env.close()

    if args.record_video:
        print(f"Recorded videos saved in: {args.video_folder}")


def evaluate(args):
    if args.model_path is None:
        raise ValueError("--model-path is required for evaluation")

    model = PPO.load(args.model_path)
    env = make_env(
        env_name=get_env_name(args.mode),
        hardcore=(args.mode == "hardcore"),
        render_mode="rgb_array" if args.record_video else "human",
        record_video=args.record_video,
        video_folder=args.video_folder,
        use_monitor=False,
        norm_obs=False,
        norm_reward=False,
    )

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.eval_episodes,
        return_episode_rewards=False,
    )

    env.close()

    print(f"Evaluation results: mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}")
    if args.record_video:
        print(f"Recorded evaluation videos saved in: {args.video_folder}")


def main():
    args = parse_args()

    if args.task == "train":
        train(args)
    elif args.task == "eval":
        evaluate(args)
    else:
        raise ValueError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    main()
