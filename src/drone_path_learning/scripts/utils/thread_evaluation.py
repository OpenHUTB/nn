import argparse
import logging
import math
import os
from configparser import ConfigParser
from pathlib import Path

import cv2
import gym
import gym_env
import numpy as np
from PyQt5 import QtCore
from stable_baselines3 import PPO, SAC, TD3
from tqdm import tqdm

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def rule_based_policy(obs):
    """Simple rule-based policy for LGMD comparison."""
    obs = np.squeeze(obs, axis=0)

    for i in range(5):
        obs[i] = obs[i] / 2 + 0.5

    obs_weight = np.array([1.0, 3.0, 3.0, -3.0, -1.0, 3.0])
    action_sum = np.sum(obs * obs_weight)
    action_sum = np.clip(action_sum, -math.radians(40), math.radians(40))

    return np.array([action_sum])


class EvaluateThread(QtCore.QThread):
    def __init__(
        self,
        eval_path,
        config,
        model_file,
        eval_ep_num,
        eval_env=None,
        eval_dynamics=None,
    ):
        super().__init__()
        logger.info("Initializing evaluation thread")

        self.cfg = ConfigParser()
        if not self.cfg.read(config):
            raise FileNotFoundError(f"Config file not found or unreadable: {config}")

        if eval_env is not None:
            self.cfg.set("options", "env_name", eval_env)

        if eval_env == "NH_center":
            self.cfg.set("environment", "accept_radius", str(1))

        if eval_dynamics is not None:
            self.cfg.set("options", "dynamic_name", eval_dynamics)

        self.env = gym.make("airsim-env-v0")
        self.env.set_config(self.cfg)

        self.eval_path = eval_path
        self.model_file = model_file
        self.eval_ep_num = eval_ep_num
        self.eval_env = self.cfg.get("options", "env_name")
        self.eval_dynamics = self.cfg.get("options", "dynamic_name")

    def terminate(self):
        logger.info("Evaluation terminated")

    def run(self):
        self.results = self.run_drl_model()

    def run_drl_model(self):
        logger.info("Start evaluation")
        algo = self.cfg.get("options", "algo")
        if algo == "TD3":
            model = TD3.load(self.model_file, env=self.env)
        elif algo == "SAC":
            model = SAC.load(self.model_file, env=self.env)
        elif algo == "PPO":
            model = PPO.load(self.model_file, env=self.env)
        else:
            raise ValueError(f"Unsupported algo for evaluation: {algo}")
        self.env.model = model

        obs = self.env.reset()
        episode_num = 0
        reward_sum = np.array([0.0])
        episode_successes = []
        episode_crashes = []
        traj_list_all = []
        action_list_all = []
        state_list_all = []
        obs_list_all = []

        traj_list = []
        action_list = []
        state_raw_list = []
        step_num_list = []
        obs_list = []

        cv2.waitKey(1)

        while episode_num < self.eval_ep_num:
            unscaled_action, _ = model.predict(obs, deterministic=True)

            new_obs, reward, done, info = self.env.step(unscaled_action)
            traj_list.append(self.env.dynamic_model.get_position())
            action_list.append(unscaled_action)
            state_raw_list.append(self.env.dynamic_model.state_raw)
            obs_list.append(obs)

            obs = new_obs
            reward_sum[-1] += reward

            if done:
                episode_num += 1
                maybe_is_success = info.get("is_success")
                maybe_is_crash = info.get("is_crash")
                logger.info(
                    "Episode %d | reward=%.4f | success=%s",
                    episode_num,
                    reward_sum[-1],
                    maybe_is_success,
                )
                episode_successes.append(float(maybe_is_success))
                episode_crashes.append(float(maybe_is_crash))
                reward_sum = np.append(reward_sum, 0.0)
                obs = self.env.reset()
                if info.get("is_success"):
                    traj_list.append(1)
                    action_list.append(1)
                    step_num_list.append(info.get("step_num"))
                elif info.get("is_crash"):
                    traj_list.append(2)
                    action_list.append(2)
                else:
                    traj_list.append(3)
                    action_list.append(3)

                traj_list_all.append(traj_list)
                action_list_all.append(action_list)
                state_list_all.append(state_raw_list)
                obs_list_all.append(obs_list)
                traj_list = []
                action_list = []
                state_raw_list = []
                obs_list = []

        eval_folder = os.path.join(
            self.eval_path,
            f"eval_{self.eval_ep_num}_{self.eval_env}_{self.eval_dynamics}",
        )
        os.makedirs(eval_folder, exist_ok=True)
        np.save(os.path.join(eval_folder, "traj_eval"), np.array(traj_list_all, dtype=object))
        np.save(os.path.join(eval_folder, "action_eval"), np.array(action_list_all, dtype=object))
        np.save(os.path.join(eval_folder, "state_eval"), np.array(state_list_all, dtype=object))
        np.save(os.path.join(eval_folder, "obs_eval"), np.array(obs_list_all, dtype=object))

        avg_reward = reward_sum[: self.eval_ep_num].mean()
        success_rate = np.mean(episode_successes)
        crash_rate = np.mean(episode_crashes)
        avg_success_steps = np.mean(step_num_list) if step_num_list else float("nan")

        logger.info(
            "Average reward=%.4f | success_rate=%.4f | crash_rate=%.4f | avg_success_steps=%s",
            avg_reward,
            success_rate,
            crash_rate,
            avg_success_steps,
        )

        results = [avg_reward, success_rate, crash_rate, avg_success_steps]
        np.save(os.path.join(eval_folder, "results"), np.array(results))

        return results

    def run_rule_policy(self):
        obs = self.env.reset()
        episode_num = 0
        reward_sum = np.array([0.0])
        while episode_num < self.eval_ep_num:
            unscaled_action = rule_based_policy(obs)
            new_obs, reward, done, info = self.env.step(unscaled_action)
            reward_sum[-1] += reward

            obs = new_obs
            if done:
                episode_num += 1
                logger.info(
                    "Episode %d | reward=%.4f | success=%s",
                    episode_num,
                    reward_sum[-1],
                    info.get("is_success"),
                )
                reward_sum = np.append(reward_sum, 0.0)
                obs = self.env.reset()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone evaluation runner")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")

    parser.add_argument("--eval-path", default=None, help="Single mode: run folder path")
    parser.add_argument("--config", default=None, help="Single mode: config path")
    parser.add_argument("--model-file", default=None, help="Single mode: model file path")
    parser.add_argument("--eval-eps", type=int, default=50)
    parser.add_argument("--eval-env", default=None)
    parser.add_argument("--eval-dynamics", default=None)

    parser.add_argument("--eval-logs-path", default=None, help="Multi mode: logs root")
    parser.add_argument("--eval-logs-name", default="Maze", help="Multi mode: label only")
    return parser


def main() -> None:
    args = get_parser().parse_args()

    if args.mode == "single":
        if not args.eval_path:
            raise ValueError("--eval-path is required when --mode single")
        eval_path = resolve_path(args.eval_path)
        config_file = resolve_path(args.config) if args.config else eval_path / "config" / "config.ini"
        model_file = (
            resolve_path(args.model_file)
            if args.model_file
            else eval_path / "models" / "model_sb3.zip"
        )

        evaluate_thread = EvaluateThread(
            str(eval_path),
            str(config_file),
            str(model_file),
            args.eval_eps,
            args.eval_env,
            args.eval_dynamics,
        )
        evaluate_thread.run()
        return

    if not args.eval_logs_path:
        raise ValueError("--eval-logs-path is required when --mode multi")

    eval_logs_path = resolve_path(args.eval_logs_path)
    model_list = []
    for train_name in os.listdir(eval_logs_path):
        train_dir = os.path.join(eval_logs_path, train_name)
        for repeat_name in os.listdir(train_dir):
            model_list.append(os.path.join(train_dir, repeat_name))

    results_list = []
    for i in tqdm(range(len(model_list))):
        eval_path = model_list[i]
        config_file = os.path.join(eval_path, "config", "config.ini")
        model_file = os.path.join(eval_path, "models", "model_sb3.zip")

        logger.info("[%d/%d] Evaluating %s", i + 1, len(model_list), eval_path)
        evaluate_thread = EvaluateThread(
            eval_path,
            config_file,
            model_file,
            args.eval_eps,
            args.eval_env,
            args.eval_dynamics,
        )
        results_list.append(evaluate_thread.run_drl_model())

    os.makedirs(PROJECT_ROOT / "logs_eval" / "results", exist_ok=True)
    np.save(
        PROJECT_ROOT
        / "logs_eval"
        / "results"
        / f"eval_{args.eval_eps}_{args.eval_logs_name}_{args.eval_env}_{args.eval_dynamics}",
        np.array(results_list),
    )


if __name__ == "__main__":
    main()
