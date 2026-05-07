import argparse
import ast
import datetime
import logging
import os
import sys
from configparser import ConfigParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bootstrap_project_paths() -> None:
    for path in (PROJECT_ROOT, PROJECT_ROOT / "gym_env"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


_bootstrap_project_paths()

import gym
import gym_env
import numpy as np
import torch as th
import wandb
from PyQt5 import QtCore
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from wandb.integration.sb3 import WandbCallback

try:
    from .custom_policy_sb3 import (
        CNN_FC,
        CNN_GAP_BN,
        CNN_GAP_new,
        CNN_MobileNet,
        No_CNN,
    )
except ImportError:
    from scripts.utils.custom_policy_sb3 import (
        CNN_FC,
        CNN_GAP_BN,
        CNN_GAP_new,
        CNN_MobileNet,
        No_CNN,
    )

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

DEFAULT_CONFIG_BASENAME = "config_NH_center_Multirotor_3D"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training thread without plot")
    parser.add_argument(
        "-c",
        "--config",
        help="Config basename in configs/ (without .ini) or an explicit .ini path",
        default=DEFAULT_CONFIG_BASENAME,
    )
    parser.add_argument("-n", "--note", help="Training objective note", default="")
    return parser


def resolve_config_path(raw: str) -> Path:
    path = Path(raw)
    if path.suffix.lower() != ".ini":
        path = Path("configs") / f"{raw}.ini"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


class TrainingThread(QtCore.QThread):
    """QThread for DRL policy training."""

    def __init__(self, config: str):
        super().__init__()
        logger.info("Initializing training thread")

        self.cfg = ConfigParser()
        if not self.cfg.read(config):
            raise FileNotFoundError(f"Config file not found or unreadable: {config}")

        self.project_name = self.cfg.get("options", "env_name")

        self.env = gym.make("airsim-env-v0")
        self.env.set_config(self.cfg)

        self.wandb_run = None
        if self.cfg.getboolean("options", "use_wandb"):
            self.wandb_run = wandb.init(
                project=self.project_name,
                notes=self.cfg.get("wandb", "notes"),
                name=self.cfg.get("wandb", "name"),
                sync_tensorboard=True,
                save_code=True,
            )

    def terminate(self):
        logger.info("Training thread terminated")

    def run(self):
        logger.info("Training thread started")

        now_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        file_path = os.path.join(
            "logs",
            self.project_name,
            f"{now_string}_{self.cfg.get('options', 'dynamic_name')}_{self.cfg.get('options', 'policy_name')}_{self.cfg.get('options', 'algo')}",
        )
        log_path = os.path.join(file_path, "tb_logs")
        model_path = os.path.join(file_path, "models")
        config_path = os.path.join(file_path, "config")
        data_path = os.path.join(file_path, "data")
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(config_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)

        with open(os.path.join(config_path, "config.ini"), "w", encoding="utf-8") as configfile:
            self.cfg.write(configfile)

        feature_num_state = self.env.dynamic_model.state_feature_length
        feature_num_cnn = self.cfg.getint("options", "cnn_feature_num")
        policy_name = self.cfg.get("options", "policy_name")

        activation_function = (
            th.nn.Tanh
            if self.cfg.get("options", "activation_function") == "tanh"
            else th.nn.ReLU
        )

        if policy_name == "mlp":
            policy_base = "MlpPolicy"
            policy_kwargs = {"activation_fn": activation_function}
        else:
            policy_base = "CnnPolicy"
            if policy_name == "CNN_FC":
                policy_used = CNN_FC
            elif policy_name == "CNN_GAP":
                policy_used = CNN_GAP_new
            elif policy_name == "CNN_GAP_BN":
                policy_used = CNN_GAP_BN
            elif policy_name == "CNN_MobileNet":
                policy_used = CNN_MobileNet
            elif policy_name == "No_CNN":
                policy_used = No_CNN
            else:
                raise ValueError(f"Unsupported policy_name: {policy_name}")

            policy_kwargs = {
                "features_extractor_class": policy_used,
                "features_extractor_kwargs": {
                    "features_dim": feature_num_state + feature_num_cnn,
                    "state_feature_dim": feature_num_state,
                },
                "activation_fn": activation_function,
            }

        policy_kwargs["net_arch"] = ast.literal_eval(self.cfg.get("options", "net_arch"))

        algo = self.cfg.get("options", "algo")
        logger.info("Algorithm selected: %s", algo)
        training_device = "cpu" if algo == "PPO" and policy_name == "mlp" else "auto"

        resume_model_path = None
        if self.cfg.has_option("options", "resume_model_path"):
            raw_resume_path = self.cfg.get("options", "resume_model_path").strip()
            if raw_resume_path:
                candidate = Path(raw_resume_path)
                if not candidate.is_absolute():
                    candidate = PROJECT_ROOT / candidate
                resume_model_path = candidate.resolve()
                if not resume_model_path.exists():
                    raise FileNotFoundError(f"Resume model not found: {resume_model_path}")

        is_resume_training = resume_model_path is not None
        if is_resume_training:
            logger.info("Resume training from model: %s", resume_model_path)
            if algo == "PPO":
                model = PPO.load(str(resume_model_path), env=self.env, tensorboard_log=log_path, device=training_device)
            elif algo == "SAC":
                model = SAC.load(str(resume_model_path), env=self.env, tensorboard_log=log_path, device="auto")
            elif algo == "TD3":
                model = TD3.load(str(resume_model_path), env=self.env, tensorboard_log=log_path, device="auto")
            else:
                raise ValueError(f"Invalid algo name: {algo}")
        else:
            if algo == "PPO":
                gae_lambda = 0.95
                if self.cfg.has_option("DRL", "gae_lambda"):
                    gae_lambda = self.cfg.getfloat("DRL", "gae_lambda")

                vf_coef = 0.5
                if self.cfg.has_option("DRL", "vf_coef"):
                    vf_coef = self.cfg.getfloat("DRL", "vf_coef")

                target_kl = None
                if self.cfg.has_option("DRL", "target_kl"):
                    target_kl = self.cfg.getfloat("DRL", "target_kl")

                model = PPO(
                    policy_base,
                    self.env,
                    n_steps=self.cfg.getint("DRL", "n_steps"),
                    batch_size=self.cfg.getint("DRL", "batch_size"),
                    n_epochs=self.cfg.getint("DRL", "n_epochs"),
                    gamma=self.cfg.getfloat("DRL", "gamma"),
                    gae_lambda=gae_lambda,
                    ent_coef=self.cfg.getfloat("DRL", "ent_coef"),
                    vf_coef=vf_coef,
                    clip_range=self.cfg.getfloat("DRL", "clip_range"),
                    target_kl=target_kl,
                    max_grad_norm=self.cfg.getfloat("DRL", "max_grad_norm"),
                    learning_rate=self.cfg.getfloat("DRL", "learning_rate"),
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=log_path,
                    device=training_device,
                    seed=0,
                    verbose=2,
                )
            elif algo == "SAC":
                n_actions = self.env.action_space.shape[-1]
                noise_sigma = self.cfg.getfloat("DRL", "action_noise_sigma") * np.ones(n_actions)
                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma)
                model = SAC(
                    policy_base,
                    self.env,
                    action_noise=action_noise,
                    policy_kwargs=policy_kwargs,
                    buffer_size=self.cfg.getint("DRL", "buffer_size"),
                    gamma=self.cfg.getfloat("DRL", "gamma"),
                    learning_starts=self.cfg.getint("DRL", "learning_starts"),
                    learning_rate=self.cfg.getfloat("DRL", "learning_rate"),
                    batch_size=self.cfg.getint("DRL", "batch_size"),
                    train_freq=(self.cfg.getint("DRL", "train_freq"), "step"),
                    gradient_steps=self.cfg.getint("DRL", "gradient_steps"),
                    tensorboard_log=log_path,
                    seed=0,
                    verbose=2,
                )
            elif algo == "TD3":
                n_actions = self.env.action_space.shape[-1]
                noise_sigma = self.cfg.getfloat("DRL", "action_noise_sigma") * np.ones(n_actions)
                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma)
                model = TD3(
                    policy_base,
                    self.env,
                    action_noise=action_noise,
                    learning_rate=self.cfg.getfloat("DRL", "learning_rate"),
                    gamma=self.cfg.getfloat("DRL", "gamma"),
                    policy_kwargs=policy_kwargs,
                    learning_starts=self.cfg.getint("DRL", "learning_starts"),
                    batch_size=self.cfg.getint("DRL", "batch_size"),
                    train_freq=(self.cfg.getint("DRL", "train_freq"), "step"),
                    gradient_steps=self.cfg.getint("DRL", "gradient_steps"),
                    buffer_size=self.cfg.getint("DRL", "buffer_size"),
                    tensorboard_log=log_path,
                    seed=0,
                    verbose=2,
                )
            else:
                raise ValueError(f"Invalid algo name: {algo}")

        logger.info("Training device: %s", getattr(model, "device", "unknown"))
        logger.info("Start training model")
        total_timesteps = self.cfg.getint("options", "total_timesteps")
        reset_num_timesteps = not is_resume_training
        if self.cfg.has_option("options", "reset_num_timesteps"):
            reset_num_timesteps = self.cfg.getboolean("options", "reset_num_timesteps")
        self.env.model = model
        self.env.data_path = data_path

        checkpoint_freq = 10000
        if self.cfg.has_option("options", "checkpoint_freq"):
            checkpoint_freq = self.cfg.getint("options", "checkpoint_freq")

        local_checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=model_path,
            name_prefix="model_sb3_ckpt",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        if self.cfg.getboolean("options", "use_wandb"):
            callback_list = CallbackList(
                [
                    local_checkpoint_callback,
                    WandbCallback(
                        model_save_freq=0,
                        gradient_save_freq=5000,
                        verbose=2,
                    ),
                ]
            )
            model.learn(
                total_timesteps,
                log_interval=1,
                callback=callback_list,
                reset_num_timesteps=reset_num_timesteps,
            )
        else:
            model.learn(
                total_timesteps,
                callback=local_checkpoint_callback,
                reset_num_timesteps=reset_num_timesteps,
            )

        model_name = "model_sb3"
        model.save(os.path.join(model_path, model_name))

        logger.info("Training finished")
        logger.info("Model saved to: %s", model_path)



def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    config_file = resolve_config_path(args.config)
    logger.info("Using config file: %s", config_file)

    training_thread = TrainingThread(str(config_file))
    training_thread.run()


if __name__ == "__main__":
    main()
