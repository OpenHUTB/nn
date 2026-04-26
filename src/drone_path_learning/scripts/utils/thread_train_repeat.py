try:
    from .custom_policy_sb3 import (
        CNN_FC,
        CNN_GAP,
        CNN_GAP_BN,
        No_CNN,
        CNN_MobileNet,
        CNN_GAP_new,
    )
except ImportError:
    from scripts.utils.custom_policy_sb3 import (
        CNN_FC,
        CNN_GAP,
        CNN_GAP_BN,
        No_CNN,
        CNN_MobileNet,
        CNN_GAP_new,
    )
import datetime
import gym
import gym_env
import numpy as np
from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
import wandb
from PyQt5 import QtCore
import argparse
import ast
from configparser import ConfigParser
import torch as th
import os


def get_parser():
    parser = argparse.ArgumentParser(description="Training thread without plot")
    parser.add_argument(
        "-c",
        "--config",
        help="config file name in configs folder, such as config_default",
        default="config_NH_center_Multirotor_3D",
    )
    parser.add_argument(
        "-n", "--note", help="training objective", default="depth_upper_split_5"
    )

    return parser


class TrainingThread(QtCore.QThread):
    """
    用于策略训练的QT线程
    """

    def __init__(self, config):
        super(TrainingThread, self).__init__()
        print("init training thread")

        # 配置
        self.cfg = ConfigParser()
        self.cfg.read(config)

        env_name = self.cfg.get("options", "env_name")
        self.project_name = env_name

        # 创建Gym环境
        self.env = gym.make("airsim-env-v0")
        self.env.set_config(self.cfg)

    def terminate(self):
        print("TrainingThread terminated")

    def run(self, seed=0):
        print("run training thread")

        # wandb相关
        if self.cfg.getboolean("options", "use_wandb"):
            run = wandb.init(
                project=self.project_name,
                notes=self.cfg.get("wandb", "notes"),
                name=self.cfg.get("wandb", "name") + "_" + str(seed),
                sync_tensorboard=True,  # 自动上传sb3的tensorboard指标
                save_code=True,  # 可选
            )

        # ! -----------------------------------初始化文件夹-----------------------------------------
        now = datetime.datetime.now()
        now_string = now.strftime("%Y_%m_%d_%H_%M")
        file_path = (
            "logs_new/"
            + self.project_name
            + "/"
            + now_string
            + "_"
            + self.cfg.get("options", "dynamic_name")
            + "_"
            + self.cfg.get("options", "policy_name")
            + "_"
            + self.cfg.get("options", "algo")
        )
        log_path = file_path + "/tb_logs"
        model_path = file_path + "/models"
        config_path = file_path + "/config"
        data_path = file_path + "/data"
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(config_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)  # 创建用于保存q_map的数据目录

        # 保存配置文件
        with open(os.path.join(config_path, "config.ini"), "w") as configfile:
            self.cfg.write(configfile)

        #! -----------------------------------策略选择-------------------------------------
        feature_num_state = self.env.dynamic_model.state_feature_length
        feature_num_cnn = self.cfg.getint("options", "cnn_feature_num")
        policy_name = self.cfg.get("options", "policy_name")

        # 特征提取网络
        if self.cfg.get("options", "activation_function") == "tanh":
            activation_function = th.nn.Tanh
        else:
            activation_function = th.nn.ReLU

        if policy_name == "mlp":
            policy_base = "MlpPolicy"
            policy_kwargs = dict(activation_fn=activation_function)
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
                raise Exception("policy select error: ", policy_name)

            policy_kwargs = dict(
                features_extractor_class=policy_used,
                features_extractor_kwargs=dict(
                    features_dim=feature_num_state + feature_num_cnn,
                    state_feature_dim=feature_num_state,
                ),
                activation_fn=activation_function,
            )

        # 特征提取后的全连接网络
        net_arch_list = ast.literal_eval(self.cfg.get("options", "net_arch"))
        policy_kwargs["net_arch"] = net_arch_list

        # ! ---------------------------------算法选择-------------------------------------
        algo = self.cfg.get("options", "algo")
        print("algo: ", algo)
        if algo == "PPO":
            model = PPO(
                policy_base,
                self.env,
                n_steps=self.cfg.getint("DRL", "n_steps"),
                batch_size=self.cfg.getint("DRL", "batch_size"),
                n_epochs=self.cfg.getint("DRL", "n_epochs"),
                gamma=self.cfg.getfloat("DRL", "gamma"),
                ent_coef=self.cfg.getfloat("DRL", "ent_coef"),
                clip_range=self.cfg.getfloat("DRL", "clip_range"),
                max_grad_norm=self.cfg.getfloat("DRL", "max_grad_norm"),
                learning_rate=self.cfg.getfloat("DRL", "learning_rate"),
                policy_kwargs=policy_kwargs,
                tensorboard_log=log_path,
                seed=seed,
                verbose=2,
            )
        elif algo == "SAC":
            n_actions = self.env.action_space.shape[-1]
            noise_sigma = self.cfg.getfloat("DRL", "action_noise_sigma") * np.ones(
                n_actions
            )
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions), sigma=noise_sigma
            )
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
                seed=seed,
                verbose=2,
            )
        elif algo == "TD3":
            # TD3的噪声对象
            n_actions = self.env.action_space.shape[-1]
            noise_sigma = self.cfg.getfloat("DRL", "action_noise_sigma") * np.ones(
                n_actions
            )
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions), sigma=noise_sigma
            )
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
                seed=seed,
                verbose=2,
            )
        else:
            raise Exception("Invalid algo name : ", algo)

        # TODO 创建评估回调
        # eval_freq = self.cfg.getint('TD3', 'eval_freq')
        # n_eval_episodes = self.cfg.getint('TD3', 'n_eval_episodes')
        # eval_callback = EvalCallback(self.env, best_model_save_path= file_path + '/eval',
        #                      log_path= file_path + '/eval', eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
        #                      deterministic=True, render=False)

        # ! -------------------------------------训练-----------------------------------------
        print("start training model")
        total_timesteps = self.cfg.getint("options", "total_timesteps")
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
            # if algo == 'TD3' or algo == 'SAC':
            #     wandb.watch(model.actor, log_freq=100, log="all")  # 记录梯度
            # elif algo == 'PPO':
            #     wandb.watch(model.policy, log_freq=100, log="all")
            callback_list = CallbackList(
                [
                    local_checkpoint_callback,
                    WandbCallback(
                        # Avoid wandb.save() symlink on Windows (WinError 1314).
                        # Local checkpoints are still written by local_checkpoint_callback.
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
            )
        else:
            model.learn(total_timesteps, callback=local_checkpoint_callback)

        # ! ---------------------------模型保存----------------------------------------------------
        model_name = "model_sb3"
        model.save(model_path + "/" + model_name)

        print("training finished")
        print("model saved to: {}".format(model_path))
        del model

        if self.cfg.getboolean("options", "use_wandb"):
            run.finish()


def main():
    parser = get_parser()
    args = parser.parse_args()

    config_file = "configs_new/" + args.config + ".ini"
    # config_file = 'configs_new\config_Trees_SimpleMultirotor.ini'

    print(config_file)

    training_thread = TrainingThread(config_file)

    repeat_num = 3
    for i in range(repeat_num):
        training_thread.run(seed=i)

    print("training end.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("system exit")
