"""主要训练脚本用于无人机强化学习代理"""

import argparse
import importlib.util
import inspect
import logging
import os
from pathlib import Path
from typing import Optional
import yaml
import numpy as np

# 设置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _supports_progress_bar() -> bool:
    """Return True when optional deps needed by SB3 progress bar are installed."""
    return (
        importlib.util.find_spec("tqdm") is not None
        and importlib.util.find_spec("rich") is not None
    )


def load_config(config_path: str) -> dict:
    """从YAML文件加载配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"从 {config_path} 加载配置")
    return config


def create_environment(config: dict):
    """创建和包装环境"""
    from ..envs.base_drone_env import AirSimDroneEnv
    from ..envs.wrappers import create_wrapped_env

    env_config = config.get("environment", {})
    obs_config = config.get("observation", {})

    # 创建基础环境
    target_points = [np.array(pt) for pt in env_config.get("target_points", [])]

    env = AirSimDroneEnv(
        ip_address=env_config.get("ip_address", "127.0.0.1"),
        image_shape=tuple(env_config.get("image_shape", [84, 84])),
        max_steps=config.get("training", {}).get("max_steps_per_episode", 500),
        target_points=target_points if target_points else None,
        action_duration=env_config.get("action_duration", 0.5),
        velocity_step=env_config.get("velocity_step", 5.0),
        verbose=config.get("logging", {}).get("verbose", 1) >= 2,
    )

    # 应用包装器
    env = create_wrapped_env(
        env,
        frame_stack=obs_config.get("frame_stack", 4),
        normalize_obs=obs_config.get("normalize", True),
        clip_reward=False,
        action_repeat=config.get("action", {}).get("repeat", 1),
    )

    logger.info(
        f"环境已创建: obs_space={env.observation_space}, "
        f"action_space={env.action_space}"
    )

    return env


def create_model(env, config: dict):
    """使用指定配置创建PPO/DQN模型"""
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecTransposeImage,
    )

    training_config = config.get("training", {})
    algorithm = training_config.get("algorithm", "PPO").upper()

    # 创建向量化环境
    n_envs = config.get("hardware", {}).get("n_envs", 1)
    vec_env_type = config.get("hardware", {}).get("vec_env_type", "dummy")

    if n_envs == 1:
        vec_env = DummyVecEnv([lambda: env])
    else:
        if vec_env_type == "subproc":
            vec_env = SubprocVecEnv([lambda: env for _ in range(n_envs)])
        else:
            vec_env = DummyVecEnv([lambda: env for _ in range(n_envs)])

    # 转置图像用于 CNN（PyTorch 格式）
    vec_env = VecTransposeImage(vec_env)

    # 确定策略类型
    policy = "CnnPolicy"  # 用于图像观测

    # 基于算法创建模型
    device = config.get("hardware", {}).get("device", "auto")
    verbose = config.get("logging", {}).get("verbose", 1)

    if algorithm == "PPO":
        ppo_config = config.get("ppo", {})
        model = PPO(
            policy,
            vec_env,
            learning_rate=float(ppo_config.get("learning_rate", 3e-4)),
            n_steps=int(ppo_config.get("n_steps", 2048)),
            batch_size=int(ppo_config.get("batch_size", 64)),
            n_epochs=int(ppo_config.get("n_epochs", 10)),
            gamma=float(ppo_config.get("gamma", 0.99)),
            gae_lambda=float(ppo_config.get("gae_lambda", 0.95)),
            clip_range=float(ppo_config.get("clip_range", 0.2)),
            ent_coef=float(ppo_config.get("ent_coef", 0.0)),
            vf_coef=float(ppo_config.get("vf_coef", 0.5)),
            max_grad_norm=float(ppo_config.get("max_grad_norm", 0.5)),
            use_sde=ppo_config.get("use_sde", False),
            device=device,
            verbose=verbose,
            tensorboard_log=config.get("logging", {}).get(
                "tensorboard_log", "./data/logs/"
            ),
        )

    elif algorithm == "DQN":
        dqn_config = config.get("dqn", {})
        model = DQN(
            policy,
            vec_env,
            learning_rate=float(dqn_config.get("learning_rate", 1e-4)),
            buffer_size=int(dqn_config.get("buffer_size", 1000000)),
            learning_starts=int(dqn_config.get("learning_starts", 50000)),
            batch_size=int(dqn_config.get("batch_size", 32)),
            tau=float(dqn_config.get("tau", 1.0)),
            gamma=float(dqn_config.get("gamma", 0.99)),
            device=device,
            verbose=verbose,
            tensorboard_log=config.get("logging", {}).get(
                "tensorboard_log", "./data/logs/"
            ),
        )

    else:
        raise ValueError(f"未知算法: {algorithm}")

    logger.info(f"模型已创建: {algorithm}")
    return model, vec_env


def create_callbacks(config: dict):
    """创建训练回调"""
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from .callbacks import CustomCallback, ProgressCallback

    logging_config = config.get("logging", {})

    # 不同 stable-baselines3 版本对 CheckpointCallback 的参数支持略有差异。
    checkpoint_kwargs = {
        "save_freq": int(config.get("training", {}).get("checkpoint_freq", 10000)),
        "save_path": logging_config.get("checkpoint_dir", "./data/checkpoints/"),
        "name_prefix": "drone_model",
        "save_replay_buffer": False,
    }

    callback_params = inspect.signature(CheckpointCallback.__init__).parameters
    if "save_vecnorm_stats" in callback_params:
        checkpoint_kwargs["save_vecnorm_stats"] = False

    # 检查点回调
    checkpoint_callback = CheckpointCallback(**checkpoint_kwargs)

    # 自定义日志回调
    custom_callback = CustomCallback(
        log_dir=logging_config.get("log_dir", "./data/logs/"),
        verbose=logging_config.get("verbose", 1),
    )

    # 进度回调
    progress_callback = ProgressCallback(verbose=logging_config.get("verbose", 1))

    callbacks = [checkpoint_callback, custom_callback, progress_callback]

    logger.info(f"已创建 {len(callbacks)} 个回调")
    return callbacks


def train(
    config_path: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    load_model_path: Optional[str] = None,
):
    """
    主训练函数。

    Args:
        config_path: 配置YAML文件的路径
        total_timesteps: 覆盖配置中的总时间步长
        load_model_path: 可选的预训练模型加载路径
    """
    # 加载配置
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    config = load_config(config_path)
    logger.info(f"训练配置: {config}")

    # Create directories
    for key in ["log_dir", "checkpoint_dir", "best_model_dir"]:
        dir_path = config.get("logging", {}).get(key, "")
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    # 创建环境
    logger.info("正在创建环境...")
    env = create_environment(config)

    # 验证环境
    logger.info("正在验证环境...")
    try:
        from stable_baselines3.common.env_checker import check_env

        check_env(env.unwrapped)
        logger.info("✓ 环境验证通过")
    except Exception as e:
        logger.error(f"✗ 环境验证失败: {e}")
        raise

    # 创建或加载模型
    if load_model_path:
        logger.info(f"正在从 {load_model_path} 加载模型...")
        from stable_baselines3 import PPO, DQN

        algorithm = config.get("training", {}).get("algorithm", "PPO").upper()
        if algorithm == "PPO":
            model = PPO.load(load_model_path, env=env)
        else:
            model = DQN.load(load_model_path, env=env)
        logger.info("✓ 模型加载成功")
    else:
        logger.info("正在创建模型...")
        model, vec_env = create_model(env, config)
        logger.info("✓ 模型创建成功")

    # 创建回调
    logger.info("正在创建回调...")
    callbacks = create_callbacks(config)

    # 开始训练
    logger.info("开始训练...")
    train_timesteps = total_timesteps or config.get("training", {}).get(
        "total_timesteps", 1000000
    )

    try:
        use_progress_bar = _supports_progress_bar()
        if not use_progress_bar:
            logger.warning(
                "未检测到 tqdm/rich，已自动关闭 progress_bar。"
                "如需进度条，请安装: pip install stable-baselines3[extra]"
            )

        model.learn(
            total_timesteps=int(train_timesteps),
            callback=callbacks,
            log_interval=10,
            progress_bar=use_progress_bar,
        )
        logger.info("✓ 训练成功完成")
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise
    finally:
        # 保存最终模型
        if config.get("logging", {}).get("save_final_model", True):
            final_model_path = os.path.join(
                config.get("logging", {}).get("best_model_dir", "./data/best_model/"),
                "final_model",
            )
            model.save(final_model_path)
            logger.info(f"✓ 最终模型已保存到 {final_model_path}")

        # 清理
        env.close()
        logger.info("环境已关闭")


def main():
    """命令行界面"""
    parser = argparse.ArgumentParser(description="训练无人机强化学习代理")
    parser.add_argument("--config", type=str, default=None, help="配置YAML文件的路径")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="总训练时间步长（覆盖配置）",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="预训练模型的路径，用于继续训练",
    )

    args = parser.parse_args()

    train(
        config_path=args.config,
        total_timesteps=args.timesteps,
        load_model_path=args.load,
    )


if __name__ == "__main__":
    main()
