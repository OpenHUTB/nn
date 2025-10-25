import sys
import os
import shutil
import logging
import random
import string
from datetime import datetime
from typing import Optional, List, Tuple

import wandb
from wandb.integration.sb3 import WandbCallback
import argparse

from uitb.simulator import Simulator
from uitb.utils.functions import output_path, timeout_input
from stable_baselines3.common.save_util import load_from_zip_file

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def generate_random_name(length: int = 8) -> str:
    """生成随机运行名称"""
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(length))


def get_checkpoint_path(checkpoint_dir: str, args_checkpoint: Optional[str], resume: bool) -> Optional[str]:
    """
    获取 checkpoint 路径（处理 --checkpoint 和 --resume 参数）

    Args:
        checkpoint_dir: checkpoint 存储目录
        args_checkpoint: 命令行指定的 checkpoint 文件名
        resume: 是否自动恢复最新 checkpoint

    Returns:
        有效的 checkpoint 路径或 None

    Raises:
        FileNotFoundError: 当目录不存在或 checkpoint 不存在时
        ValueError: 当参数组合无效时
    """
    # 检查参数合法性
    if args_checkpoint and resume:
        raise ValueError("同时指定 --checkpoint 和 --resume 参数，参数冲突")

    if not (args_checkpoint or resume):
        return None

    # 验证目录存在性
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint 目录不存在: {checkpoint_dir}")

    # 获取所有有效的 checkpoint 文件
    existing_checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if os.path.isfile(os.path.join(checkpoint_dir, f)) and f.endswith('.zip')
    ]

    # 检查是否有可用 checkpoint
    if not existing_checkpoints:
        raise FileNotFoundError(f"Checkpoint 目录为空（未找到 .zip 文件）: {checkpoint_dir}")

    # 处理指定 checkpoint 的情况
    if args_checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, args_checkpoint)
        # 检查文件是否存在且是 zip 文件
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"指定的 checkpoint 文件不存在: {checkpoint_path}")
        if not checkpoint_path.endswith('.zip'):
            raise ValueError(f"指定的 checkpoint 不是 zip 文件: {checkpoint_path}")
        return checkpoint_path
    else:
        # 按创建时间排序，取最新的 checkpoint
        # 增加排序稳定性（如果创建时间相同则按文件名排序）
        return sorted(
            existing_checkpoints,
            key=lambda x: (os.path.getctime(x), x)
        )[-1]


def backup_checkpoints(checkpoint_dir: str) -> None:
    """备份已存在的 checkpoint 目录"""
    if not os.path.isdir(checkpoint_dir):
        return

    existing_checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if os.path.isfile(os.path.join(checkpoint_dir, f))
    ]

    if existing_checkpoints:
        last_modified = max(os.path.getctime(f) for f in existing_checkpoints)
        timestamp = datetime.fromtimestamp(last_modified).strftime('%Y%m%d_%H%M%S')
        backup_dir = f"{checkpoint_dir}_{timestamp}"
        shutil.move(checkpoint_dir, backup_dir)
        logger.info(f"已备份原有 checkpoint 到: {backup_dir}")


def load_wandb_id(checkpoint_path: str) -> Optional[str]:
    """从 checkpoint 中加载 wandb run id"""
    try:
        data, _, _ = load_from_zip_file(checkpoint_path)
        return data.get("policy_kwargs", {}).get("wandb_id")
    except Exception as e:
        logger.warning(f"无法从 checkpoint 加载 wandb ID: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='强化学习智能体训练脚本')
    parser.add_argument('config_file_path', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='指定要恢复的 checkpoint 文件名（默认：从头开始训练）')
    parser.add_argument('--resume', action='store_true', help='自动恢复最新的 checkpoint')
    parser.add_argument('--eval', type=int, default=None, const=400000, nargs='?',
                        help='评估频率（每隔多少时间步评估一次，默认：400000）')
    parser.add_argument('--eval_info_keywords', type=str, nargs='*', default=[],
                        help='评估时需要记录的额外 info 关键字')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别（默认：INFO）')
    args = parser.parse_args()

    # 调整日志级别
    logger.setLevel(args.log_level)

    try:
        # 构建模拟器
        logger.info(f"正在构建模拟器，配置文件：{args.config_file_path}")
        simulator_folder = Simulator.build(args.config_file_path)
        simulator = Simulator.get(simulator_folder)
        config = simulator.config
        logger.info(f"模拟器初始化完成，名称：{config.get('simulator_name')}")

        # 处理 checkpoint 目录
        checkpoint_dir = os.path.join(simulator._simulator_folder, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 获取 checkpoint 路径和 wandb ID
        resume_training = args.resume or (args.checkpoint is not None)
        checkpoint_path = None
        wandb_id = None

        if resume_training:
            checkpoint_path = get_checkpoint_path(checkpoint_dir, args.checkpoint, args.resume)
            logger.info(f"将从 checkpoint 恢复训练：{checkpoint_path}")
            wandb_id = load_wandb_id(checkpoint_path)
        else:
            backup_checkpoints(checkpoint_dir)

        # 处理运行名称
        run_name = config.get("simulator_name")
        if not run_name:
            logger.info("未指定运行名称，等待用户输入...")
            run_name = timeout_input(
                "请为本次运行命名（30秒未输入将生成随机名称）：",
                timeout=30,
                default=generate_random_name()
            ).replace("-", "_").strip()
            config["simulator_name"] = run_name
            logger.info(f"运行名称确定为：{run_name}")

        # 初始化 wandb
        project_name = config.get("project", "uitb")
        logger.info(f"正在初始化 wandb，项目：{project_name}，运行名称：{run_name}")
        if wandb_id is None:
            wandb_id = wandb.util.generate_id()
            logger.info(f"生成新的 wandb ID：{wandb_id}")

        run = wandb.init(
            id=wandb_id,
            resume="allow",
            project=project_name,
            name=run_name,
            config=config,
            sync_tensorboard=True,
            save_code=True,
            dir=output_path(),
            reinit=True  # 允许重复初始化
        )

        # 初始化 RL 模型
        rl_algorithm = config["rl"]["algorithm"]
        logger.info(f"正在初始化 RL 模型，算法：{rl_algorithm}")
        rl_cls = simulator.get_class("rl", rl_algorithm)
        rl_model = rl_cls(
            simulator,
            checkpoint_path=checkpoint_path,
            wandb_id=wandb_id
        )

        # 开始训练
        logger.info("开始训练...")
        with_evaluation = args.eval is not None
        rl_model.learn(
            WandbCallback(verbose=2),
            with_evaluation=with_evaluation,
            eval_freq=args.eval if with_evaluation else None,
            eval_info_keywords=tuple(args.eval_info_keywords)
        )

        logger.info("训练完成")
        run.finish()

    except Exception as e:
        logger.error(f"训练过程出错：{str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()