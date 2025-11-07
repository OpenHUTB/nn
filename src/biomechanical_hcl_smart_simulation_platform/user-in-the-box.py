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
    """生成随机运行名称（字母+数字组合）

    Args:
        length: 随机名称长度，默认8位

    Returns:
        随机字符串
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(length))


def get_checkpoint_path(checkpoint_dir: str, args_checkpoint: Optional[str], resume: bool) -> Optional[str]:
    """获取有效的checkpoint路径（处理--checkpoint和--resume参数）

    Args:
        checkpoint_dir: checkpoint存储目录
        args_checkpoint: 命令行指定的checkpoint文件名
        resume: 是否自动恢复最新checkpoint

    Returns:
        有效的checkpoint路径或None

    Raises:
        FileNotFoundError: 目录不存在或checkpoint不存在时
        ValueError: 参数组合无效时
    """
    # 检查参数合法性（互斥参数）
    if args_checkpoint and resume:
        raise ValueError("参数冲突：不能同时指定--checkpoint和--resume")

    if not (args_checkpoint or resume):
        return None

    # 验证checkpoint目录存在性
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint目录不存在: {checkpoint_dir}")

    # 获取所有有效的checkpoint文件（仅.zip文件）
    existing_checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if os.path.isfile(os.path.join(checkpoint_dir, f)) and f.endswith('.zip')
    ]

    # 检查是否有可用checkpoint
    if not existing_checkpoints:
        raise FileNotFoundError(f"Checkpoint目录为空（未找到.zip文件）: {checkpoint_dir}")

    # 处理指定checkpoint的情况
    if args_checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, args_checkpoint)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"指定的checkpoint文件不存在: {checkpoint_path}")
        if not checkpoint_path.endswith('.zip'):
            raise ValueError(f"指定的checkpoint不是zip文件: {checkpoint_path}")
        return checkpoint_path
    else:
        # 按创建时间排序（时间相同则按文件名），取最新的checkpoint
        return sorted(
            existing_checkpoints,
            key=lambda x: (os.path.getctime(x), x)
        )[-1]


def backup_checkpoints(checkpoint_dir: str) -> None:
    """备份已存在的checkpoint目录（仅当目录非空时）

    Args:
        checkpoint_dir: 需要备份的checkpoint目录
    """
    if not os.path.isdir(checkpoint_dir):
        return

    # 获取目录中所有文件
    existing_files = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if os.path.isfile(os.path.join(checkpoint_dir, f))
    ]

    if existing_files:
        # 以最新文件的修改时间作为备份目录的时间戳
        last_modified = max(os.path.getctime(f) for f in existing_files)
        timestamp = datetime.fromtimestamp(last_modified).strftime('%Y%m%d_%H%M%S')
        backup_dir = f"{checkpoint_dir}_{timestamp}"

        # 执行备份
        shutil.move(checkpoint_dir, backup_dir)
        logger.info(f"已备份原有checkpoint到: {backup_dir}")


def load_wandb_id(checkpoint_path: str) -> Optional[str]:
    """从checkpoint中加载wandb run id（用于恢复训练）

    Args:
        checkpoint_path: checkpoint文件路径

    Returns:
        从checkpoint中解析的wandb id，失败则返回None
    """
    try:
        data, _, _ = load_from_zip_file(checkpoint_path)
        return data.get("policy_kwargs", {}).get("wandb_id")
    except Exception as e:
        logger.warning(f"无法从checkpoint加载wandb ID: {str(e)}")
        return None


def setup_wandb(project_name: str, run_name: str, config: dict,
                wandb_id: Optional[str]) -> wandb.wandb_sdk.wandb_run.Run:
    """初始化wandb运行实例

    Args:
        project_name: 项目名称
        run_name: 运行名称
        config: 配置字典
        wandb_id: 用于恢复的wandb id

    Returns:
        初始化后的wandb run对象
    """
    # 生成新的wandb id（如果没有从checkpoint获取到）
    if wandb_id is None:
        wandb_id = wandb.util.generate_id()
        logger.info(f"生成新的wandb ID：{wandb_id}")

    return wandb.init(
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


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='强化学习智能体训练脚本')
    parser.add_argument('config_file_path', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='指定要恢复的checkpoint文件名（默认：从头开始训练）')
    parser.add_argument('--resume', action='store_true', help='自动恢复最新的checkpoint')
    parser.add_argument('--eval', type=int, default=None, const=400000, nargs='?',
                        help='评估频率（每隔多少时间步评估一次，默认：400000）')
    parser.add_argument('--eval_info_keywords', type=str, nargs='*', default=[],
                        help='评估时需要记录的额外info关键字')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别（默认：INFO）')
    args = parser.parse_args()

    # 调整日志级别
    logger.setLevel(args.log_level)

    try:
        # 构建并初始化模拟器
        logger.info(f"正在构建模拟器，配置文件：{args.config_file_path}")
        simulator_folder = Simulator.build(args.config_file_path)
        simulator = Simulator.get(simulator_folder)
        config = simulator.config
        logger.info(f"模拟器初始化完成，名称：{config.get('simulator_name')}")

        # 处理checkpoint目录（确保目录存在）
        checkpoint_dir = os.path.join(simulator._simulator_folder, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 获取checkpoint路径和wandb ID（如果是恢复训练）
        resume_training = args.resume or (args.checkpoint is not None)
        checkpoint_path = None
        wandb_id = None

        if resume_training:
            checkpoint_path = get_checkpoint_path(checkpoint_dir, args.checkpoint, args.resume)
            logger.info(f"将从checkpoint恢复训练：{checkpoint_path}")
            wandb_id = load_wandb_id(checkpoint_path)
        else:
            # 非恢复模式下备份已有checkpoint
            backup_checkpoints(checkpoint_dir)

        # 确定运行名称（优先使用配置，无则用户输入或随机生成）
        run_name = config.get("simulator_name")
        if not run_name:
            logger.info("未指定运行名称，等待用户输入...")
            run_name = timeout_input(
                "请为本次运行命名（30秒未输入将生成随机名称）：",
                timeout=30,
                default=generate_random_name()
            ).replace("-", "_").strip()
            config["simulator_name"] = run_name  # 更新配置
            logger.info(f"运行名称确定为：{run_name}")

        # 初始化wandb
        project_name = config.get("project", "uitb")
        logger.info(f"正在初始化wandb，项目：{project_name}，运行名称：{run_name}")
        run = setup_wandb(project_name, run_name, config, wandb_id)

        # 初始化RL模型
        rl_algorithm = config["rl"]["algorithm"]
        logger.info(f"正在初始化RL模型，算法：{rl_algorithm}")
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