import os
import sys
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Dict
import numpy as np
import mujoco
from mujoco import viewer

# MuJoCo多功能工具集：提供模型可视化、性能测试和格式转换的一站式解决方案
# 核心架构：统一的模型加载接口 + 模块化功能组件 + 命令行驱动

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("mujoco_utils")


def load_model(model_path: str) -> Tuple[Optional[mujoco.MjModel], Optional[mujoco.MjData]]:
    """
    加载MuJoCo模型（支持XML和MJB格式）
    
    参数:
        model_path: 模型文件路径
        
    返回:
        加载成功返回(model, data)元组，失败返回(None, None)
    """
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None, None

    try:
        if model_path.endswith('.mjb'):
            model = mujoco.MjModel.from_binary_path(model_path)
        else:
            model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        logger.info(f"成功加载模型: {model_path}")
        return model, data
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}", exc_info=True)
        return None, None


def convert_model(input_path: str, output_path: str) -> bool:
    """
    转换模型格式（XML↔MJB）
    
    参数:
        input_path: 输入模型路径
        output_path: 输出模型路径（需指定扩展名.xml或.mjb）
        
    返回:
        转换成功返回True，失败返回False
    """
    model, _ = load_model(input_path)
    if not model:
        return False

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"创建输出目录: {output_dir}")
        except Exception as e:
            logger.error(f"无法创建输出目录: {str(e)}")
            return False

    try:
        if output_path.endswith('.mjb'):
            mujoco.save_model(model, output_path)
            logger.info(f"二进制模型已保存至: {output_path}")
        else:
            # 处理XML格式保存
            xml_content = mujoco.mj_saveLastXMLToString(model, output_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            logger.info(f"XML模型已保存至: {output_path}")
        return True
    except Exception as e:
        logger.error(f"模型转换失败: {str(e)}", exc_info=True)
        return False


def test_speed(
    model_path: str,
    nstep: int = 10000,
    nthread: int = 1,
    ctrlnoise: float = 0.01
) -> None:
    """
    测试模型模拟速度
    
    参数:
        model_path: 模型文件路径
        nstep: 每线程模拟步数
        nthread: 测试线程数
        ctrlnoise: 控制噪声强度
    """
    model, _ = load_model(model_path)
    if not model:
        return

    # 参数验证
    if nstep <= 0:
        logger.error("步数必须为正数")
        return
    if nthread <= 0:
        logger.error("线程数必须为正数")
        return

    # 生成控制噪声
    ctrl = ctrlnoise * np.random.randn(nstep, model.nu)
    logger.info(f"开始速度测试: 线程数={nthread}, 每线程步数={nstep}")

    def simulate_thread(thread_id: int) -> float:
        """单线程模拟函数"""
        mj_data = mujoco.MjData(model)
        start = time.perf_counter()
        for i in range(nstep):
            mj_data.ctrl[:] = ctrl[i]
            mujoco.mj_step(model, mj_data)
        end = time.perf_counter()
        duration = end - start
        logger.debug(f"线程 {thread_id} 完成，耗时: {duration:.2f}秒")
        return duration

    # 执行多线程测试
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=nthread) as executor:
        thread_durations: List[float] = list(executor.map(simulate_thread, range(nthread)))
    total_time = time.perf_counter() - start_time

    # 计算性能指标
    total_steps = nstep * nthread
    steps_per_sec = total_steps / total_time
    realtime_factor = (total_steps * model.opt.timestep) / total_time

    logger.info("\n===== 速度测试结果 =====")
    logger.info(f"总步数: {total_steps:,}")
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info(f"每秒步数: {steps_per_sec:.0f}")
    logger.info(f"实时因子: {realtime_factor:.2f}x")
    logger.info(f"线程平均耗时: {np.mean(thread_durations):.2f}秒 (±{np.std(thread_durations):.2f})")


def visualize(model_path: str) -> None:
    """
    可视化模型并运行模拟
    
    参数:
        model_path: 模型文件路径
    """
    model, data = load_model(model_path)
    if not model:
        return

    logger.info("启动可视化窗口（按ESC键退出，鼠标可交互操作）")
    try:
        with viewer.launch_passive(model, data) as v:
            while v.is_running():
                mujoco.mj_step(model, data)
                v.sync()
        logger.info("可视化窗口已关闭")
    except Exception as e:
        logger.error(f"可视化过程出错: {str(e)}", exc_info=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuJoCo功能整合工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 可视化命令
    viz_parser = subparsers.add_parser("visualize", help="可视化模型并运行模拟")
    viz_parser.add_argument("model", help="模型文件路径（XML或MJB）")

    # 速度测试命令
    speed_parser = subparsers.add_parser("testspeed", help="测试模型模拟速度")
    speed_parser.add_argument("model", help="模型文件路径")
    speed_parser.add_argument("--nstep", type=int, default=10000, help="每线程模拟步数")
    speed_parser.add_argument("--nthread", type=int, default=1, help="测试线程数量")
    speed_parser.add_argument("--ctrlnoise", type=float, default=0.01, help="控制噪声强度")

    # 模型转换命令
    convert_parser = subparsers.add_parser("convert", help="转换模型格式（XML↔MJB）")
    convert_parser.add_argument("input", help="输入模型路径")
    convert_parser.add_argument("output", help="输出模型路径（需指定.xml或.mjb扩展名）")

    args = parser.parse_args()

    # 命令映射
    command_handlers: Dict[str, callable] = {
        "visualize": lambda: visualize(args.model),
        "testspeed": lambda: test_speed(args.model, args.nstep, args.nthread, args.ctrlnoise),
        "convert": lambda: convert_model(args.input, args.output)
    }

    # 执行命令
    try:
        command_handlers[args.command]()
    except KeyError:
        logger.error(f"未知命令: {args.command}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"程序执行失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
    