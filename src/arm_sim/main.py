"""机械臂仿真主入口 - 基于 MuJoCo 的 IndexSimulator 任务控制

用法：
    python main.py              # 默认配置
    python main.py --no-viewer  # 无 GUI 模式
"""

import time
import argparse
import yaml
import mujoco
import logging
from simulator import IndexSimulator
from task import ChoicePanelTask

# ── 日志配置 ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="MuJoCo 机械臂仿真任务")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="配置文件路径")
    parser.add_argument("--model", type=str, default="simulation.xml",
                        help="MuJoCo 模型 XML 路径")
    parser.add_argument("--no-viewer", action="store_true",
                        help="无 GUI 模式（仅日志输出）")
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="最大任务执行轮数（0=无限）")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("程序启动 | 配置=%s | 模型=%s", args.config, args.model)

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 初始化仿真器 + 任务
    sim = IndexSimulator(args.config, args.model)
    task = ChoicePanelTask(config, sim)

    sim.reset()
    task.reset()

    episode_count = 0

    try:
        viewer = None
        if not args.no_viewer:
            try:
                viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -15
                viewer.cam.distance = 0.6
                viewer.cam.lookat = [0.45, -0.15, 0.8]
                logger.info("可视化窗口已启动")
            except Exception as e:
                logger.warning("可视化启动失败: %s，切换到无窗口模式", e)

        while sim.is_running:
            sim.step()
            status = task.update()

            if status["done"]:
                episode_count += 1
                logger.info("任务结束 (第 %d 轮)", episode_count)
                if args.max_episodes > 0 and episode_count >= args.max_episodes:
                    break
                sim.reset()
                task.reset()

            if viewer:
                viewer.sync()
                time.sleep(0.001)

    except KeyboardInterrupt:
        logger.info("用户中断 (Ctrl+C)")
    finally:
        sim.close()
        logger.info("仿真结束 | 总步数=%d | 任务轮数=%d", sim.current_step, episode_count)


if __name__ == "__main__":
    main()