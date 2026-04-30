import argparse
import time
from configparser import ConfigParser
from pathlib import Path

import gym
import gym_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="AirSim 环境快速测试")
    parser.add_argument(
        "--config",
        default="configs/config_NH_center_Multirotor_3D.ini",
        help="配置文件路径（支持项目相对路径或绝对路径）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    print("=" * 56)
    print("AirSim 环境快速测试")
    print("=" * 56)
    print(f"[步骤] 使用配置文件: {cfg_path}")

    cfg = ConfigParser()
    if not cfg.read(str(cfg_path)):
        raise FileNotFoundError(f"配置文件不存在或不可读取: {cfg_path}")

    env = gym.make("airsim-env-v0")
    env.set_config(cfg)
    env.reset()

    total_steps = 50
    report_interval_sec = 1.0

    step = 0
    fps_window_start = time.time()
    fps_window_counter = 0
    test_start = time.time()

    print(f"[步骤] 开始执行环境步进测试（总步数: {total_steps}）")

    try:
        for _ in range(total_steps):
            action = [5, 0]
            _, _, done, _ = env.step(action)
            step += 1

            if done:
                env.reset()

            fps_window_counter += 1
            now = time.time()
            elapsed = now - fps_window_start
            if elapsed >= report_interval_sec:
                fps = fps_window_counter / elapsed
                progress = (step / total_steps) * 100
                print(f"[运行中] 进度: {progress:6.2f}% | 当前FPS: {fps:6.2f}")
                fps_window_counter = 0
                fps_window_start = now

    except KeyboardInterrupt:
        print("\n[提示] 用户中断测试。")
    finally:
        # Release AirSim API control/pause state to avoid vehicle freezing after exit.
        env.close()

    total_elapsed = time.time() - test_start
    avg_fps = step / total_elapsed if total_elapsed > 0 else 0.0

    print("-" * 56)
    print(f"[结果] 完成步数: {step}")
    print(f"[结果] 总耗时: {total_elapsed:.2f} 秒")
    print(f"[结果] 平均FPS: {avg_fps:.2f}")
    print("[结论] 环境基础通路测试完成。")


if __name__ == "__main__":
    main()
