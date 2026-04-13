"""
快速启动脚本，用于无人机强化学习路径规划项目
运行此脚本来验证设置和快速开始训练
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """检查是否已安装所有必需的依赖项"""
    logger.info("正在检查依赖项...")

    required = {
        "gymnasium": "Gymnasium (Gym)",
        "stable_baselines3": "Stable Baselines3",
        "numpy": "NumPy",
        "torch": "PyTorch",
        "yaml": "PyYAML",
        "PIL": "Pillow",
    }

    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            logger.info(f"✓ {name}")
        except ImportError:
            logger.error(f"✗ {name} - 未安装")
            missing.append(name)

    if missing:
        logger.error(f"\n缺少依赖项: {', '.join(missing)}")
        logger.info("通过以下命令安装: pip install -r requirements.txt")
        return False

    logger.info("✓ 所有依赖项已安装\n")
    return True


def validate_environment():
    """验证 AirSim 环境设置"""
    logger.info("正在验证 AirSim 环境...")

    try:
        from src.envs.base_drone_env import AirSimDroneEnv
        from stable_baselines3.common.env_checker import check_env

        logger.info("正在创建环境...")
        env = AirSimDroneEnv(verbose=False)

        logger.info("正在检查环境格式...")
        check_env(env.unwrapped)

        logger.info("✓ 环境验证成功\n")
        env.close()
        return True

    except Exception as e:
        logger.error(f"✗ 环境验证失败: {e}")
        logger.info("确保 AirSim 模拟器正在运行!")
        return False


def show_menu():
    """显示交互式菜单"""
    while True:
        print("\n" + "=" * 50)
        print("DRONE RL PATH PLANNING - QUICK START")
        print("=" * 50)
        print("1. 验证环境")
        print("2. 从头开始训练新模型")
        print("3. 从检查点继续训练")
        print("4. 评估模型")
        print("5. 可视化结果")
        print("6. 查看 TensorBoard 日志")
        print("0. 退出")
        print("=" * 50)

        choice = input("请选择选项 (0-6): ").strip()

        if choice == "1":
            validate_environment()

        elif choice == "2":
            train_new()

        elif choice == "3":
            train_continue()

        elif choice == "4":
            evaluate()

        elif choice == "5":
            visualize()

        elif choice == "6":
            tensorboard()

        elif choice == "0":
            print("Goodbye!")
            break

        else:
            print("无效选项, try again")


def train_new():
    """训练新模型"""
    print("\n--- 训练新模型 ---")

    timesteps = input("总时间步长 [1000000]: ").strip()
    timesteps = int(timesteps) if timesteps else 1000000

    from src.training.train import train

    try:
        train(total_timesteps=timesteps)
        logger.info("✓ 训练完成!")
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练失败: {e}")


def train_continue():
    """从检查点继续训练"""
    print("\n--- 继续训练 ---")

    model_path = input("Model path [./data/best_model/final_model]: ").strip()
    model_path = model_path or "./data/best_model/final_model"

    timesteps = input("频次时间步长 [500000]: ").strip()
    timesteps = int(timesteps) if timesteps else 500000

    from src.training.train import train

    try:
        train(load_model_path=model_path, total_timesteps=timesteps)
        logger.info("✓ 训练完成!")
    except Exception as e:
        logger.error(f"训练失败: {e}")


def evaluate():
    """评估模型"""
    print("\n--- 评估模型 ---")

    model_path = input("Model path [./data/best_model/final_model]: ").strip()
    model_path = model_path or "./data/best_model/final_model"

    episodes = input("回合数 [10]: ").strip()
    episodes = int(episodes) if episodes else 10

    results_dir = input("Results directory [./data/results/]: ").strip()
    results_dir = results_dir or "./data/results/"

    from src.evaluation.evaluate import evaluate_model

    try:
        evaluate_model(
            model_path=model_path, n_episodes=episodes, results_dir=results_dir
        )
        logger.info("✓ 评估完成!")
    except Exception as e:
        logger.error(f"评估失败: {e}")


def visualize():
    """可视化结果"""
    print("\n--- 可视化结果 ---")

    results_dir = input("Results directory [./data/results/]: ").strip()
    results_dir = results_dir or "./data/results/"

    from src.evaluation.trajectory_vis import visualize_evaluation_results

    try:
        visualize_evaluation_results(results_dir)
        logger.info("✓ 可视化完成!")
    except Exception as e:
        logger.error(f"可视化失败: {e}")


def tensorboard():
    """启动 TensorBoard"""
    print("\n--- TensorBoard ---")

    import subprocess
    import shutil

    log_dir = "./data/logs/"

    logger.info(f"启动 TensorBoard，日志位置 {log_dir}")
    logger.info("在浏览器中打开 http://localhost:6006")

    try:
        # 较新的 TensorBoard 版本可能不再暴露 tensorboard.__main__。
        # 优先使用 CLI 入口，失败时回退到 tensorboard.main 模块。
        tensorboard_exe = shutil.which("tensorboard")
        if tensorboard_exe:
            cmd = [tensorboard_exe, f"--logdir={log_dir}", "--port=6006"]
        else:
            cmd = [
                sys.executable,
                "-m",
                "tensorboard.main",
                f"--logdir={log_dir}",
                "--port=6006",
            ]

        subprocess.run(cmd)
    except Exception as e:
        logger.error(f"启动 TensorBoard 失败: {e}")


def main():
    """主入口点"""
    logger.info("启动无人机强化学习路径规划项目\n")

    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)

    # 展示菜单
    try:
        show_menu()
    except KeyboardInterrupt:
        print("\n\n退出")
        sys.exit(0)


if __name__ == "__main__":
    main()
