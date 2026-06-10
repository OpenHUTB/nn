"""
carla_2d_deeprl — 主入口
=========================
遵循 OpenHUTB/nn 项目约定，提供统一的模块启动入口。

用法:
  python main.py              查看帮助
  python main.py demo          运行预训练模型演示（需 Carla 服务器）
  python main.py train         训练新 DQN 模型（需 Carla 服务器）
  python main.py test          运行自动化测试
"""
import sys
import subprocess


MODES = {
    "demo":  "python -m models.run --path models/model/base-model.pt",
    "train": "python -m models.train --episodes 20000 --save models/model/trained",
    "test":  "python -m pytest tests/ -v",
}


def main():
    if len(sys.argv) < 2:
        print("carla_2d_deeprl — 主入口")
        print("=" * 40)
        print("用法: python main.py [demo|train|test]")
        print()
        print("  demo  运行预训练模型演示")
        print("  train 训练新 DQN 模型")
        print("  test  运行自动化测试")
        print()
        print("示例:")
        print("  python main.py test     # 运行所有测试（无需 Carla）")
        print("  python main.py demo     # 需要 Carla 服务器运行中")
        return

    mode = sys.argv[1]
    if mode not in MODES:
        print(f"未知模式: {mode}")
        print(f"可选模式: {', '.join(MODES.keys())}")
        sys.exit(1)

    print(f"执行模式: {mode}")
    print(f"命令: {MODES[mode]}")
    print("-" * 40)
    subprocess.run(MODES[mode], shell=True)


if __name__ == "__main__":
    main()
