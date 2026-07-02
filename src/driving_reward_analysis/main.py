"""自动驾驶奖励函数分析系统 - 主入口.

运行方式:
    python main.py              # 执行完整分析
    python main.py --mode plot  # 仅绘制理论曲线
    python main.py --mode sim   # 仅运行仿真分析
"""

import sys
import argparse
from utils import load_config, generate_scenario_data, compute_rewards, ensure_output_dir
from visualizer import (
    plot_reward_curves,
    plot_scenario_rewards,
    plot_speed_reward_comparison,
    plot_distance_reward_comparison,
)


def run_analysis(output_dir: str) -> None:
    """执行完整分析流程."""
    print("=" * 60)
    print("  自动驾驶奖励函数分析系统")
    print("=" * 60)

    # 1. 加载配置
    config = load_config()
    print(f"\n[Config] 目标速度: {config['speed_reward']['target_speed']} m/s")
    print(f"[Config] 安全距离: {config['distance_reward']['safe_distance']} m")
    print(f"[Config] 临界距离: {config['distance_reward']['critical_distance']} m")
    print(f"[Config] 最大加速度: {config['comfort_reward']['max_acceleration']} m/s²")

    # 2. 绘制理论奖励曲线
    print("\n[Plot] 生成奖励函数理论曲线...")
    plot_reward_curves(output_dir, config)

    # 3. 生成场景数据
    print("\n[Sim] 生成仿真场景数据...")
    scenario = generate_scenario_data(num_steps=200, dt=0.1)

    # 4. 计算奖励值
    print("[Sim] 计算奖励值...")
    rewards = compute_rewards(scenario, config)

    # 5. 输出统计信息
    print("\n[Stats] 奖励统计:")
    for name, key in [
        ("速度跟踪奖励", "speed_reward"),
        ("距离控制奖励", "distance_reward"),
        ("舒适性奖励", "comfort_reward"),
        ("总奖励", "total_reward"),
    ]:
        vals = rewards[key]
        print(f"  {name}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
              f"min={vals.min():.3f}, max={vals.max():.3f}")

    # 6. 绘制场景奖励变化图
    print("\n[Plot] 生成场景仿真奖励曲线...")
    plot_scenario_rewards(rewards, output_dir)

    # 7. 参数对比图
    print("\n[Plot] 生成参数对比图...")
    plot_speed_reward_comparison(output_dir, config)
    plot_distance_reward_comparison(output_dir, config)

    print(f"\n{'=' * 60}")
    print(f"  分析完成！效果图已保存至: {output_dir}")
    print(f"{'=' * 60}")


def main() -> None:
    """主函数."""
    parser = argparse.ArgumentParser(description="自动驾驶奖励函数分析系统")
    parser.add_argument(
        "--mode",
        choices=["all", "plot", "sim"],
        default="all",
        help="运行模式: all=全部, plot=仅绘图, sim=仅仿真 (default: all)",
    )
    args = parser.parse_args()

    output_dir = ensure_output_dir("outputs")

    if args.mode in ("all", "plot"):
        config = load_config()
        plot_reward_curves(output_dir, config)
        plot_speed_reward_comparison(output_dir, config)
        plot_distance_reward_comparison(output_dir, config)

    if args.mode in ("all", "sim"):
        config = load_config()
        scenario = generate_scenario_data()
        rewards = compute_rewards(scenario, config)
        plot_scenario_rewards(rewards, output_dir)

        print("\n[Stats] 奖励统计:")
        for name, key in [
            ("速度跟踪奖励", "speed_reward"),
            ("距离控制奖励", "distance_reward"),
            ("舒适性奖励", "comfort_reward"),
            ("总奖励", "total_reward"),
        ]:
            vals = rewards[key]
            print(f"  {name}: mean={vals.mean():.3f}, std={vals.std():.3f}")

    print(f"\n完成！输出目录: {output_dir}")


if __name__ == "__main__":
    main()
