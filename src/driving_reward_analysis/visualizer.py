"""可视化模块：生成奖励函数分析图表."""

import os
import numpy as np
import matplotlib.pyplot as plt


# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_reward_curves(
    output_dir: str,
    config: dict,
    prefix: str = "",
) -> str:
    """绘制三类奖励函数的理论曲线.

    Args:
        output_dir: 输出目录路径
        config: 配置字典

    Returns:
        保存的图片路径
    """
    speed_cfg = config["speed_reward"]
    distance_cfg = config["distance_reward"]
    comfort_cfg = config["comfort_reward"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ---- 速度跟踪奖励 ----
    v = np.linspace(0, 50, 200)
    speed_r = -((v - speed_cfg["target_speed"]) / speed_cfg["tolerance"]) ** 2
    speed_r = np.clip(speed_r, -100, 0)

    ax = axes[0]
    ax.plot(v, speed_r, "b-", linewidth=2)
    ax.axvline(speed_cfg["target_speed"], color="r", linestyle="--", alpha=0.7,
               label=f'Target={speed_cfg["target_speed"]} m/s')
    ax.fill_between(
        v, -10, speed_r,
        where=(v > speed_cfg["target_speed"] - speed_cfg["tolerance"])
        & (v < speed_cfg["target_speed"] + speed_cfg["tolerance"]),
        alpha=0.15, color="green",
    )
    ax.set_xlabel("Speed (m/s)", fontsize=11)
    ax.set_ylabel("Reward", fontsize=11)
    ax.set_title("Speed Tracking Reward", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- 距离控制奖励 ----
    d = np.linspace(0, 30, 200)
    safe = distance_cfg["safe_distance"]
    critical = distance_cfg["critical_distance"]
    dist_r = np.zeros_like(d)
    dist_r[d < critical] = -100
    mask = (d >= critical) & (d < safe)
    dist_r[mask] = -10 * (safe - d[mask])

    ax = axes[1]
    ax.plot(d, dist_r, "orange", linewidth=2)
    ax.axvline(critical, color="r", linestyle="--", alpha=0.7,
               label=f'Critical={critical} m')
    ax.axvline(safe, color="g", linestyle="--", alpha=0.7,
               label=f'Safe={safe} m')
    ax.set_xlabel("Distance (m)", fontsize=11)
    ax.set_ylabel("Reward", fontsize=11)
    ax.set_title("Distance Control Reward", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- 舒适性奖励 ----
    a = np.linspace(-5, 5, 200)
    comfort_r = -(a**2)
    comfort_r = np.clip(comfort_r, -(comfort_cfg["max_acceleration"]**2) * 2, 0)

    ax = axes[2]
    ax.plot(a, comfort_r, "green", linewidth=2)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvspan(
        -comfort_cfg["max_acceleration"], comfort_cfg["max_acceleration"],
        alpha=0.15, color="green", label=f'±{comfort_cfg["max_acceleration"]} m/s²'
    )
    ax.set_xlabel("Acceleration (m/s²)", fontsize=11)
    ax.set_ylabel("Reward", fontsize=11)
    ax.set_title("Comfort Reward", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Autonomous Driving Reward Function Analysis",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{prefix}reward_func_analysis.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")
    return save_path


def plot_scenario_rewards(
    rewards: dict,
    output_dir: str,
    prefix: str = "",
) -> str:
    """绘制场景仿真中的奖励变化曲线.

    Args:
        rewards: 包含 time, speed_reward, distance_reward, comfort_reward, total_reward 的字典
        output_dir: 输出目录路径

    Returns:
        保存的图片路径
    """
    t = rewards["time"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    titles = [
        "Speed Tracking Reward",
        "Distance Control Reward",
        "Comfort Reward",
        "Total Reward",
    ]
    keys = ["speed_reward", "distance_reward", "comfort_reward", "total_reward"]
    colors = ["blue", "orange", "green", "purple"]

    for i, (ax, title, key, color) in enumerate(zip(axes, titles, keys, colors)):
        ax.plot(t, rewards[key], color=color, linewidth=1.5)
        ax.set_ylabel("Reward", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    fig.suptitle("Reward Evolution During Simulation",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{prefix}reward_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")
    return save_path


def plot_speed_reward_comparison(
    output_dir: str,
    config: dict,
) -> str:
    """绘制不同目标速度下的速度奖励对比图.

    Args:
        output_dir: 输出目录路径
        config: 配置字典

    Returns:
        保存的图片路径
    """
    v = np.linspace(0, 50, 200)
    tolerance = config["speed_reward"]["tolerance"]

    targets = [20.0, 25.0, 30.0]
    colors = ["blue", "orange", "green"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for target, color in zip(targets, colors):
        r = -((v - target) / tolerance) ** 2
        r = np.clip(r, -100, 0)
        ax.plot(v, r, color=color, linewidth=2, label=f'Target={target} m/s')

    ax.set_xlabel("Speed (m/s)", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title("Speed Reward with Different Target Speeds",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "speed_reward_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")
    return save_path


def plot_distance_reward_comparison(
    output_dir: str,
    config: dict,
) -> str:
    """绘制不同安全距离下的距离奖励对比图.

    Args:
        output_dir: 输出目录路径
        config: 配置字典

    Returns:
        保存的图片路径
    """
    d = np.linspace(0, 30, 200)
    critical = config["distance_reward"]["critical_distance"]

    safe_values = [10.0, 15.0, 20.0]
    colors = ["orange", "red", "brown"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for safe, color in zip(safe_values, colors):
        r = np.zeros_like(d)
        r[d < critical] = -100
        mask = (d >= critical) & (d < safe)
        r[mask] = -10 * (safe - d[mask])
        ax.plot(d, r, color=color, linewidth=2, label=f'Safe={safe} m')

    ax.set_xlabel("Distance (m)", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title("Distance Reward with Different Safe Distances",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "distance_reward_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")
    return save_path
