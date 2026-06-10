"""车辆黑匣子数据可视化工具。

读取 CARLA 黑匣子 CSV 数据，绘制速度、加速度、转向、油门/刹车四合一分析图。
支持从文件读取或自动生成示例数据进行演示。

用法:
    python plot_blackbox.py                      # 使用示例数据
    python plot_blackbox.py --csv blackbox.csv   # 指定 CSV 文件
    python plot_blackbox.py --save result.png    # 保存图片
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============ 样式配置 ============
STYLE = {
    "speed": {"color": "#2196F3", "linewidth": 2},
    "accel_x": {"color": "#F44336", "linewidth": 1.5},
    "accel_y": {"color": "#4CAF50", "linewidth": 1.5},
    "steer": {"color": "#FF9800", "linewidth": 2},
    "throttle": {"color": "#4CAF50", "linewidth": 2},
    "brake": {"color": "#F44336", "linewidth": 2},
}


def generate_sample_data(duration=60, fps=20) -> pd.DataFrame:
    """生成模拟驾驶数据（城市路况：起步-巡航-刹车-转弯）。"""
    n = duration * fps
    t = np.linspace(0, duration, n)

    # 速度：起步加速 → 巡航 → 减速 → 再加速
    speed = np.zeros(n)
    speed[:n // 4] = np.linspace(0, 50, n // 4) + np.random.normal(0, 1.5, n // 4)
    speed[n // 4:n // 2] = 50 + np.random.normal(0, 2, n // 4)
    speed[n // 2:3 * n // 4] = np.linspace(50, 15, n // 4) + np.random.normal(0, 1.5, n // 4)
    speed[3 * n // 4:] = np.linspace(15, 45, n - 3 * n // 4) + np.random.normal(0, 1.5, n - 3 * n // 4)
    speed = np.clip(speed, 0, None)

    # 加速度：从速度导出 + 噪声
    accel_x = np.gradient(speed / 3.6, t) + np.random.normal(0, 0.3, n)
    accel_y = np.sin(2 * np.pi * 0.05 * t) * 2 + np.random.normal(0, 0.5, n)

    # 转向：巡航段有小幅修正，中间有一次大转弯
    steer = np.sin(2 * np.pi * 0.08 * t) * 0.15 + np.random.normal(0, 0.05, n)
    steer[n // 3:n // 3 + fps * 3] += np.sin(np.linspace(0, np.pi, fps * 3)) * 0.6
    steer = np.clip(steer, -1, 1)

    # 油门/刹车
    throttle = np.clip(speed / 60, 0, 1) + np.random.normal(0, 0.03, n)
    brake = np.clip(-accel_x / 5, 0, 1) + np.random.normal(0, 0.02, n)
    throttle = np.clip(throttle, 0, 1)
    brake = np.clip(brake, 0, 1)

    return pd.DataFrame({
        "time": np.round(t, 2),
        "speed": np.round(speed, 2),
        "steer": np.round(steer, 3),
        "throttle": np.round(throttle, 3),
        "brake": np.round(brake, 3),
        "accel_x": np.round(accel_x, 2),
        "accel_y": np.round(accel_y, 2),
        "accel_z": np.round(np.random.normal(9.8, 0.1, n), 2),
    })


def plot_speed(ax, df):
    """绘制速度曲线并标注统计信息。"""
    ax.plot(df["time"], df["speed"], label="Speed (km/h)", **STYLE["speed"])
    ax.fill_between(df["time"], df["speed"], alpha=0.15, color=STYLE["speed"]["color"])

    avg_speed = df["speed"].mean()
    max_speed = df["speed"].max()
    ax.axhline(avg_speed, color="#9E9E9E", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(f"Avg: {avg_speed:.1f}", xy=(df["time"].iloc[-1], avg_speed),
                fontsize=9, color="#616161", va="bottom")
    ax.annotate(f"Max: {max_speed:.1f}", xy=(df["time"].iloc[df["speed"].idxmax()], max_speed),
                fontsize=9, color="#F44336", va="bottom")

    ax.set_title("Vehicle Speed", fontweight="bold")
    ax.set_ylabel("Speed (km/h)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_acceleration(ax, df):
    """绘制三轴加速度。"""
    ax.plot(df["time"], df["accel_x"], label="Longitudinal (X)", **STYLE["accel_x"])
    ax.plot(df["time"], df["accel_y"], label="Lateral (Y)", **STYLE["accel_y"])
    ax.fill_between(df["time"], df["accel_x"], alpha=0.1, color=STYLE["accel_x"]["color"])
    ax.fill_between(df["time"], df["accel_y"], alpha=0.1, color=STYLE["accel_y"]["color"])

    ax.set_title("Acceleration", fontweight="bold")
    ax.set_ylabel("m/s^2")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_steering(ax, df):
    """绘制转向角并标注急转区域。"""
    ax.plot(df["time"], df["steer"], label="Steering", **STYLE["steer"])
    ax.fill_between(df["time"], df["steer"], alpha=0.15, color=STYLE["steer"]["color"])

    # 标注急转区域
    sharp = df["steer"].abs() > 0.4
    if sharp.any():
        ax.fill_between(df["time"], -1, 1, where=sharp, alpha=0.15, color="#F44336", label="Sharp turn")

    ax.set_title("Steering Angle", fontweight="bold")
    ax.set_ylabel("Steer (-1 ~ 1)")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_controls(ax, df):
    """绘制油门和刹车。"""
    ax.plot(df["time"], df["throttle"], label="Throttle", **STYLE["throttle"])
    ax.plot(df["time"], df["brake"], label="Brake", **STYLE["brake"])
    ax.fill_between(df["time"], df["throttle"], alpha=0.1, color=STYLE["throttle"]["color"])
    ax.fill_between(df["time"], df["brake"], alpha=0.1, color=STYLE["brake"]["color"])

    ax.set_title("Throttle & Brake", fontweight="bold")
    ax.set_ylabel("Control (0 ~ 1)")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_blackbox(df, save_path=None):
    """绘制四合一黑匣子分析图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Vehicle Blackbox Analysis", fontsize=16, fontweight="bold", y=0.98)

    plot_speed(axes[0, 0], df)
    plot_acceleration(axes[0, 1], df)
    plot_steering(axes[1, 0], df)
    plot_controls(axes[1, 1], df)

    for ax in axes.flat:
        ax.set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Vehicle blackbox data visualization")
    parser.add_argument("--csv", type=str, default=None, help="Path to blackbox CSV file")
    parser.add_argument("--save", type=str, default=None, help="Save chart to file (e.g. result.png)")
    parser.add_argument("--demo", action="store_true", help="Use generated sample data")
    args = parser.parse_args()

    if args.csv and Path(args.csv).exists():
        df = pd.read_csv(args.csv)
        print(f"Loaded {len(df)} records from {args.csv}")
    else:
        if args.csv:
            print(f"File not found: {args.csv}, using sample data instead")
        else:
            print("No CSV specified, using generated sample data")
        df = generate_sample_data()
        print(f"Generated {len(df)} sample records ({df['time'].iloc[-1]:.0f}s)")

    # 打印统计摘要
    print(f"\n--- Summary ---")
    print(f"Speed: avg={df['speed'].mean():.1f}, max={df['speed'].max():.1f} km/h")
    print(f"Accel X: min={df['accel_x'].min():.1f}, max={df['accel_x'].max():.1f} m/s^2")
    print(f"Steering: range=[{df['steer'].min():.3f}, {df['steer'].max():.3f}]")

    plot_blackbox(df, save_path=args.save)


if __name__ == "__main__":
    main()