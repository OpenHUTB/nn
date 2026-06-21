"""学习曲线可视化：跑一段短训 PPO，自动绘制 reward 学习曲线。

用法:
    python learning_curve.py                       # 默认 5000 timesteps
    python learning_curve.py 10000                 # 指定 timesteps
    python learning_curve.py 5000 demo.png         # 指定 timesteps + 输出文件

输出：
    - 控制台：训练摘要 markdown 表格（总轮数、总步数、均值/最大/末段奖励、平均轮长）
    - 图：双栏 PNG（左图 reward vs episode + 滚动均值；右图 episode 长度 vs episode）
"""
import glob
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

DEFAULT_TIMESTEPS = 5000
DEFAULT_OUTPUT = "learning_curve.png"
LOGS_DIR = "logs"


def train_ppo(timesteps: int):
    os.makedirs(LOGS_DIR, exist_ok=True)
    # 清空旧 monitor 日志，避免和之前训练混在一起
    for old in glob.glob(os.path.join(LOGS_DIR, "*.monitor.csv")):
        os.remove(old)

    def make_env():
        env = gym.make("BipedalWalker-v3")
        env = Monitor(env, LOGS_DIR + "/")
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=128, batch_size=32, gamma=0.99)
    model.learn(total_timesteps=timesteps)
    return env, model


def read_monitor_logs() -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(LOGS_DIR, "*.monitor.csv"))
    if not csv_files:
        csv_files = glob.glob(os.path.join(LOGS_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("Monitor 日志未找到，可能训练步数太少没有完整 episode")
    # Monitor 第一行是 #{json} 注释头，跳过
    return pd.read_csv(csv_files[0], skiprows=1)


def plot_curve(df: pd.DataFrame, output: str, timesteps: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    episodes = np.arange(1, len(df) + 1)
    rewards = df["r"].values
    lengths = df["l"].values
    # 滚动窗口至少 5，便于小数据量也能看出平滑趋势
    window = max(5, len(df) // 5)

    # 左：reward
    ax1.scatter(episodes, rewards, color="#cccccc", s=14, alpha=0.55, label="per-episode reward")
    ax1.plot(
        episodes,
        pd.Series(rewards).rolling(window=window, min_periods=1).mean(),
        color="#1f77b4",
        linewidth=2.2,
        label=f"rolling mean (window={window})",
    )
    ax1.axhline(0, color="#888888", linewidth=0.6, linestyle="--")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode reward")
    ax1.set_title(f"PPO learning curve on BipedalWalker-v3 ({timesteps} timesteps, {len(df)} episodes)")
    ax1.legend(loc="lower right")
    ax1.grid(linestyle="--", alpha=0.4)

    # 右：episode length
    ax2.scatter(episodes, lengths, color="#cccccc", s=14, alpha=0.55, label="per-episode length")
    ax2.plot(
        episodes,
        pd.Series(lengths).rolling(window=window, min_periods=1).mean(),
        color="#d62728",
        linewidth=2.2,
        label=f"rolling mean (window={window})",
    )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode length (steps)")
    ax2.set_title("Episode length over training")
    ax2.legend(loc="upper right")
    ax2.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output, dpi=100)
    plt.close(fig)


def print_stats(df: pd.DataFrame, timesteps: int) -> None:
    print("\n## 训练摘要")
    print("| 指标 | 值 |")
    print("|---|---|")
    print(f"| 训练总步数 | {timesteps} |")
    print(f"| 完成的 episode 数 | {len(df)} |")
    print(f"| 平均 episode 奖励 | {df['r'].mean():.2f} |")
    print(f"| 最高 episode 奖励 | {df['r'].max():.2f} |")
    print(f"| 末 10 个 episode 平均奖励 | {df['r'].tail(10).mean():.2f} |")
    print(f"| 平均 episode 长度 | {df['l'].mean():.1f} 步 |")
    print(f"| 最长 episode | {df['l'].max()} 步 |")


def main():
    timesteps = int(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_TIMESTEPS
    output = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_OUTPUT

    print(f"[1/3] PPO 短训 {timesteps} timesteps ...")
    t0 = time.time()
    train_ppo(timesteps)
    print(f"      OK  ({time.time() - t0:.1f}s)")

    print("[2/3] 读取 Monitor 日志 ...")
    df = read_monitor_logs()
    print(f"      OK  ({len(df)} 个完整 episode)")

    print_stats(df, timesteps)

    print(f"\n[3/3] 绘制学习曲线 → {output} ...")
    plot_curve(df, output, timesteps)
    print("      OK")
    print("完成。")


if __name__ == "__main__":
    main()
