"""模型评估 + 随机策略基准对比。

对训练好的 PPO 模型与随机策略基准在 BipedalWalker-v3 环境上各跑 N 个 episode，
量化对比 episode 奖励和 episode 长度，并绘制柱状图（带 std 误差棒）。

用法：
    python evaluate.py                          # 现训 5000 步 + 20 episodes 评估
    python evaluate.py --timesteps 20000        # 调整训练步数
    python evaluate.py --n-episodes 50          # 调整评估 episode 数
    python evaluate.py --model models/x.zip --stats models/x_stats.pkl   # 加载已有模型
    python evaluate.py --output myplot.png      # 指定输出 PNG 路径

输出：
    - 控制台：每种策略的 reward/length 摘要 + markdown 对比表
    - PNG：双栏柱状图（左 reward，右 episode length）含 std 误差棒
"""
import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

DEFAULT_TIMESTEPS = 5000
DEFAULT_N_EVAL = 20
DEFAULT_OUTPUT = "evaluate_comparison.png"
DEFAULT_MODEL_PATH = "models/ppo_eval.zip"
DEFAULT_STATS_PATH = "models/env_stats_eval.pkl"


def _wrap_vec_env_for_training():
    """训练用 env：Monitor + DummyVec + VecNormalize（含 reward 归一化）+ FrameStack。"""
    env_fns = [lambda: Monitor(gym.make("BipedalWalker-v3"), "logs/")]
    env = DummyVecEnv(env_fns)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)
    env = VecFrameStack(env, n_stack=4)
    return env


def _wrap_vec_env_for_eval(stats_path: str):
    """评估用 env：与训练同构 wrapper，但加载已保存的归一化统计且关闭训练态。"""
    env_fns = [lambda: Monitor(gym.make("BipedalWalker-v3"))]
    env = DummyVecEnv(env_fns)
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False  # 评估时返回真实环境奖励，便于人类理解
    env = VecFrameStack(env, n_stack=4)
    return env


def train_quick(timesteps: int):
    """短训 PPO + 保存模型 + 保存 VecNormalize 统计。"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    env = _wrap_vec_env_for_training()
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=32,
        gamma=0.99,
    )
    model.learn(total_timesteps=timesteps)
    model.save(DEFAULT_MODEL_PATH)
    # VecFrameStack 包装在最外层，env.venv 即 VecNormalize
    env.venv.save(DEFAULT_STATS_PATH)
    env.close()
    return DEFAULT_MODEL_PATH, DEFAULT_STATS_PATH


def evaluate_trained(model_path: str, stats_path: str, n_episodes: int):
    """加载模型 + 归一化统计，跑 n 个 episode，收集每轮奖励/长度。"""
    env = _wrap_vec_env_for_eval(stats_path)
    model = PPO.load(model_path, env=env)

    rewards, lengths = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, _ = env.step(action)
            ep_reward += float(reward[0])
            ep_length += 1
            done = bool(done_arr[0])
        rewards.append(ep_reward)
        lengths.append(ep_length)
    env.close()
    return rewards, lengths


def evaluate_random(n_episodes: int):
    """随机策略：从 action_space 均匀采样动作，跑 n 个 episode。"""
    env = gym.make("BipedalWalker-v3")
    rewards, lengths = [], []
    for _ in range(n_episodes):
        env.reset()
        terminated = False
        truncated = False
        ep_reward = 0.0
        ep_length = 0
        while not (terminated or truncated):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            ep_length += 1
        rewards.append(ep_reward)
        lengths.append(ep_length)
    env.close()
    return rewards, lengths


def print_stats(label: str, rewards, lengths):
    r = np.array(rewards)
    l = np.array(lengths)
    print(f"\n## {label}")
    print(f"   episodes: {len(r)}")
    print(
        f"   reward:   mean={r.mean():.2f}  std={r.std():.2f}  "
        f"min={r.min():.2f}  max={r.max():.2f}"
    )
    print(
        f"   length:   mean={l.mean():.1f}  std={l.std():.1f}  "
        f"min={int(l.min())}  max={int(l.max())}"
    )


def print_markdown_table(random_r, random_l, trained_r, trained_l, label_trained: str):
    print("\n## 对比摘要")
    print("| 策略 | 平均奖励 | 奖励标准差 | 最高奖励 | 平均轮长 | 最长轮 |")
    print("|---|---|---|---|---|---|")
    for label, r, l in [
        ("Random", random_r, random_l),
        (label_trained, trained_r, trained_l),
    ]:
        r_a = np.array(r)
        l_a = np.array(l)
        print(
            f"| {label} | {r_a.mean():.2f} | {r_a.std():.2f} | {r_a.max():.2f} | "
            f"{l_a.mean():.0f} | {int(l_a.max())} |"
        )


def plot_comparison(random_r, random_l, trained_r, trained_l, output: str, label_trained: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["Random policy", label_trained]
    colors = ["#888888", "#1f77b4"]

    # 左：reward 柱状图（带 std 误差棒）
    means = [np.mean(random_r), np.mean(trained_r)]
    stds = [np.std(random_r), np.std(trained_r)]
    bars = ax1.bar(labels, means, yerr=stds, color=colors, capsize=8, alpha=0.85, edgecolor="black")
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_ylabel("Mean episode reward (± std)")
    ax1.set_title(f"Reward comparison ({len(random_r)} eval episodes each)")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, m, s in zip(bars, means, stds):
        ymax = max(abs(m) + s for m, s in zip(means, stds))
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            m + (s + ymax * 0.03 if m >= 0 else -s - ymax * 0.06),
            f"{m:.1f}",
            ha="center",
            va="bottom" if m >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    # 右：episode length 柱状图
    means_l = [np.mean(random_l), np.mean(trained_l)]
    stds_l = [np.std(random_l), np.std(trained_l)]
    bars = ax2.bar(labels, means_l, yerr=stds_l, color=colors, capsize=8, alpha=0.85, edgecolor="black")
    ax2.set_ylabel("Mean episode length (± std)")
    ax2.set_title("Episode length comparison")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ymax_l = max(means_l[i] + stds_l[i] for i in range(2))
    for bar, m in zip(bars, means_l):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            m + ymax_l * 0.03,
            f"{m:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(
        "PPO vs Random Policy Baseline — BipedalWalker-v3", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="评估训练后 PPO 与随机策略基准在 BipedalWalker-v3 上的差异",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"训练步数（默认 {DEFAULT_TIMESTEPS}；若 --model 提供则忽略本项）",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=DEFAULT_N_EVAL,
        help=f"每种策略评估的 episode 数（默认 {DEFAULT_N_EVAL}）",
    )
    parser.add_argument("--model", type=str, default=None, help="已保存的 PPO 模型 .zip 路径")
    parser.add_argument("--stats", type=str, default=None, help="已保存的 VecNormalize 统计 .pkl 路径（与 --model 同时使用）")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help=f"输出 PNG 路径（默认 {DEFAULT_OUTPUT}）")
    args = parser.parse_args()

    # 选择模型来源：用户提供 vs 现训
    if args.model and args.stats:
        if not os.path.isfile(args.model) or not os.path.isfile(args.stats):
            print(f"[错误] --model 或 --stats 文件不存在", file=sys.stderr)
            sys.exit(1)
        model_path, stats_path = args.model, args.stats
        print(f"[1/3] 使用已保存模型: {model_path}  stats: {stats_path}")
        label_trained = "PPO (loaded)"
    else:
        if args.model or args.stats:
            print("[警告] --model 与 --stats 必须成对使用，本次改为现训", file=sys.stderr)
        print(f"[1/3] 现训 PPO {args.timesteps} timesteps ...")
        t0 = time.time()
        model_path, stats_path = train_quick(args.timesteps)
        print(f"      OK ({time.time() - t0:.1f}s)  model={model_path}  stats={stats_path}")
        label_trained = f"PPO ({args.timesteps} ts)"

    print(f"[2/3] 评估训练后 PPO（{args.n_episodes} episodes）...")
    t0 = time.time()
    trained_r, trained_l = evaluate_trained(model_path, stats_path, args.n_episodes)
    print(f"      OK ({time.time() - t0:.1f}s)")
    print_stats(label_trained, trained_r, trained_l)

    print(f"\n[3/3] 随机策略基准（{args.n_episodes} episodes）...")
    t0 = time.time()
    random_r, random_l = evaluate_random(args.n_episodes)
    print(f"      OK ({time.time() - t0:.1f}s)")
    print_stats("Random policy", random_r, random_l)

    print_markdown_table(random_r, random_l, trained_r, trained_l, label_trained)

    print(f"\n绘制对比图 -> {args.output}")
    plot_comparison(random_r, random_l, trained_r, trained_l, args.output, label_trained)
    print("完成。")


if __name__ == "__main__":
    main()
