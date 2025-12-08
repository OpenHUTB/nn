import argparse
from pathlib import Path

from uitb import Simulator


def make_simulator(task_name: str):
    """
    根据任务名称返回对应的 simulator 环境。

    task_name: "pointing"、"tracking" 或 "choice_reaction"
    """
    project_root = Path(__file__).resolve().parent

    if task_name == "pointing":
        sim_dir = project_root / "simulators" / "mobl_arms_index_pointing"
    elif task_name == "tracking":
        sim_dir = project_root / "simulators" / "mobl_arms_index_tracking"
    elif task_name == "choice_reaction":
        # 🔹 新增 Choice Reaction 任务入口
        sim_dir = project_root / "simulators" / "mobl_arms_index_choice_reaction"
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # README 里说明：Simulator.get(simulator_folder) 会返回一个 gym 风格的环境
    # 可以直接调用 reset / step / render 等方法。
    simulator = Simulator.get(str(sim_dir))
    return simulator


def run_episodes(env, num_episodes: int, max_steps: int):
    """
    用随机动作跑若干个 episode，主要是演示 env 的使用。
    """
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        episode_reward = 0.0

        print(f"\n=== Episode {ep + 1}/{num_episodes} ===")

        while not done and step < max_steps:
            # 这里先用随机策略，作业如果需要你可以换成自己的策略
            action = env.action_space.sample()

            # gymnasium 接口：obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1

            # 如果你想看实时画面（而不是只出视频），可以打开这一行：
            # env.render()

        print(f"Episode reward: {episode_reward:.3f} (steps: {step})")


def main():
    parser = argparse.ArgumentParser(
        description="User-in-the-Box demo for Pointing, Tracking & Choice Reaction"
    )
    parser.add_argument(
        "--task",
        # 🔹 在命令行参数里加入 choice_reaction 选项
        choices=["pointing", "tracking", "choice_reaction"],
        default="pointing",
        help="选择要运行的任务：pointing / tracking / choice_reaction",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="要运行的 episode 数",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="每个 episode 最多运行多少步（防止无限循环）",
    )
    args = parser.parse_args()

    env = make_simulator(args.task)
    try:
        run_episodes(env, args.num_episodes, args.max_steps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
