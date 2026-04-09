"""无人机路径规划的轨迹可视化"""

import json
import os
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """在 3D 空间中可视化无人机轨迹"""

    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """
        初始化可视化工具。

        Args:
            figsize: 图形大小（宽度，高度）
        """
        self.figsize = figsize

    def plot_single_trajectory(
        self,
        positions: np.ndarray,
        target_points: Optional[np.ndarray] = None,
        obstacles: Optional[List[Dict]] = None,
        title: str = "无人机轨迹",
        save_path: Optional[str] = None,
    ):
        """
        绘制单个 3D 轨迹。

        Args:
            positions: 形状为 (N, 3) 的 3D 位置数组
            target_points: 形状为 (M, 3) 的目标航点数组
            obstacles: 包含"位置"和"半径"的障碍物字典列表
            title: 图形标题
            save_path: 保存图形的路径
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        # 绘制轨迹
        if len(positions) > 0:
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                "b-",
                linewidth=2,
                label="轨迹",
            )
            ax.scatter(
                positions[0, 0],
                positions[0, 1],
                positions[0, 2],
                c="green",
                s=100,
                marker="o",
                label="起点",
            )
            ax.scatter(
                positions[-1, 0],
                positions[-1, 1],
                positions[-1, 2],
                c="red",
                s=100,
                marker="s",
                label="终点",
            )

        # 绘制目标点
        if target_points is not None and len(target_points) > 0:
            ax.scatter(
                target_points[:, 0],
                target_points[:, 1],
                target_points[:, 2],
                c="gold",
                s=200,
                marker="*",
                label="目标",
                edgecolors="black",
                linewidth=1,
            )

        # Plot obstacles
        if obstacles:
            for obs in obstacles:
                pos = obs.get("position", [0, 0, 0])
                radius = obs.get("radius", 1.0)

                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 10)
                x_obs = radius * np.outer(np.cos(u), np.sin(v)) + pos[0]
                y_obs = radius * np.outer(np.sin(u), np.sin(v)) + pos[1]
                z_obs = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]

                ax.plot_surface(x_obs, y_obs, z_obs, alpha=0.3, color="red")

        # Labels and formatting
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Set equal aspect ratio
        max_range = (
            np.array(
                [
                    positions[:, 0].max() - positions[:, 0].min(),
                    positions[:, 1].max() - positions[:, 1].min(),
                    positions[:, 2].max() - positions[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"图形已保存到 {save_path}")

        return fig, ax

    def plot_multiple_trajectories(
        self,
        trajectories_list: List[Tuple[np.ndarray, str]],
        target_points: Optional[np.ndarray] = None,
        title: str = "多条轨迹",
        save_path: Optional[str] = None,
    ):
        """
        在同一图形上绘制多个 3D 轨迹。

        Args:
            trajectories_list: List of (positions, label) tuples
            target_points: Target waypoints
            title: Plot title
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        colors = ["blue", "red", "green", "orange", "purple"]

        for i, (positions, label) in enumerate(trajectories_list):
            color = colors[i % len(colors)]
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color=color,
                linewidth=2,
                label=label,
            )

        # Plot target points
        if target_points is not None and len(target_points) > 0:
            ax.scatter(
                target_points[:, 0],
                target_points[:, 1],
                target_points[:, 2],
                c="gold",
                s=200,
                marker="*",
                label="Targets",
                edgecolors="black",
                linewidth=1,
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")

        return fig, ax

    def plot_trajectory_metrics(
        self,
        positions: np.ndarray,
        rewards: Optional[np.ndarray] = None,
        collisions: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ):
        """
        Plot trajectory metrics over time.

        Args:
            positions: Array of shape (N, 3)
            rewards: Array of shape (N,) with step rewards
            collisions: Boolean array indicating collisions
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        # Distance from origin over time
        distances = np.linalg.norm(positions, axis=1)
        axes[0, 0].plot(distances)
        axes[0, 0].set_ylabel("Distance from Origin (m)")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_title("Distance from Origin")
        axes[0, 0].grid(True, alpha=0.3)

        # Height over time
        axes[0, 1].plot(positions[:, 2])
        axes[0, 1].set_ylabel("Height Z (m)")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_title("Altitude")
        axes[0, 1].grid(True, alpha=0.3)

        # Rewards
        if rewards is not None:
            axes[1, 0].plot(rewards)
            axes[1, 0].set_ylabel("Reward")
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_title("Reward over Time")
            axes[1, 0].grid(True, alpha=0.3)

        # Collisions
        if collisions is not None:
            axes[1, 1].scatter(
                np.where(collisions)[0],
                np.ones(np.sum(collisions)),
                c="red",
                s=50,
                label="Collision",
            )
            axes[1, 1].set_ylabel("Collision")
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_title("Collision Indicators")
            axes[1, 1].set_ylim([0, 2])
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved metrics plot to {save_path}")

        return fig, axes


def load_trajectories_from_json(json_path: str) -> Dict:
    """Load trajectories from JSON file"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def visualize_evaluation_results(
    results_dir: str,
    traj_file: str = "trajectories.json",
    viz_dir: Optional[str] = None,
):
    """
    Visualize evaluation results.

    Args:
        results_dir: Directory containing evaluation results
        traj_file: Name of trajectories JSON file
        viz_dir: Directory to save visualizations
    """
    import matplotlib.pyplot as plt

    if viz_dir is None:
        viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Load trajectories
    traj_path = os.path.join(results_dir, traj_file)
    if not os.path.exists(traj_path):
        logger.warning(f"Trajectory file not found: {traj_path}")
        return

    trajectories = load_trajectories_from_json(traj_path)
    logger.info(f"Loaded {len(trajectories)} trajectories")

    visualizer = TrajectoryVisualizer()

    # Visualize each episode
    for traj_data in trajectories:
        ep_num = traj_data.get("episode", 0)
        positions = np.array(traj_data.get("positions", []))

        if len(positions) > 0:
            save_path = os.path.join(viz_dir, f"trajectory_episode_{ep_num:03d}.png")
            visualizer.plot_single_trajectory(
                positions, title=f"Episode {ep_num}", save_path=save_path
            )
            plt.close("all")

    # Visualize all trajectories together
    if len(trajectories) > 1:
        traj_list = [
            (np.array(t.get("positions", [])), f"Ep. {t.get('episode', 0)}")
            for t in trajectories[:5]  # Limit to first 5
        ]
        save_path = os.path.join(viz_dir, "all_trajectories.png")
        visualizer.plot_multiple_trajectories(
            traj_list, title="Multiple Episode Trajectories", save_path=save_path
        )
        plt.close("all")

    logger.info(f"Visualizations saved to {viz_dir}")


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize drone trajectories")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save visualizations"
    )
    parser.add_argument(
        "--traj-file",
        type=str,
        default="trajectories.json",
        help="Name of trajectories JSON file",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    visualize_evaluation_results(
        results_dir=args.results_dir, traj_file=args.traj_file, viz_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
