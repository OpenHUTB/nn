"""
项目7超简版：单图碰撞风险分析
只生成一个核心的热力图
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def super_simple_collision_analysis():
    """超简版单图碰撞分析"""
    print("正在生成碰撞风险单图...")

    # 生成模拟数据
    np.random.seed(42)
    n_points = 500

    # 工作空间点
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = 0.5 + 0.3 * np.random.randn(n_points)

    x = (0.8 + 0.3 * r * np.cos(phi)) * np.cos(theta)
    y = (0.8 + 0.3 * r * np.cos(phi)) * np.sin(theta)
    z = 0.3 * r * np.sin(phi) + 0.5
    points = np.vstack([x, y, z]).T

    # 计算碰撞风险（简化算法）
    risks = []
    for point in points:
        risk = 0.0

        # 到墙壁的风险
        wall_dist = abs(point[0] - 0.8)
        if wall_dist < 0.2:
            risk += (0.2 - wall_dist) / 0.2

        # 到中心柱子的风险
        center_dist = np.sqrt(point[0] ** 2 + point[1] ** 2)
        if center_dist < 0.3:
            risk += (0.3 - center_dist) / 0.3

        # 底部安全，顶部危险
        if point[2] > 0.9:
            risk += 0.5

        risk = min(1.0, max(0.0, risk))
        risks.append(risk)

    risks = np.array(risks)

    # 创建单个图表
    fig = plt.figure(figsize=(14, 8))

    # 1. 3D风险图
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点，颜色表示风险
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=risks, cmap='RdYlGn_r',  # 红-黄-绿反向
                         alpha=0.7, s=30, edgecolors='none')

    # 添加障碍物标记
    # 墙壁
    ax.plot([0.8, 0.8], [-1, 1], [0, 0], 'k-', linewidth=3, alpha=0.5, label='墙壁')
    # 中心障碍物
    ax.scatter([0, 0], [0, 0], [0.2, 0.8], c='black', s=100, marker='^', label='障碍物')

    # 设置图表
    ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)
    ax.set_title('机械臂工作空间碰撞风险分析', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=25, azim=45)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label('碰撞风险\n(0=安全, 1=危险)', fontsize=11, rotation=0, labelpad=20)

    # 添加统计信息
    safe_ratio = np.sum(risks < 0.3) / len(risks) * 100
    danger_ratio = np.sum(risks >= 0.7) / len(risks) * 100

    info_text = f'安全点: {safe_ratio:.1f}%\n危险点: {danger_ratio:.1f}%\n总点数: {len(risks)}'

    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 保存和显示
    plt.tight_layout()
    plt.savefig('single_collision_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 打印简单统计
    print("\n" + "=" * 50)
    print("碰撞风险分析结果")
    print("=" * 50)
    print(f"安全区域占比: {safe_ratio:.1f}%")
    print(f"危险区域占比: {danger_ratio:.1f}%")
    print(f"平均风险值: {np.mean(risks):.3f}")
    print("=" * 50)
    print("图表已保存: single_collision_analysis.png")

    return risks


# 直接运行
if __name__ == "__main__":
    super_simple_collision_analysis()