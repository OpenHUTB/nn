"""
机械臂工作空间3D可视化
展示机械臂末端可达的所有位置，视觉效果很酷
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull

def workspace_visualization():
    """生成机械臂工作空间3D可视化"""
    # 模拟工作空间点云（使用逆运动学采样）
    np.random.seed(42)
    n_points = 2000

    # 生成工作空间点（模拟机械臂可达范围）
    # 这里用一个甜甜圈形状+椭球体模拟
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)

    # 主工作空间（甜甜圈形状）
    r = 0.5 + 0.2 * np.random.randn(n_points)
    x_torus = (0.8 + 0.3 * r * np.cos(phi)) * np.cos(theta)
    y_torus = (0.8 + 0.3 * r * np.cos(phi)) * np.sin(theta)
    z_torus = 0.3 * r * np.sin(phi) + 0.5

    # 添加一个椭球区域（表示额外的工作空间）
    n_ellipsoid = 500
    u = np.random.rand(n_ellipsoid)
    v = np.random.rand(n_ellipsoid)
    theta_e = 2 * np.pi * u
    phi_e = np.arccos(2 * v - 1)

    a, b, c = 0.3, 0.2, 0.4  # 椭球半轴
    x_ell = a * np.sin(phi_e) * np.cos(theta_e) - 0.2
    y_ell = b * np.sin(phi_e) * np.sin(theta_e) + 0.3
    z_ell = c * np.cos(phi_e) + 0.7

    # 合并所有点
    x = np.concatenate([x_torus, x_ell])
    y = np.concatenate([y_torus, y_ell])
    z = np.concatenate([z_torus, z_ell])

    # 创建3D图形
    fig = plt.figure(figsize=(14, 10))

    # 1. 3D散点图
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(x, y, z, c=z, cmap='plasma', alpha=0.3,
                         s=10, marker='o', edgecolors='none')

    ax1.set_xlabel('X (m)', fontsize=10, labelpad=10)
    ax1.set_ylabel('Y (m)', fontsize=10, labelpad=10)
    ax1.set_zlabel('Z (m)', fontsize=10, labelpad=10)
    ax1.set_title('工作空间点云（3D视图）', fontsize=12, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.view_init(elev=25, azim=45)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6, pad=0.1)
    cbar.set_label('高度 (m)', fontsize=9)

    # 2. 俯视图（XY平面）
    ax2 = fig.add_subplot(222)
    hexbin = ax2.hexbin(x, y, C=z, gridsize=30, cmap='YlOrRd', alpha=0.9)
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_title('工作空间俯视图（XY平面）', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # 添加颜色条
    cbar2 = plt.colorbar(hexbin, ax=ax2, shrink=0.8)
    cbar2.set_label('平均高度 (m)', fontsize=9)

    # 3. 侧视图（XZ平面）
    ax3 = fig.add_subplot(223)
    hb = ax3.hexbin(x, z, gridsize=30, cmap='Blues', alpha=0.9)
    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Z (m)', fontsize=10)
    ax3.set_title('工作空间侧视图（XZ平面）', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    cbar3 = plt.colorbar(hb, ax=ax3, shrink=0.8)
    cbar3.set_label('点密度', fontsize=9)

    # 4. 前视图（YZ平面）
    ax4 = fig.add_subplot(224)
    hb4 = ax4.hexbin(y, z, gridsize=30, cmap='Greens', alpha=0.9)
    ax4.set_xlabel('Y (m)', fontsize=10)
    ax4.set_ylabel('Z (m)', fontsize=10)
    ax4.set_title('工作空间前视图（YZ平面）', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    cbar4 = plt.colorbar(hb4, ax=ax4, shrink=0.8)
    cbar4.set_label('点密度', fontsize=9)

    plt.suptitle('6-DOF机械臂工作空间可视化分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('workspace_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='#F5F7FA')
    plt.show()

    # 额外：创建凸包和体积计算
    create_convex_hull(x, y, z)

def create_convex_hull(x, y, z):
    """计算工作空间的凸包和体积"""
    # 采样点用于凸包计算
    points = np.vstack([x[::20], y[::20], z[::20]]).T  # 下采样

    # 计算凸包
    hull = ConvexHull(points)

    # 创建凸包可视化
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制凸包
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # 闭合三角形
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2],
               'b-', alpha=0.3, linewidth=0.5)

    # 绘制原始点
    ax.scatter(x[::50], y[::50], z[::50], c='red', alpha=0.3, s=10,
              label='工作空间点')

    # 计算体积
    volume = hull.volume
    area = hull.area

    # 添加文本标注
    ax.text2D(0.05, 0.95, f'工作空间体积: {volume:.3f} m³\n'
                         f'表面积: {area:.3f} m²\n'
                         f'采样点数: {len(points)}',
             transform=ax.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title('工作空间凸包分析', fontsize=14, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=30)

    plt.tight_layout()
    plt.savefig('workspace_convex_hull.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    workspace_visualization()