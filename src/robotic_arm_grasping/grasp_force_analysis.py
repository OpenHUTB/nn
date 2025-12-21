"""
项目3：三指抓取器力分析可视化
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Wedge
from scipy import stats

def grasp_force_analysis():
    """抓取力分析与可视化"""
    np.random.seed(42)

    # 模拟抓取数据
    n_samples = 1000
    time = np.linspace(0, 5, n_samples)

    # 模拟三个手指的力数据
    finger1_force = 5 + 2 * np.sin(time * 3) + 0.5 * np.random.randn(n_samples)
    finger2_force = 6 + 1.5 * np.cos(time * 2.5) + 0.4 * np.random.randn(n_samples)
    finger3_force = 4.5 + 1.8 * np.sin(time * 2 + 1) + 0.6 * np.random.randn(n_samples)

    # 总力
    total_force = finger1_force + finger2_force + finger3_force

    # 创建图表
    fig = plt.figure(figsize=(16, 12))

    # 1. 力随时间变化（折线图）
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(time, finger1_force, 'r-', linewidth=2, alpha=0.8, label='手指1')
    ax1.plot(time, finger2_force, 'g-', linewidth=2, alpha=0.8, label='手指2')
    ax1.plot(time, finger3_force, 'b-', linewidth=2, alpha=0.8, label='手指3')
    ax1.plot(time, total_force, 'k--', linewidth=2.5, alpha=0.9, label='总力')

    ax1.set_xlabel('时间 (s)', fontsize=10)
    ax1.set_ylabel('力 (N)', fontsize=10)
    ax1.set_title('各手指力随时间变化', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(time, total_force.min(), total_force, alpha=0.1, color='gray')

    # 2. 力分布直方图
    ax2 = plt.subplot(3, 3, 2)
    bins = np.linspace(0, 15, 30)
    ax2.hist(finger1_force, bins=bins, alpha=0.6, color='red',
             label='手指1', density=True)
    ax2.hist(finger2_force, bins=bins, alpha=0.6, color='green',
             label='手指2', density=True)
    ax2.hist(finger3_force, bins=bins, alpha=0.6, color='blue',
             label='手指3', density=True)

    # 添加KDE曲线
    for force, color, label in zip([finger1_force, finger2_force, finger3_force],
                                  ['red', 'green', 'blue'],
                                  ['手指1', '手指2', '手指3']):
        kde = stats.gaussian_kde(force)
        x_kde = np.linspace(force.min(), force.max(), 100)
        ax2.plot(x_kde, kde(x_kde), color=color, linewidth=2, label=f'{label} KDE')

    ax2.set_xlabel('力 (N)', fontsize=10)
    ax2.set_ylabel('概率密度', fontsize=10)
    ax2.set_title('力分布直方图与KDE', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. 箱线图 - 修复版本兼容性问题
    ax3 = plt.subplot(3, 3, 3)
    force_data = [finger1_force, finger2_force, finger3_force]

    # 检查Matplotlib版本并选择正确的参数名
    import matplotlib
    mpl_version = matplotlib.__version__
    print(f"Matplotlib 版本: {mpl_version}")

    # 尝试使用新的参数名，如果失败则回退到旧的
    try:
        # 尝试使用新的tick_labels参数
        bp = ax3.boxplot(force_data, patch_artist=True,
                        tick_labels=['手指1', '手指2', '手指3'],
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(color='gray', linewidth=1.5),
                        capprops=dict(color='gray', linewidth=1.5))
    except TypeError:
        # 如果失败，使用旧的labels参数
        bp = ax3.boxplot(force_data, patch_artist=True,
                        labels=['手指1', '手指2', '手指3'],
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(color='gray', linewidth=1.5),
                        capprops=dict(color='gray', linewidth=1.5))

    # 设置颜色
    colors = ['lightcoral', 'lightgreen', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_ylabel('力 (N)', fontsize=10)
    ax3.set_title('力统计箱线图', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 添加均值点
    means = [np.mean(f) for f in force_data]
    ax3.scatter(range(1, 4), means, color='black', s=80, marker='D',
               label='均值', zorder=5)

    # 4. 力相关矩阵热图
    ax4 = plt.subplot(3, 3, 4)
    forces = np.vstack([finger1_force, finger2_force, finger3_force])
    corr_matrix = np.corrcoef(forces)

    im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    # 添加数值标签
    for i in range(3):
        for j in range(3):
            text = ax4.text(j, i, f'{corr_matrix[i, j]:.2f}',
                           ha='center', va='center', color='white',
                           fontsize=11, fontweight='bold')

    ax4.set_xticks(range(3))
    ax4.set_yticks(range(3))
    ax4.set_xticklabels(['手指1', '手指2', '手指3'], fontsize=10)
    ax4.set_yticklabels(['手指1', '手指2', '手指3'], fontsize=10)
    ax4.set_title('手指力相关性热图', fontsize=12, fontweight='bold')

    # 添加颜色条
    plt.colorbar(im, ax=ax4, shrink=0.8)

    # 5. 力平衡雷达图
    ax5 = plt.subplot(3, 3, 5, polar=True)
    categories = ['力大小', '稳定性', '响应速度', '同步性', '效率']
    N = len(categories)

    # 计算各项指标（模拟）
    values_f1 = [np.mean(finger1_force)/10, 0.8, 0.7, 0.6, 0.75]
    values_f2 = [np.mean(finger2_force)/10, 0.9, 0.8, 0.7, 0.85]
    values_f3 = [np.mean(finger3_force)/10, 0.7, 0.6, 0.65, 0.7]

    # 封闭多边形
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_f1 += values_f1[:1]
    values_f2 += values_f2[:1]
    values_f3 += values_f3[:1]
    angles += angles[:1]

    ax5.plot(angles, values_f1, 'o-', linewidth=2, color='red',
             label='手指1', alpha=0.7)
    ax5.fill(angles, values_f1, alpha=0.1, color='red')

    ax5.plot(angles, values_f2, 'o-', linewidth=2, color='green',
             label='手指2', alpha=0.7)
    ax5.fill(angles, values_f2, alpha=0.1, color='green')

    ax5.plot(angles, values_f3, 'o-', linewidth=2, color='blue',
             label='手指3', alpha=0.7)
    ax5.fill(angles, values_f3, alpha=0.1, color='blue')

    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories, fontsize=9)
    ax5.set_ylim(0, 1)
    ax5.set_title('手指性能雷达图', fontsize=12, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. 力时频分析 - 修复数组维度问题
    ax6 = plt.subplot(3, 3, 6)

    # 创建正确的2D时频数据
    t = np.linspace(0, 5, 200)
    f = np.linspace(0, 10, 100)
    T, F = np.meshgrid(t, f)

    # 修复：确保Z是2D数组
    # 原来的错误：使用了[None, :]创建了额外的维度
    # 修复：直接使用广播创建2D数组
    Z = np.sin(2 * np.pi * F * T) * np.exp(-0.1 * T) * np.sin(2 * np.pi * 2 * T)

    # 验证Z是2D
    print(f"Z的形状: {Z.shape}, 应该是(100, 200)")

    contour = ax6.contourf(T, F, np.abs(Z), 20, cmap='hot', alpha=0.9)
    ax6.set_xlabel('时间 (s)', fontsize=10)
    ax6.set_ylabel('频率 (Hz)', fontsize=10)
    ax6.set_title('力信号时频分析', fontsize=12, fontweight='bold')

    plt.colorbar(contour, ax=ax6, shrink=0.8, label='幅度')

    # 7. 累积力分布
    ax7 = plt.subplot(3, 3, 7)

    for force, color, label in zip([finger1_force, finger2_force, finger3_force],
                                  ['red', 'green', 'blue'],
                                  ['手指1', '手指2', '手指3']):
        sorted_force = np.sort(force)
        cdf = np.arange(1, len(sorted_force) + 1) / len(sorted_force)
        ax7.plot(sorted_force, cdf, color=color, linewidth=2, label=label)

    ax7.set_xlabel('力 (N)', fontsize=10)
    ax7.set_ylabel('累积概率', fontsize=10)
    ax7.set_title('累积分布函数 (CDF)', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    # 8. 力向量图（模拟）
    ax8 = plt.subplot(3, 3, 8)

    # 模拟三个力的向量
    angles = [0, 120, 240]  # 三个手指的角度（度）
    lengths = [np.mean(finger1_force), np.mean(finger2_force), np.mean(finger3_force)]

    for angle, length, color, label in zip(angles, lengths,
                                          ['red', 'green', 'blue'],
                                          ['手指1', '手指2', '手指3']):
        rad = np.deg2rad(angle)
        dx = length * np.cos(rad) / 10
        dy = length * np.sin(rad) / 10

        ax8.arrow(0, 0, dx, dy, head_width=0.5, head_length=0.3,
                 fc=color, ec=color, linewidth=2, alpha=0.8, label=label)

        # 添加标签
        ax8.text(dx*1.1, dy*1.1, f'{length:.1f}N', fontsize=9,
                color=color, fontweight='bold')

    # 绘制合力向量
    total_dx = sum(l * np.cos(np.deg2rad(a)) / 10 for a, l in zip(angles, lengths))
    total_dy = sum(l * np.sin(np.deg2rad(a)) / 10 for a, l in zip(angles, lengths))

    ax8.arrow(0, 0, total_dx, total_dy, head_width=0.5, head_length=0.3,
             fc='black', ec='black', linewidth=3, alpha=0.9,
             label='合力')

    ax8.set_xlim(-2, 2)
    ax8.set_ylim(-2, 2)
    ax8.set_xlabel('X方向', fontsize=10)
    ax8.set_ylabel('Y方向', fontsize=10)
    ax8.set_title('力向量合成', fontsize=12, fontweight='bold')
    ax8.legend(loc='upper right', fontsize=8)
    ax8.grid(True, alpha=0.3)
    ax8.set_aspect('equal')

    # 9. 抓取稳定性指标
    ax9 = plt.subplot(3, 3, 9)

    # 计算稳定性指标（模拟）
    stability_metrics = {
        '力平衡度': 0.85,
        '同步误差': 0.12,
        '最大过冲': 0.18,
        '响应时间': 0.08,
        '稳态误差': 0.05
    }

    metrics_names = list(stability_metrics.keys())
    metrics_values = list(stability_metrics.values())

    bars = ax9.barh(metrics_names, metrics_values, color=plt.cm.viridis(
        np.linspace(0.2, 0.8, len(metrics_names))))

    # 添加数值标签
    for bar, value in zip(bars, metrics_values):
        width = bar.get_width()
        ax9.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.2f}', va='center', fontsize=9)

    ax9.set_xlabel('指标值', fontsize=10)
    ax9.set_title('抓取稳定性指标', fontsize=12, fontweight='bold')
    ax9.set_xlim(0, 1)
    ax9.grid(True, alpha=0.3, axis='x')

    plt.suptitle('三指抓取器力分析与可视化', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('grasp_force_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='#F5F7FA')
    plt.show()

    # 生成总结统计
    generate_summary_statistics(finger1_force, finger2_force, finger3_force, total_force)

def generate_summary_statistics(f1, f2, f3, total):
    """生成统计摘要"""
    print("\n" + "="*60)
    print("抓取力统计摘要")
    print("="*60)

    stats_data = {
        '手指1': f1,
        '手指2': f2,
        '手指3': f3,
        '总力': total
    }

    for name, data in stats_data.items():
        print(f"\n{name}:")
        print(f"  均值: {np.mean(data):.2f} N")
        print(f"  标准差: {np.std(data):.2f} N")
        print(f"  最小值: {np.min(data):.2f} N")
        print(f"  最大值: {np.max(data):.2f} N")
        print(f"  中位数: {np.median(data):.2f} N")
        print(f"  变异系数: {np.std(data)/np.mean(data)*100:.1f}%")

    # 计算力平衡度
    mean_forces = [np.mean(f1), np.mean(f2), np.mean(f3)]
    balance_score = 1 - np.std(mean_forces) / np.mean(mean_forces)
    print(f"\n力平衡度: {balance_score:.3f} (1为完美平衡)")
    print("="*60)

if __name__ == "__main__":
    grasp_force_analysis()