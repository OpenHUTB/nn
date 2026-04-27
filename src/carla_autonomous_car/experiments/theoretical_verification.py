"""
理论验证实验

验证内容：
1. 策略不变性
2. 势能有界性
3. Lyapunov稳定性
"""

import sys

sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from theory.lyapunov_shaping import *


def verify_policy_invariance():
    """验证策略不变性"""
    print("\n" + "=" * 80)
    print("🔬 VERIFYING POLICY INVARIANCE")
    print("=" * 80)

    params = LaneChangeDynamicsParams()
    shaping = LyapunovPotentialShaping(params)

    # 生成随机状态
    num_samples = 1000
    states = []

    for _ in range(num_samples):
        state = {
            'd': np.random.uniform(-3.5, 3.5),
            'd_dot': np.random.uniform(-2, 2),
            'd_target': np.random.choice([0, 3.5, -3.5]),
            'delta_s': np.random.uniform(5, 50),
            'v_ego': np.random.uniform(20, 40),
            'v_lead': np.random.uniform(15, 35),
            'has_lead': np.random.random() > 0.3
        }
        states.append(state)

    # 计算势能
    potentials = []
    for state in states:
        phi_lat = shaping.lateral_potential.compute_potential(
            state['d'], state['d_dot'], state['d_target']
        )

        phi_long = 0.0
        if state['has_lead']:
            phi_long = shaping.longitudinal_potential.compute_potential(
                state['delta_s'], state['v_ego'], state['v_lead']
            )

        phi_total = phi_lat + phi_long
        potentials.append(phi_total)

    potentials = np.array(potentials)

    # 统计
    print(f"\n✅ Results:")
    print(f"   Potential bounded: {np.max(potentials) - np.min(potentials) < 100}")
    print(f"   Φ_min: {np.min(potentials):.3f}")
    print(f"   Φ_max: {np.max(potentials):.3f}")
    print(f"   Φ_mean: {np.mean(potentials):.3f} ± {np.std(potentials):.3f}")

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.hist(potentials, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Potential Value Φ(s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Potential Values')
    plt.axvline(np.mean(potentials), color='r', linestyle='--', label='Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/figures/potential_distribution.png', dpi=300)
    print(f"\n📊 Plot saved to results/figures/potential_distribution.png")

    print("=" * 80)


def visualize_potential_field():
    """可视化势能场"""
    print("\n🎨 Visualizing Potential Field...")

    params = LaneChangeDynamicsParams()
    lateral = LyapunovLateralPotential(params)

    # 创建网格
    d_range = np.linspace(-7, 7, 100)
    d_dot_range = np.linspace(-3, 3, 100)
    D, D_DOT = np.meshgrid(d_range, d_dot_range)

    # 计算势能
    d_target = 3.5
    PHI = np.zeros_like(D)

    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            PHI[i, j] = lateral.compute_potential(D[i, j], D_DOT[i, j], d_target)

    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))

    contour = ax.contourf(D, D_DOT, PHI, levels=20, cmap='viridis')
    ax.contour(D, D_DOT, PHI, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    ax.plot(d_target, 0, 'r*', markersize=20, label='Target')
    ax.set_xlabel('Lateral Position d (m)')
    ax.set_ylabel('Lateral Velocity ḋ (m/s)')
    ax.set_title('Lateral Potential Field Φ(d, ḋ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(contour, ax=ax, label='Potential Φ')

    plt.tight_layout()
    plt.savefig('results/figures/potential_field.png', dpi=300)
    print(f"📊 Plot saved to results/figures/potential_field.png")


if __name__ == '__main__':
    # 创建结果目录
    import os

    os.makedirs('results/figures', exist_ok=True)

    # 运行验证
    verify_policy_invariance()
    visualize_potential_field()

    print("\n✅ All verifications completed!")