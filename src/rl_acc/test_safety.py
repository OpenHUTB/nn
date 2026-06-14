"""
安全约束模块测试脚本
验证安全约束功能
"""

import numpy as np
import matplotlib.pyplot as plt
from safety import SafetyConstraintsManager, SafetyConstraints, SafetyLevel
from safety.aeb import AEBController, AEBState
from safety.monitor import SafetyMonitor


def test_safety_constraints():
    """测试安全约束功能"""
    print("=" * 60)
    print("安全约束模块测试")
    print("=" * 60)

    # 初始化安全约束管理器
    safety_mgr = SafetyConstraintsManager()

    # 测试场景
    test_cases = [
        {
            'name': '安全距离',
            'distance': 50.0,
            'ego_speed': 25.0,
            'target_speed': 20.0,
            'acceleration': 0.0
        },
        {
            'name': '警告距离',
            'distance': 15.0,
            'ego_speed': 25.0,
            'target_speed': 20.0,
            'acceleration': 0.0
        },
        {
            'name': '危险距离',
            'distance': 8.0,
            'ego_speed': 25.0,
            'target_speed': 20.0,
            'acceleration': 0.0
        },
        {
            'name': '紧急情况',
            'distance': 5.0,
            'ego_speed': 30.0,
            'target_speed': 15.0,
            'acceleration': 0.0
        }
    ]

    print("\n测试场景:")
    print("-" * 60)

    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   距离: {case['distance']}m, 自车速度: {case['ego_speed']}m/s, "
              f"前车速度: {case['target_speed']}m/s")

        # 综合检查
        status = safety_mgr.comprehensive_check(
            case['distance'],
            case['ego_speed'],
            case['target_speed'],
            case['acceleration']
        )

        print(f"   安全等级: {status.level.value}")
        print(f"   消息: {status.message}")
        print(f"   建议动作: {status.recommended_action}")

        # 计算安全加速度
        safe_accel = safety_mgr.calculate_safe_acceleration(
            case['distance'],
            case['ego_speed'],
            case['target_speed']
        )
        print(f"   建议加速度: {safe_accel:.2f} m/s²")

    return safety_mgr


def test_aeb():
    """测试AEB功能"""
    print("\n" + "=" * 60)
    print("AEB（自动紧急制动）测试")
    print("=" * 60)

    # 初始化AEB控制器
    aeb = AEBController()

    # 测试场景
    test_cases = [
        {'distance': 50.0, 'relative_speed': 5.0, 'ego_speed': 25.0},
        {'distance': 25.0, 'relative_speed': 8.0, 'ego_speed': 25.0},
        {'distance': 15.0, 'relative_speed': 10.0, 'ego_speed': 25.0},
        {'distance': 10.0, 'relative_speed': 12.0, 'ego_speed': 25.0},
        {'distance': 5.0, 'relative_speed': 15.0, 'ego_speed': 25.0},
    ]

    print("\nAEB激活测试:")
    print("-" * 60)

    for i, case in enumerate(test_cases, 1):
        TTC = case['distance'] / case['relative_speed'] if case['relative_speed'] > 0 else float('inf')

        activate, state, reason = aeb.should_activate(
            case['distance'],
            case['relative_speed'],
            case['ego_speed']
        )

        braking_level, decel = aeb.calculate_braking_level(
            case['distance'],
            case['relative_speed']
        )

        print(f"\n{i}. 距离={case['distance']}m, 相对速度={case['relative_speed']}m/s, TTC={TTC:.2f}s")
        print(f"   激活: {'是' if activate else '否'}, 状态: {state.value}")
        print(f"   制动等级: {braking_level.value}, 减速度: {decel:.2f} m/s²")
        print(f"   原因: {reason}")


def test_safety_monitor():
    """测试安全监控功能"""
    print("\n" + "=" * 60)
    print("安全监控测试")
    print("=" * 60)

    # 初始化安全监控器
    monitor = SafetyMonitor()

    # 模拟记录事件
    events = [
        {'type': 'ttc_warning', 'distance': 25.0, 'ego': 25.0, 'target': 20.0, 'TTC': 4.0},
        {'type': 'distance_violation', 'distance': 12.0, 'ego': 25.0, 'target': 20.0, 'TTC': 2.4},
        {'type': 'emergency', 'distance': 5.0, 'ego': 30.0, 'target': 15.0, 'TTC': 0.5},
        {'type': 'recovery', 'distance': 50.0, 'ego': 22.0, 'target': 22.0, 'TTC': float('inf')},
    ]

    for event in events:
        monitor.record_event(
            event['type'],
            event['distance'],
            event['ego'],
            event['target'],
            event['TTC'],
            f"Test event: {event['type']}"
        )

    # 打印统计摘要
    monitor.print_summary()

    return monitor


def visualize_safety_zones():
    """可视化安全区域"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：安全距离 vs 速度
    ax1 = axes[0]
    speeds = np.linspace(0, 35, 50)
    safe_distances = []

    safety_mgr = SafetyConstraintsManager()

    for speed in speeds:
        safe_dist = safety_mgr.get_safe_distance(speed, 20.0)  # 前车速度20m/s
        safe_distances.append(safe_dist)

    ax1.plot(speeds, safe_distances, 'b-', linewidth=2)
    ax1.axhline(y=safety_mgr.constraints.min_safe_distance, color='r',
                linestyle='--', label='Min Safe Distance')
    ax1.fill_between(speeds, 0, safe_distances, alpha=0.3, label='Safe Zone')
    ax1.fill_between(speeds, safe_distances, 100, alpha=0.1, color='gray')
    ax1.set_xlabel('Ego Speed (m/s)')
    ax1.set_ylabel('Safe Distance (m)')
    ax1.set_title('Safe Distance vs Speed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 35)
    ax1.set_ylim(0, 100)

    # 右图：TTC vs 制动等级
    ax2 = axes[1]
    distances = np.linspace(1, 50, 100)
    relative_speed = 10.0  # 相对速度10m/s

    TTCs = distances / relative_speed

    # 定义制动区域
    emergency_zone = TTCs < 1.0
    danger_zone = (TTCs >= 1.0) & (TTCs < 3.0)
    warning_zone = (TTCs >= 3.0) & (TTCs < 5.0)
    safe_zone = TTCs >= 5.0

    ax2.fill_between(distances[emergency_zone], 0, 1,
                     color='red', alpha=0.5, label='Emergency')
    ax2.fill_between(distances[danger_zone], 0, 1,
                     color='orange', alpha=0.5, label='Danger')
    ax2.fill_between(distances[warning_zone], 0, 1,
                     color='yellow', alpha=0.5, label='Warning')
    ax2.fill_between(distances[safe_zone], 0, 1,
                     color='green', alpha=0.5, label='Safe')

    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Safety Level')
    ax2.set_title('TTC Safety Zones')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('safety_zones.png', dpi=150)
    print("\n可视化结果已保存到 safety_zones.png")
    plt.show()


def test_constrain_action():
    """测试动作约束"""
    print("\n" + "=" * 60)
    print("动作约束测试")
    print("=" * 60)

    safety_mgr = SafetyConstraintsManager()

    test_cases = [
        {'desired': 2.0, 'distance': 50.0, 'ego': 25.0, 'target': 20.0},
        {'desired': 0.5, 'distance': 20.0, 'ego': 25.0, 'target': 20.0},
        {'desired': 0.0, 'distance': 10.0, 'ego': 30.0, 'target': 20.0},
        {'desired': -1.0, 'distance': 5.0, 'ego': 30.0, 'target': 15.0},
    ]

    print("\n约束前后加速度对比:")
    print("-" * 60)

    for case in test_cases:
        constrained = safety_mgr.constrain_action(
            case['desired'],
            case['distance'],
            case['ego'],
            case['target']
        )

        print(f"\n距离={case['distance']}m, 自车={case['ego']}m/s, 前车={case['target']}m/s")
        print(f"  期望加速度: {case['desired']:.2f} m/s²")
        print(f"  约束后加速度: {constrained:.2f} m/s²")


if __name__ == "__main__":
    # 运行所有测试
    test_safety_constraints()
    test_aeb()
    test_safety_monitor()
    test_constrain_action()

    # 可视化
    visualize_safety_zones()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
