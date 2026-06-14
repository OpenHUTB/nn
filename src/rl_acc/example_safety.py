"""
安全约束模块使用示例
演示如何将安全约束集成到ACC系统
"""

import numpy as np
import config
from safety import SafetyConstraintsManager, SafetyConstraints
from safety.aeb import AEBController
from safety.monitor import SafetyMonitor


def example_basic_safety_check():
    """基本安全检查示例"""
    print("=" * 60)
    print("安全约束模块 - 基本使用示例")
    print("=" * 60)

    # 初始化安全约束管理器
    safety_mgr = SafetyConstraintsManager()

    # 模拟ACC场景
    ego_speed = 25.0  # 90 km/h
    target_speed = 20.0  # 72 km/h
    distance = 30.0  # 与前车距离30m

    print(f"\n场景:")
    print(f"  自车速度: {ego_speed:.1f} m/s ({ego_speed * 3.6:.1f} km/h)")
    print(f"  前车速度: {target_speed:.1f} m/s ({target_speed * 3.6:.1f} km/h)")
    print(f"  跟车距离: {distance:.1f} m")

    # 综合安全检查
    status = safety_mgr.comprehensive_check(
        distance=distance,
        ego_speed=ego_speed,
        target_speed=target_speed,
        acceleration=0.0
    )

    print(f"\n安全检查结果:")
    print(f"  安全等级: {status.level.value.upper()}")
    print(f"  距离安全: {'✓' if status.distance_safe else '✗'}")
    print(f"  速度安全: {'✓' if status.speed_safe else '✗'}")
    print(f"  加速安全: {'✓' if status.acceleration_safe else '✗'}")
    print(f"  TTC安全: {'✓' if status.ttc_safe else '✗'}")
    print(f"  消息: {status.message}")
    print(f"  建议动作: {status.recommended_action}")

    # 计算安全加速度
    safe_accel = safety_mgr.calculate_safe_acceleration(
        distance, ego_speed, target_speed
    )
    print(f"\n建议的加速度: {safe_accel:.2f} m/s²")

    # 计算安全距离
    safe_dist = safety_mgr.get_safe_distance(ego_speed, target_speed)
    print(f"计算的安全距离: {safe_dist:.2f} m")


def example_aeb_integration():
    """AEB集成示例"""
    print("\n" + "=" * 60)
    print("AEB（自动紧急制动）集成示例")
    print("=" * 60)

    # 初始化AEB控制器
    aeb = AEBController()

    # 模拟危险场景
    ego_speed = 30.0  # 108 km/h
    target_speed = 20.0  # 72 km/h
    distance = 20.0  # 20m距离

    relative_speed = ego_speed - target_speed

    print(f"\n场景:")
    print(f"  自车速度: {ego_speed:.1f} m/s ({ego_speed * 3.6:.1f} km/h)")
    print(f"  前车速度: {target_speed:.1f} m/s ({target_speed * 3.6:.1f} km/h)")
    print(f"  相对速度: {relative_speed:.1f} m/s")
    print(f"  跟车距离: {distance:.1f} m")

    # 检查AEB是否应该激活
    activate, state, reason = aeb.should_activate(
        distance, relative_speed, ego_speed
    )

    print(f"\nAEB状态检查:")
    print(f"  是否激活: {'是' if activate else '否'}")
    print(f"  AEB状态: {state.value}")
    print(f"  原因: {reason}")

    if activate:
        braking_level, decel = aeb.calculate_braking_level(distance, relative_speed)
        print(f"  制动等级: {braking_level.value}")
        print(f"  建议减速度: {decel:.2f} m/s²")


def example_safety_monitor():
    """安全监控示例"""
    print("\n" + "=" * 60)
    print("安全监控示例")
    print("=" * 60)

    # 初始化安全监控器
    monitor = SafetyMonitor()

    # 模拟一系列事件
    print("\n模拟事件记录...")

    events = [
        {'dist': 40.0, 'ego': 25.0, 'target': 22.0, 'TTC': 10.0},
        {'dist': 30.0, 'ego': 26.0, 'target': 22.0, 'TTC': 7.5},
        {'dist': 20.0, 'ego': 27.0, 'target': 22.0, 'TTC': 5.0},
        {'dist': 15.0, 'ego': 28.0, 'target': 20.0, 'TTC': 3.75},
        {'dist': 10.0, 'ego': 28.0, 'target': 18.0, 'TTC': 2.0},
        {'dist': 8.0, 'ego': 28.0, 'target': 18.0, 'TTC': 1.6},
        {'dist': 5.0, 'ego': 30.0, 'target': 15.0, 'TTC': 0.5},  # 紧急
    ]

    for i, event in enumerate(events, 1):
        # 确定事件类型
        if event['TTC'] < 1.0:
            event_type = 'emergency'
        elif event['TTC'] < 3.0:
            event_type = 'danger'
        elif event['TTC'] < 5.0:
            event_type = 'warning'
        elif event['dist'] < 15.0:
            event_type = 'distance_violation'
        else:
            event_type = 'normal'

        monitor.record_event(
            event_type,
            event['dist'],
            event['ego'],
            event['target'],
            event['TTC'],
            f"Event {i}: TTC={event['TTC']:.2f}s"
        )

        print(f"  事件{i}: 距离={event['dist']}m, TTC={event['TTC']:.2f}s -> {event_type}")

    # 打印统计摘要
    monitor.print_summary()


def example_custom_constraints():
    """自定义约束示例"""
    print("\n" + "=" * 60)
    print("自定义约束参数示例")
    print("=" * 60)

    # 创建自定义约束
    custom_constraints = SafetyConstraints(
        min_safe_distance=15.0,      # 更大的安全距离
        warning_distance=25.0,
        caution_distance=40.0,
        max_speed=30.0,              # 更低的最大速度
        max_acceleration=1.5,         # 更舒适的加速度
        max_deceleration=-4.0,
        ttc_danger_threshold=4.0,     # 更严格的TTC阈值
        ttc_warning_threshold=6.0,
        driver_reaction_time=2.0
    )

    safety_mgr = SafetyConstraintsManager(custom_constraints)

    print("\n自定义约束参数:")
    print(f"  最小安全距离: {custom_constraints.min_safe_distance}m")
    print(f"  最大速度: {custom_constraints.max_speed}m/s")
    print(f"  TTC危险阈值: {custom_constraints.ttc_danger_threshold}s")

    # 测试
    ego_speed = 28.0
    target_speed = 22.0
    distance = 20.0

    status = safety_mgr.comprehensive_check(distance, ego_speed, target_speed, 0.0)

    print(f"\n测试结果 (距离={distance}m, 自车={ego_speed}m/s):")
    print(f"  安全等级: {status.level.value}")
    print(f"  消息: {status.message}")


def example_acc_integration():
    """ACC集成示例"""
    print("\n" + "=" * 60)
    print("ACC系统安全集成示例")
    print("=" * 60)

    # 初始化所有组件
    safety_mgr = SafetyConstraintsManager()
    aeb = AEBController()
    monitor = SafetyMonitor()

    # 模拟ACC控制循环
    print("\n模拟ACC控制循环:")

    # 初始状态
    ego_speed = 25.0
    target_speed = 22.0
    distance = 50.0
    current_accel = 0.0

    steps = [
        {'step': 1, 'dist': 50.0, 'target': 22.0},
        {'step': 2, 'dist': 45.0, 'target': 22.0},
        {'step': 3, 'dist': 38.0, 'target': 23.0},
        {'step': 4, 'dist': 30.0, 'target': 23.0},
        {'step': 5, 'dist': 22.0, 'target': 24.0},
        {'step': 6, 'dist': 15.0, 'target': 24.0},
        {'step': 7, 'dist': 10.0, 'target': 24.0},
        {'step': 8, 'dist': 8.0, 'target': 25.0},
    ]

    for step_info in steps:
        distance = step_info['dist']
        target_speed = step_info['target']
        relative_speed = ego_speed - target_speed

        # 1. 检查AEB
        aeb_activate, aeb_state, _ = aeb.should_activate(
            distance, relative_speed, ego_speed
        )

        # 2. 安全约束检查
        status = safety_mgr.comprehensive_check(
            distance, ego_speed, target_speed, current_accel
        )

        # 3. 计算安全加速度
        safe_accel = safety_mgr.calculate_safe_acceleration(
            distance, ego_speed, target_speed
        )

        # 4. 如果AEB激活，使用AEB的制动
        if aeb_activate:
            _, aeb_decel = aeb.calculate_braking_level(distance, relative_speed)
            current_accel = aeb_decel
            action = f"AEB: {aeb_state.value}"
        else:
            # 使用约束后的加速度
            current_accel = safety_mgr.constrain_action(
                safe_accel, distance, ego_speed, target_speed
            )
            action = status.recommended_action

        # 5. 更新状态（简化模型）
        ego_speed = max(0, ego_speed + current_accel * 0.1)

        # 6. 记录事件
        TTC = distance / relative_speed if relative_speed > 0 else float('inf')
        if status.level.value != 'safe':
            monitor.record_event(
                status.level.value,
                distance, ego_speed, target_speed, TTC,
                f"Step {step_info['step']}: {status.message}"
            )

        # 打印
        print(f"  Step {step_info['step']}: 距离={distance:.0f}m, "
              f"TTC={TTC:.1f}s, 速度={ego_speed:.1f}m/s, "
              f"加速度={current_accel:.2f}m/s², "
              f"动作={action}")

    # 打印最终统计
    monitor.print_summary()


if __name__ == "__main__":
    # 运行所有示例
    example_basic_safety_check()
    example_aeb_integration()
    example_safety_monitor()
    example_custom_constraints()
    example_acc_integration()

    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)
