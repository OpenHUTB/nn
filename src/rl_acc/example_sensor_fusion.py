"""
传感器融合模块使用示例
演示如何集成传感器融合到ACC系统
"""

import numpy as np
import config
from sensors import SensorFusion, RadarSensor, CameraSensor, LidarSensor


def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("传感器融合模块 - 基本使用示例")
    print("=" * 60)

    # 初始化传感器
    radar = RadarSensor(
        max_range=config.RADAR_MAX_RANGE,
        field_of_view=config.RADAR_FOV
    )

    camera = CameraSensor(
        detection_range=config.CAMERA_MAX_RANGE,
        field_of_view=config.CAMERA_FOV
    )

    lidar = LidarSensor(
        max_range=config.LIDAR_MAX_RANGE,
        horizontal_fov=config.LIDAR_FOV
    )

    # 初始化融合器
    fusion = SensorFusion(
        radar_weight=config.RADAR_WEIGHT,
        camera_weight=config.CAMERA_WEIGHT,
        lidar_weight=config.LIDAR_WEIGHT,
        fusion_threshold=config.FUSION_THRESHOLD
    )

    # 模拟场景：自车以 25 m/s 行驶，前方有两辆车
    ego_speed = 25.0  # 约 90 km/h

    # 前方目标
    front_objects = [
        {
            'distance': 40.0,
            'speed': 20.0,
            'azimuth': 0.0
        },
        {
            'distance': 80.0,
            'speed': 22.0,
            'azimuth': 3.0
        }
    ]

    print(f"\n自车状态:")
    print(f"  速度: {ego_speed:.1f} m/s ({ego_speed * 3.6:.1f} km/h)")
    print(f"\n前方目标:")
    for i, obj in enumerate(front_objects, 1):
        print(f"  目标 {i}: 距离 {obj['distance']}m, "
              f"速度 {obj['speed']:.1f} m/s ({obj['speed'] * 3.6:.1f} km/h), "
              f"方位 {obj['azimuth']}°")

    # 执行传感器检测
    radar_detections = radar.detect(ego_speed, front_objects)
    camera_detections = camera.detect(ego_speed, front_objects)
    lidar_detections = lidar.detect(ego_speed, front_objects)

    print(f"\n检测结果:")
    print(f"  雷达检测: {len(radar_detections)} 个目标")
    print(f"  摄像头检测: {len(camera_detections)} 个目标")
    print(f"  激光雷达检测: {len(lidar_detections)} 个目标")

    # 传感器融合
    fused_objects = fusion.fuse_sensors(
        radar_detections,
        camera_detections,
        lidar_detections
    )

    print(f"\n融合结果:")
    print(f"  融合后目标数: {len(fused_objects)}")

    for obj in fused_objects:
        print(f"\n  目标 {obj.object_id}:")
        print(f"    距离: {obj.distance:.2f} m")
        print(f"    相对速度: {obj.relative_speed:.2f} m/s")
        print(f"    置信度: {obj.confidence:.3f}")
        print(f"    数据来源: {', '.join(obj.data_sources)}")

        # 计算TTC
        TTC = fusion.get_TTC(ego_speed)
        if TTC:
            print(f"    TTC: {TTC:.2f} 秒")


def example_integration_with_acc():
    """集成到ACC系统的示例"""
    print("\n" + "=" * 60)
    print("传感器融合模块 - ACC集成示例")
    print("=" * 60)

    # 初始化组件
    fusion = SensorFusion(
        radar_weight=0.4,
        camera_weight=0.3,
        lidar_weight=0.3
    )

    # 模拟ACC决策流程
    ego_speed = 30.0  # 108 km/h

    # 模拟前方交通
    front_objects = [
        {'distance': 30.0, 'speed': 25.0, 'azimuth': 0.0},
        {'distance': 60.0, 'speed': 28.0, 'azimuth': -2.0},
        {'distance': 100.0, 'speed': 22.0, 'azimuth': 1.5},
    ]

    # 检测和融合
    radar = RadarSensor()
    camera = CameraSensor()
    lidar = LidarSensor()

    fused_objects = fusion.fuse_sensors(
        radar.detect(ego_speed, front_objects),
        camera.detect(ego_speed, front_objects),
        lidar.detect(ego_speed, front_objects)
    )

    # 获取最近目标用于ACC控制
    closest = fusion.get_closest_object()

    if closest:
        print(f"\nACC控制决策:")
        print(f"  最近目标距离: {closest.distance:.2f} m")
        print(f"  相对速度: {closest.relative_speed:.2f} m/s")

        # TTC计算
        TTC = fusion.get_TTC(ego_speed)
        if TTC:
            print(f"  碰撞时间 (TTC): {TTC:.2f} 秒")

            if TTC < 3.0:
                print(f"  ⚠️ 警告: TTC较低，需要减速！")
            elif TTC < 5.0:
                print(f"  ⚡ 注意: 适当减速")
            else:
                print(f"  ✓ 安全距离充足")

        # 安全距离检查
        safety_distance = config.SAFETY_DISTANCE
        if closest.distance < safety_distance:
            print(f"  🔴 紧急: 距离小于安全距离 ({safety_distance}m)！")
            print(f"  建议: 立即制动")
        elif closest.distance < safety_distance * 1.5:
            print(f"  🟡 警告: 距离接近安全距离")
            print(f"  建议: 减速保持距离")
        else:
            print(f"  🟢 安全: 距离充足")
            print(f"  建议: 保持当前速度或加速")


def example_sensor_comparison():
    """传感器性能对比示例"""
    print("\n" + "=" * 60)
    print("传感器性能对比")
    print("=" * 60)

    radar = RadarSensor()
    camera = CameraSensor()
    lidar = LidarSensor()

    ego_speed = 20.0

    distances = [20, 50, 80, 100, 150, 200]

    print(f"\n{'距离':<8} {'雷达':<10} {'摄像头':<10} {'激光雷达':<10}")
    print("-" * 40)

    for dist in distances:
        objects = [{'distance': dist, 'speed': 18.0, 'azimuth': 0.0}]

        radar_ok = len(radar.detect(ego_speed, objects)) > 0
        camera_ok = len(camera.detect(ego_speed, objects)) > 0
        lidar_ok = len(lidar.detect(ego_speed, objects)) > 0

        radar_status = "✓" if radar_ok else "✗"
        camera_status = "✓" if camera_ok else "✗"
        lidar_status = "✓" if lidar_ok else "✗"

        print(f"{dist}m{'':<4} {radar_status:<10} {camera_status:<10} {lidar_status:<10}")


if __name__ == "__main__":
    # 运行所有示例
    example_basic_usage()
    example_integration_with_acc()
    example_sensor_comparison()

    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)
