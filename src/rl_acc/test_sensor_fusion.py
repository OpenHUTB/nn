"""
传感器融合模块测试脚本
验证传感器融合功能
"""

import numpy as np
import matplotlib.pyplot as plt
from sensors import SensorFusion, RadarSensor, CameraSensor, LidarSensor


def test_sensor_fusion():
    """测试传感器融合功能"""
    print("=" * 60)
    print("传感器融合模块测试")
    print("=" * 60)

    # 初始化传感器
    radar = RadarSensor()
    camera = CameraSensor()
    lidar = LidarSensor()

    # 初始化融合器
    fusion = SensorFusion(
        radar_weight=0.4,
        camera_weight=0.3,
        lidar_weight=0.3,
        fusion_threshold=0.5
    )

    # 模拟前方目标
    front_objects = [
        {'distance': 50.0, 'speed': 20.0, 'azimuth': 0.0},   # 前方50米，速度20m/s
        {'distance': 80.0, 'speed': 18.0, 'azimuth': 5.0},   # 前方80米，速度18m/s
        {'distance': 30.0, 'speed': 15.0, 'azimuth': -2.0},  # 前方30米，速度15m/s
    ]

    ego_speed = 22.0  # 自车速度 22 m/s

    print("\n自车速度: {} m/s ({:.1f} km/h)".format(ego_speed, ego_speed * 3.6))

    # 执行检测
    radar_detections = radar.detect(ego_speed, front_objects)
    camera_detections = camera.detect(ego_speed, front_objects)
    lidar_detections = lidar.detect(ego_speed, front_objects)

    print("\n雷达检测数量:", len(radar_detections))
    print("摄像头检测数量:", len(camera_detections))
    print("激光雷达检测数量:", len(lidar_detections))

    # 执行融合
    fused_objects = fusion.fuse_sensors(
        radar_detections,
        camera_detections,
        lidar_detections
    )

    print("\n融合后目标数量:", len(fused_objects))
    print("\n融合结果:")
    print("-" * 60)

    for obj in fused_objects:
        print(f"目标 {obj.object_id}:")
        print(f"  距离: {obj.distance:.2f} m")
        print(f"  相对速度: {obj.relative_speed:.2f} m/s")
        print(f"  方位角: {np.degrees(obj.azimuth):.2f}°")
        print(f"  置信度: {obj.confidence:.3f}")
        print(f"  数据来源: {', '.join(obj.data_sources)}")
        print()

    # 计算TTC
    TTC = fusion.get_TTC(ego_speed)
    if TTC is not None:
        print(f"与最近目标的碰撞时间 (TTC): {TTC:.2f} 秒")
    else:
        print("无法计算TTC（目标正在远离或静止）")

    return fused_objects, fusion


def visualize_sensor_fusion(fused_objects, ego_speed):
    """可视化传感器融合结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：俯视图
    ax1 = axes[0]
    ax1.set_xlim(-100, 100)
    ax1.set_ylim(0, 200)
    ax1.set_xlabel('横向距离 (m)')
    ax1.set_ylabel('纵向距离 (m)')
    ax1.set_title('传感器融合俯视图')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 绘制自车
    ax1.plot(0, 0, 'bs', markersize=15, label='自车')

    # 绘制检测目标
    for obj in fused_objects:
        x = obj.distance * np.sin(obj.azimuth)
        y = obj.distance * np.cos(obj.azimuth)
        size = 100 * obj.confidence
        ax1.plot(x, y, 'ro', markersize=size/10)
        ax1.annotate(f'T{obj.object_id}\n{obj.distance:.1f}m',
                    (x, y), textcoords="offset points",
                    xytext=(10, 10), fontsize=9)

    # 右图：距离-置信度图
    ax2 = axes[1]
    distances = [obj.distance for obj in fused_objects]
    confidences = [obj.confidence for obj in fused_objects]
    object_ids = [f'T{obj.object_id}' for obj in fused_objects]

    bars = ax2.bar(object_ids, confidences, color='skyblue')
    ax2.set_xlabel('目标 ID')
    ax2.set_ylabel('置信度')
    ax2.set_title('融合目标置信度')
    ax2.set_ylim(0, 1)

    # 添加距离标签
    for i, (bar, dist) in enumerate(zip(bars, distances)):
        ax2.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f'{dist:.1f}m',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('sensor_fusion_test.png', dpi=150)
    print("\n可视化结果已保存到 sensor_fusion_test.png")
    plt.show()


def compare_sensors():
    """对比不同传感器的检测能力"""
    print("\n" + "=" * 60)
    print("传感器性能对比")
    print("=" * 60)

    radar = RadarSensor()
    camera = CameraSensor()
    lidar = LidarSensor()

    # 测试不同距离的目标
    distances = [20, 50, 80, 100, 150, 200]
    ego_speed = 20.0

    print("\n距离\t\t雷达\t\t摄像头\t\t激光雷达")
    print("-" * 60)

    for dist in distances:
        front_objects = [{'distance': dist, 'speed': 18.0, 'azimuth': 0.0}]

        radar_det = radar.detect(ego_speed, front_objects)
        camera_det = camera.detect(ego_speed, front_objects)
        lidar_det = lidar.detect(ego_speed, front_objects)

        radar_ok = "✓" if len(radar_det) > 0 else "✗"
        camera_ok = "✓" if len(camera_det) > 0 else "✗"
        lidar_ok = "✓" if len(lidar_det) > 0 else "✗"

        print(f"{dist}m\t\t{radar_ok}\t\t{camera_ok}\t\t{lidar_ok}")


if __name__ == "__main__":
    # 运行基本测试
    fused_objects, fusion = test_sensor_fusion()

    # 对比传感器性能
    compare_sensors()

    # 可视化结果
    if fused_objects:
        visualize_sensor_fusion(fused_objects, ego_speed=22.0)
