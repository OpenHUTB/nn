import carla
import math
import time
import random

def main():
    # 连接 CARLA 服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    map = world.get_map()

    # 获取蓝图库
    blueprint_lib = world.get_blueprint_library()
    vehicle_bp = blueprint_lib.find('vehicle.tesla.model3')

    # 生成主车（自车），放在直道方便测试车道线
    spawn_points = map.get_spawn_points()
    ego_spawn = spawn_points[0]
    ego_vehicle = world.spawn_actor(vehicle_bp, ego_spawn)
    print("✅ 主车生成成功，启动车道线检测+LDW车道偏离预警+LKA车道居中辅助演示")

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 车道相关参数
    lane_center_offset = 0.0  # 车道中心偏移量
    lane_width = 3.5          # 标准车道宽度（米）
    lane_deviation_threshold = 0.6  # 偏离预警阈值（米）
    lka_correction_strength = 0.15 # 车道居中修正强度

    try:
        # 基础巡航速度
        target_speed = 20.0  # km/h

        for _ in range(1500):
            world.tick()

            # 获取自车状态
            ego_transform = ego_vehicle.get_transform()
            ego_velocity = ego_vehicle.get_velocity()
            ego_speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)

            # 获取当前道路信息
            waypoint = map.get_waypoint(ego_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if waypoint:
                # 计算车辆相对车道中心的偏移
                lane_center = waypoint.transform.location
                vehicle_pos = ego_transform.location
                # 计算车辆在道路横向上的偏移量（简化版）
                dx = vehicle_pos.x - lane_center.x
                dy = vehicle_pos.y - lane_center.y
                lane_offset = math.sqrt(dx**2 + dy**2)

                # 修正偏移方向（基于车辆朝向）
                vehicle_yaw = math.radians(ego_transform.rotation.yaw)
                lane_yaw = math.radians(waypoint.transform.rotation.yaw)
                # 计算车辆相对车道的横向偏移
                cross = dx * math.sin(vehicle_yaw) - dy * math.cos(vehicle_yaw)
                lane_offset = abs(cross)
                lane_offset_direction = cross

            else:
                lane_offset = 0.0
                lane_offset_direction = 0.0

            # 车辆控制逻辑
            control = carla.VehicleControl()

            # 基础巡航控制
            if ego_speed < target_speed:
                control.throttle = 0.4
            else:
                control.throttle = 0.0
            control.brake = 0.0

            # 车道偏离预警 + 车道居中辅助
            if lane_offset > lane_deviation_threshold:
                # 触发车道偏离预警（LDW）
                print(f"⚠️  【车道偏离预警】偏离车道中心：{lane_offset:.2f}m")

                # 车道居中辅助（LKA），自动回正方向盘
                # 根据偏移方向修正转向
                if lane_offset_direction > 0:
                    # 向右偏，向左打方向
                    control.steer = -lka_correction_strength
                else:
                    # 向左偏，向右打方向
                    control.steer = lka_correction_strength
            else:
                # 车道居中，保持方向盘回正
                control.steer = 0.0
                print(f"🚗 【正常行驶】速度：{ego_speed:.1f}km/h | 车道偏移：{lane_offset:.2f}m")

            ego_vehicle.apply_control(control)
            time.sleep(0.05)

    finally:
        # 恢复设置并释放资源
        settings.synchronous_mode = False
        world.apply_settings(settings)
        ego_vehicle.destroy()
        print("\n✅ 所有车辆资源已安全释放")

if __name__ == '__main__':
    main()