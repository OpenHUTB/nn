import carla
import math
import time

def main():
    # 连接 CARLA 服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    map = world.get_map()

    # 获取蓝图库
    blueprint_lib = world.get_blueprint_library()
    vehicle_bp = blueprint_lib.find('vehicle.tesla.model3')

    # 生成车辆
    spawn_point = map.get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("✅ 车辆生成成功，启动车道保持控制")

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    try:
        for _ in range(800):
            world.tick()

            # 获取车辆当前状态
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)

            # 获取车辆当前车道信息
            waypoint = map.get_waypoint(transform.location)
            if waypoint:
                next_waypoint = waypoint.next(2.0)[0]
                # 计算车辆与车道中心线的偏差
                lane_vector = next_waypoint.transform.location - transform.location
                # 计算横向偏差，简单纠偏逻辑
                lateral_error = lane_vector.y

                # 车道保持控制逻辑（和你之前的固定转向完全不同）
                control = carla.VehicleControl()
                control.throttle = 0.35  # 稳定巡航速度
                # 根据横向偏差自动调整方向盘，保持在车道中心
                control.steer = max(-0.3, min(0.3, -lateral_error * 0.5))
                control.brake = 0.0

                vehicle.apply_control(control)

                print(f"🚗 车道保持中 | 速度: {speed:.1f} km/h | 转向: {control.steer:.2f}")
            else:
                # 丢失车道信息时减速
                vehicle.apply_control(carla.VehicleControl(brake=0.3))
                print("⚠️  车道丢失，正在减速")

            time.sleep(0.05)

    finally:
        # 安全释放资源
        settings.synchronous_mode = False
        world.apply_settings(settings)
        vehicle.destroy()
        print("\n✅ 资源已安全释放")

if __name__ == '__main__':
    main()