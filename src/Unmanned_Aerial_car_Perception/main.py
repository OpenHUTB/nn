import carla


def main():
    # 1. 连接Carla模拟器
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
        print("✅ 成功连接Carla模拟器！")
        print("📌 当前仿真地图：", world.get_map().name)

        # 2. 获取车辆蓝图
        vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")

        # 3. 改用Carla内置的合法生成点（无碰撞）
        spawn_points = world.get_map().get_spawn_points()  # 获取所有合法生成点
        if spawn_points:
            vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])  # 用第一个合法点
            print("🚗 成功生成特斯拉车辆，ID：", vehicle.id)

            # 车辆简单前进
            vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
            print("🚙 车辆已启动前进！")
        else:
            print("⚠️ 未找到合法的车辆生成点")

    except Exception as e:
        print("❌ 调用失败：", e)


if __name__ == "__main__":
    main()