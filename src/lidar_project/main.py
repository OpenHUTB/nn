import carla
import random
import time
import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    blueprint_library = world.get_blueprint_library()
    vehicle_list = []

    try:
        print("正在生成交通拥堵场景...")
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        count = 0
        for spawn_point in spawn_points:
            if count >= 10:
                break

            bp = random.choice(blueprint_library.filter('vehicle.*'))
            try:
                vehicle = world.spawn_actor(bp, spawn_point)
                vehicle_list.append(vehicle)
                vehicle.set_autopilot(True)
                count += 1
            except:
                continue

        for _ in range(50):
            world.tick()
            time.sleep(0.05)

        print("开始录制日志...")
        log_file = "block_log.rec"
        client.start_recorder(log_file)

        for _ in range(200):
            world.tick()
            time.sleep(0.05)

        client.stop_recorder()
        print(f"日志已保存：{log_file}")

        # 机器学习：拥堵检测
        print("\n==== 机器学习模型检测拥堵车辆 ====")
        
        data = []
        labels = []

        for veh in vehicle_list:
            vel = veh.get_velocity()
            speed = np.sqrt(vel.x**2 + vel.y**2)
            data.append([speed])
            labels.append(1 if speed < 0.2 else 0)

        # 训练模型
        model = LogisticRegression()
        model.fit(data, labels)

        # 模型推理
        for idx, veh in enumerate(vehicle_list):
            vel = veh.get_velocity()
            speed = np.sqrt(vel.x**2 + vel.y**2)
            result = model.predict([[speed]])[0]

            if result == 1:
                print(f"🚗 车辆 {idx}：【拥堵中】 速度 = {speed:.2f}")
            else:
                print(f"🚗 车辆 {idx}：正常行驶 速度 = {speed:.2f}")

        print("\n✅ 机器学习拥堵检测完成！")

    finally:
        for v in vehicle_list:
            if v.is_alive:
                v.destroy()

        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == '__main__':
    main()