import carla
import random
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

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
    camera = None

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

        # 挂载RGB相机，用于抓拍堵车画面
        if vehicle_list:
            cam_bp = blueprint_library.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', '1280')
            cam_bp.set_attribute('image_size_y', '720')
            cam_bp.set_attribute('fov', '90')
            cam_tf = carla.Transform(carla.Location(x=1.8, z=2.5), carla.Rotation(pitch=-18))
            camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle_list[0])

            # 图片保存回调
            def save_scene_img(img):
                img.save_to_disk(f"traffic_capture_{img.frame}.png")
            camera.listen(save_scene_img)
            print("已挂载抓拍相机，自动保存拥堵画面截图")

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

        # 机器学习拥堵检测
        print("\n==== 机器学习模型检测拥堵车辆 ====")
        data = []
        labels = []
        speeds = []

        for veh in vehicle_list:
            vel = veh.get_velocity()
            speed = np.sqrt(vel.x**2 + vel.y**2)
            x = veh.get_location().x
            y = veh.get_location().y
            
            speeds.append(speed)
            data.append([speed, x, y])
            labels.append(1 if speed < 0.2 else 0)

        # 模型训练
        model = LogisticRegression(max_iter=200)
        model.fit(data, labels)
        pred = model.predict(data)
        acc = accuracy_score(labels, pred)

        # 输出单车检测结果
        for idx, veh in enumerate(vehicle_list):
            vel = veh.get_velocity()
            speed = np.sqrt(vel.x**2 + vel.y**2)
            result = pred[idx]
            if result == 1:
                print(f"🚗 车辆 {idx}：【拥堵中】 速度 = {speed:.2f}")
            else:
                print(f"🚗 车辆 {idx}：正常行驶 速度 = {speed:.2f}")

        # 模型评估统计
        print("\n==== 机器学习模型评估 ====")
        print(f"✅ 模型准确率: {acc:.2%}")
        print(f"🚦 拥堵车辆总数: {sum(pred)} 辆")
        print(f"📊 所有车辆平均速度: {np.mean(speeds):.2f} m/s")

        # 保存训练模型
        with open("congestion_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("✅ 训练完成，模型已保存为：congestion_model.pkl")

        # 展示模型内部参数
        print("\n==== 查看训练好的机器学习模型 ====")
        print("模型名称: 逻辑回归拥堵分类器")
        print("使用算法: Logistic Regression")
        print("输入特征数量:", model.n_features_in_)
        print("模型系数(权重):", np.round(model.coef_, 3))
        print("模型截距:", round(model.intercept_[0], 3))
        print("特征 = 车速 + X坐标 + Y坐标")
        print("标签 = 拥堵(1) / 正常行驶(0)")
        print("\n✅ 机器学习全程完成！")

    finally:
        # 销毁相机
        if camera and camera.is_alive:
            camera.destroy()
        # 销毁车辆
        for v in vehicle_list:
            if v.is_alive:
                v.destroy()
        # 恢复仿真设置
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == '__main__':
    main()