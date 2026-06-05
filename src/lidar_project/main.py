import carla
import random
import time
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# -------------------------- 辅助函数 --------------------------
def set_random_weather(world):
    weathers = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon
    ]
    world.set_weather(random.choice(weathers))
    print("🌤️ 已设置随机天气")

# -------------------------- 主程序（只生成视频版） --------------------------
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    tm = client.get_trafficmanager(8000)
    tm.set_global_distance_to_leading_vehicle(0.5)
    tm.set_random_device_seed(42)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_list = []
    camera = None
    video_writer = None

    video_filename = "congestion_video.mp4"
    frame_width = 1280
    frame_height = 720
    fps = 15

    set_random_weather(world)

    try:
        print("🚗 正在生成交通拥堵场景（高密度+低速）...")
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        count = 0
        for spawn_point in spawn_points:
            if count >= 20:
                break
            bp = random.choice(blueprint_library.filter('vehicle.*'))
            try:
                vehicle = world.spawn_actor(bp, spawn_point)
                vehicle_list.append(vehicle)
                vehicle.set_autopilot(True)

                tm.ignore_lights_percentage(vehicle, 100)
                tm.vehicle_percentage_speed_difference(vehicle, -90)
                tm.distance_to_leading_vehicle(vehicle, 0.5)
                tm.set_desired_speed(vehicle, 5)

                count += 1
            except:
                continue

        print("⏳ 让车流稳定下来...")
        for _ in range(200):
            world.tick()

        if vehicle_list:
            cam_bp = blueprint_library.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', str(frame_width))
            cam_bp.set_attribute('image_size_y', str(frame_height))
            cam_bp.set_attribute('fov', '110')

            cam_tf = carla.Transform(carla.Location(x=-12, y=0, z=10), carla.Rotation(pitch=-50))
            camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle_list[0], attachment_type=carla.AttachmentType.SpringArm)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

            # 只写入视频，不保存截图
            def write_video_frame(img):
                array = np.frombuffer(img.raw_data, dtype=np.uint8)
                array = array.reshape((img.height, img.width, 4))
                frame = array[:, :, :3]
                frame = frame[:, :, ::-1]
                video_writer.write(frame)

            camera.listen(write_video_frame)
            print("📸 相机已启动 → 开始录制堵车视频")

        print("📹 开始录制仿真日志...")
        log_file = "block_log.rec"
        client.start_recorder(log_file)

        for _ in range(1200):
            world.tick()

        client.stop_recorder()
        print(f"✅ 日志已保存：{log_file}")

        # ======================
        # 采集多特征数据
        # ======================
        print("\n==== 📊 采集多特征数据 ====")
        data = []
        speeds = []
        accels = []

        for veh in vehicle_list:
            vel = veh.get_velocity()
            speed = np.sqrt(vel.x**2 + vel.y**2)
            accel = veh.get_acceleration()
            acc = np.sqrt(accel.x**2 + accel.y**2)
            dist_to_front = random.uniform(0.5, 2.0)
            lane_offset = random.uniform(-1.2, 1.2)
            yaw = veh.get_transform().rotation.yaw

            speeds.append(speed)
            accels.append(acc)
            data.append([speed, acc, dist_to_front, lane_offset, yaw])

        labels = [1 if s < 1.5 else 0 for s in speeds]

        df = pd.DataFrame(data, columns=["speed", "acceleration", "dist_to_front", "lane_offset", "yaw"])
        df["label"] = labels
        df.to_csv("congestion_data.csv", index=False)
        print("✅ 数据集已保存：congestion_data.csv")

        print("\n==== 🤖 机器学习模型训练 ====")
        X = np.array(data)
        y = np.array(labels)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logreg = LogisticRegression(max_iter=300)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        logreg.fit(X_scaled, y)
        rf.fit(X_scaled, y)

        pred_logreg = logreg.predict(X_scaled)
        pred_rf = rf.predict(X_scaled)

        acc_logreg = accuracy_score(y, pred_logreg)
        acc_rf = accuracy_score(y, pred_rf)

        for idx, veh in enumerate(vehicle_list):
            speed = speeds[idx]
            res_log = pred_logreg[idx]
            res_rf = pred_rf[idx]
            print(f"🚗 车辆{idx} | 速度={speed:.2f} | 逻辑回归={'拥堵' if res_log else '正常'} | 随机森林={'拥堵' if res_rf else '正常'}")

        print("\n==== 📈 模型评估 ====")
        print(f"逻辑回归准确率: {acc_logreg:.2%}")
        print(f"随机森林准确率: {acc_rf:.2%}")
        print(f"拥堵车辆数: {sum(pred_rf)} / {len(vehicle_list)}")
        print(f"平均速度: {np.mean(speeds):.2f} m/s")

        print("\n混淆矩阵:")
        print(confusion_matrix(y, pred_rf))
        print("\n分类报告:")
        print(classification_report(y, pred_rf))

        with open("congestion_rf_model.pkl", "wb") as f:
            pickle.dump(rf, f)
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        print("\n🎉 全部功能执行完毕！")
        print(f"🎬 视频已生成：congestion_video.mp4（约8秒，无截图）")

    finally:
        if camera and camera.is_alive:
            camera.destroy()
        if video_writer:
            video_writer.release()
        for v in vehicle_list:
            if v.is_alive:
                v.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()