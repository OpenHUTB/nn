import carla
import random
import time
import os
import json
import numpy as np
import pandas as pd
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import datetime

# ======================
# 深度强化学习智能体（Q-Learning）
# ======================
class TrafficDQNAgent:
    def __init__(self):
        self.q_table = {}
        self.lr = 0.1
        self.gamma = 0.9
        self.actions = [0, 1, 2]

    def get_state(self, speed, acc, distance):
        return (round(speed, 1), round(acc, 1), round(distance, 1))

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
        
        old_q = self.q_table[state][action]
        target_q = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] = old_q + self.lr * (target_q - old_q)

# -------------------------- 辅助函数 --------------------------
def set_random_weather(world):
    weathers = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon
    ]
    world.set_weather(random.choice(weathers))
    print("🌤️ 已设置随机天气")

def clean_old_files():
    files_to_clean = ["congestion_video.mp4"]
    for f in files_to_clean:
        if os.path.exists(f):
            os.remove(f)

def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -------------------------- 主程序 --------------------------
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    world = client.get_world()
    tm = client.get_trafficmanager(8000)
    tm.set_global_distance_to_leading_vehicle(0.4)
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
    clean_old_files()
    set_random_weather(world)

    rl_agent = TrafficDQNAgent()
    total_reward = 0

    try:
        print("🚗 正在生成高密度交通拥堵场景...")
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        count = 0
        for spawn_point in spawn_points:
            if count >= 25:
                break
            bp = random.choice(blueprint_library.filter('vehicle.*'))
            try:
                vehicle = world.spawn_actor(bp, spawn_point)
                vehicle_list.append(vehicle)
                vehicle.set_autopilot(True)

                tm.ignore_lights_percentage(vehicle, 100)
                tm.vehicle_percentage_speed_difference(vehicle, -92)
                tm.distance_to_leading_vehicle(vehicle, 0.4)
                tm.set_desired_speed(vehicle, 4)

                count += 1
            except:
                continue

        print("⏳ 等待车流稳定拥堵...")
        for _ in range(250):
            world.tick()

        if vehicle_list:
            cam_bp = blueprint_library.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', str(frame_width))
            cam_bp.set_attribute('image_size_y', str(frame_height))
            cam_bp.set_attribute('fov', '115')

            cam_tf = carla.Transform(carla.Location(x=-14, y=0, z=11), carla.Rotation(pitch=-55))
            camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle_list[0], attachment_type=carla.AttachmentType.SpringArm)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

            def write_video_frame(img):
                array = np.frombuffer(img.raw_data, dtype=np.uint8)
                array = array.reshape((img.height, img.width, 4))
                frame = array[:, :, :3]
                frame = frame[:, :, ::-1]
                video_writer.write(frame)

            camera.listen(write_video_frame)
            print("📸 相机已启动 → 录制堵车视频")

        print("📹 开始录制仿真 + 深度强化学习控制...")
        log_file = "block_log.rec"
        client.start_recorder(log_file)

        congestion_frames = {i:0 for i in range(len(vehicle_list))}
        stop_and_go = {i:0 for i in range(len(vehicle_list))}

        for frame_idx in range(1400):
            world.tick()
            for idx, veh in enumerate(vehicle_list):
                if veh.is_alive:
                    speed = np.sqrt(veh.get_velocity().x**2 + veh.get_velocity().y**2)
                    acc = np.sqrt(veh.get_acceleration().x**2 + veh.get_acceleration().y**2)
                    dist = random.uniform(0.4, 1.8)
                    state = rl_agent.get_state(speed, acc, dist)

                    action = rl_agent.choose_action(state)

                    if action == 0:
                        tm.set_desired_speed(veh, 2.0)
                    elif action == 1:
                        tm.set_desired_speed(veh, 5.0)
                    else:
                        tm.set_desired_speed(veh, 8.0)

                    if 0.5 < speed < 2.0:
                        reward = 5
                    elif speed < 0.2:
                        reward = -5
                    else:
                        reward = 1
                    total_reward += reward

                    next_speed = np.sqrt(veh.get_velocity().x**2 + veh.get_velocity().y**2)
                    next_state = rl_agent.get_state(next_speed, acc, dist)
                    rl_agent.learn(state, action, reward, next_state)

                    if speed < 1.2:
                        congestion_frames[idx] += 1
                    if speed < 0.2:
                        stop_and_go[idx] += 1

        client.stop_recorder()
        print(f"✅ 日志已保存")
        print(f"🧠 深度强化学习累计奖励：{total_reward:.2f}")

        # ======================
        # 原有功能全部保留
        # ======================
        print("\n==== 📊 多维度特征采集 ====")
        data = []
        speeds = []
        accels = []
        congestion_times = []
        stop_times = []

        for idx, veh in enumerate(vehicle_list):
            vel = veh.get_velocity()
            speed = np.sqrt(vel.x**2 + vel.y**2)
            accel = veh.get_acceleration()
            acc = np.sqrt(accel.x**2 + accel.y**2)
            dist_to_front = random.uniform(0.4, 1.8)
            lane_offset = random.uniform(-1.0, 1.0)
            yaw = veh.get_transform().rotation.yaw
            congestion_time = round(congestion_frames[idx] * 0.05, 2)
            stop_time = round(stop_and_go[idx] * 0.05, 2)

            speeds.append(speed)
            accels.append(acc)
            congestion_times.append(congestion_time)
            stop_times.append(stop_time)
            data.append([speed, acc, dist_to_front, lane_offset, yaw, congestion_time, stop_time])

        labels = [1 if s < 1.2 else 0 for s in speeds]
        congestion_levels = ["严重拥堵" if s < 0.4 else "中度拥堵" if s < 1.5 else "轻微拥堵" if s < 3 else "畅通" for s in speeds]

        df = pd.DataFrame(data, columns=[
            "speed", "acceleration", "dist_to_front", 
            "lane_offset", "yaw", "congestion_time", "stop_time"
        ])
        df["label"] = labels
        df["congestion_level"] = congestion_levels
        df.to_csv("congestion_data.csv", index=False)
        print("✅ 数据集已保存：congestion_data.csv")

        print("\n==== 🤖 机器学习模型训练 ====")
        X = np.array(data)
        y = np.array(labels)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logreg = LogisticRegression(max_iter=500)
        rf = RandomForestClassifier(n_estimators=150, random_state=42)

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
            level = congestion_levels[idx]
            print(f"🚗 车辆{idx} | 速度={speed:.2f} | {level} | 拥堵={res_rf}")

        print("\n==== 📈 模型性能评估 ====")
        print(f"逻辑回归准确率: {acc_logreg:.2%}")
        print(f"随机森林准确率: {acc_rf:.2%}")
        print(f"拥堵车辆: {sum(pred_rf)}/{len(vehicle_list)}")
        print(f"平均车速: {np.mean(speeds):.2f} m/s")
        print(f"平均拥堵时长: {np.mean(congestion_times):.2f}s")
        print(f"平均停车时长: {np.mean(stop_times):.2f}s")

        print("\n混淆矩阵:")
        print(confusion_matrix(y, pred_rf))
        print("\n分类报告:")
        print(classification_report(y, pred_rf))

        with open("congestion_rf_model.pkl", "wb") as f:
            pickle.dump(rf, f)
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        with open("drl_agent.pkl", "wb") as f:
            pickle.dump(rl_agent, f)

        result = {
            "create_time": get_current_time(),
            "total_vehicles": len(vehicle_list),
            "congested_count": int(sum(pred_rf)),
            "avg_speed": round(float(np.mean(speeds)), 2),
            "avg_congestion_time": round(float(np.mean(congestion_times)), 2),
            "avg_stop_time": round(float(np.mean(stop_times)), 2),
            "rf_accuracy": round(float(acc_rf), 4),
            "logreg_accuracy": round(float(acc_logreg), 4),
            "drl_total_reward": round(total_reward, 2),
            "congestion_level": "严重拥堵" if sum(pred_rf)/len(vehicle_list) > 0.6 else "中度拥堵"
        }
        with open("congestion_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        with open("congestion_report.txt", "w", encoding="utf-8") as f:
            f.write("==== 交通拥堵检测分析报告 ====\n")
            f.write(f"生成时间：{get_current_time()}\n")
            f.write(f"总车辆数：{len(vehicle_list)}\n")
            f.write(f"拥堵车辆数：{sum(pred_rf)}\n")
            f.write(f"平均车速：{np.mean(speeds):.2f} m/s\n")
            f.write(f"随机森林准确率：{acc_rf:.2%}\n")
            f.write(f"深度强化学习总奖励：{total_reward:.2f}\n")
            f.write(f"拥堵等级：{result['congestion_level']}\n")

        print("\n🎉 全部功能执行完毕！")
        print(f"🎬 视频：{video_filename}")
        print(f"📊 数据集：congestion_data.csv")
        print(f"📄 JSON报告：congestion_result.json")
        print(f"📑 文本报告：congestion_report.txt")
        print(f"🤖 ML模型：congestion_rf_model.pkl")
        print(f"🧠 DRL模型：drl_agent.pkl\n")

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