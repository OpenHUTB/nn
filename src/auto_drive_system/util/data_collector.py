import os
import csv
import time
import random
import shutil
import numpy as np
import carla
import util.camera as cs  # 导入刚才修改的模块
import cv2

# --- 配置 ---
DATA_DIR = './my_driving_dataset'
IMG_DIR = os.path.join(DATA_DIR, 'images')
LABELS_PATH = os.path.join(DATA_DIR, 'labels.csv')
TOTAL_FRAMES = 2000

# --- 0. 预处理：强制删除旧数据集 ---
if os.path.exists(DATA_DIR):
    print(f"🗑️ 正在删除旧的数据集目录: {DATA_DIR}")
    shutil.rmtree(DATA_DIR)
os.makedirs(IMG_DIR, exist_ok=True)
print(f"📂 创建新数据目录: {IMG_DIR}")

sensors = []


def main():
    client = None
    vehicle = None

    try:
        # --- 1. 连接 CARLA ---
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"🌍 连接到地图: {world.get_map().name}")

        # --- 2. 配置同步模式 ---
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)

        # --- 3. 配置交通管理器 ---
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(42)

        blueprint_library = world.get_blueprint_library()

        # --- 4. 生成车辆 ---
        vehicle_bp_list = blueprint_library.filter("vehicle.tesla.model3")
        if not vehicle_bp_list:
            vehicle_bp_list = blueprint_library.filter("vehicle.*")

        v_bp = random.choice(vehicle_bp_list)
        spawn_points = world.get_map().get_spawn_points()

        if not spawn_points:
            print("❌ 地图中没有生成点！")
            return

        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(v_bp, spawn_point)
        print(f"🚗 车辆已生成: {v_bp.id}")

        # --- 5. 强制唤醒车辆 ---
        vehicle.set_target_velocity(carla.Vector3D(x=1.0, y=0, z=0))
        time.sleep(0.1)

        # --- 6. 设置摄像头 ---
        pygame_size = {"image_x": 1152, "image_y": 600}
        try:
            # 实例化 cameraManage 类
            camera_manager = cs.cameraManage(world, vehicle, pygame_size)
            cameras = camera_manager.camaraGenarate()
            print("📷 摄像头已绑定")
        except Exception as e:
            print(f"⚠️ 摄像头绑定失败: {e}")
            return

        # --- 7. 开启自动驾驶 ---
        vehicle.set_autopilot(True)
        tm.set_desired_speed(vehicle, 15.0)
        tm.auto_lane_change(vehicle, True)

        print(">>> 🚀 开始采集数据...")

        frame_count = 0
        max_fail_count = 30
        fail_count = 0

        labels_file = open(LABELS_PATH, 'w', newline='')
        csv_writer = csv.writer(labels_file)
        csv_writer.writerow(['steer', 'throttle', 'brake'])

        try:
            while frame_count < TOTAL_FRAMES:
                try:
                    # --- 核心循环 ---
                    world.tick()
                    time.sleep(0.01)

                    # --- 获取数据 ---
                    sensor_data = camera_manager.get_data()

                    # --- 检查数据完整性 ---
                    if (sensor_data['Front'] is None or
                            sensor_data['Rear'] is None or
                            sensor_data['Left'] is None or
                            sensor_data['Right'] is None):
                        fail_count += 1
                        if fail_count > max_fail_count:
                            print(f"❌ 错误：连续 {max_fail_count} 帧未接收到传感器数据，退出。")
                            break
                        continue

                    fail_count = 0

                    # --- 图像拼接逻辑 ---
                    f = sensor_data['Front']
                    r = sensor_data['Rear']
                    l = sensor_data['Left']
                    ri = sensor_data['Right']

                    # 1. 前后拼接
                    img_front_rear = np.concatenate((f, r), axis=1)
                    # 2. 左右拼接
                    img_left_right = np.concatenate((l, ri), axis=1)
                    # 3. 上下拼接
                    img_combined = np.concatenate((img_front_rear, img_left_right), axis=0)

                    # --- 保存数据 ---
                    img_path = os.path.join(IMG_DIR, f"{frame_count:06d}.jpg")
                    cv2.imwrite(img_path, img_combined)

                    control = vehicle.get_control()
                    csv_writer.writerow([control.steer, control.throttle, control.brake])

                    frame_count += 1

                    if frame_count % 50 == 0:
                        velocity = vehicle.get_velocity()
                        current_speed = np.linalg.norm([velocity.x, velocity.y])
                        print(f"📸 已采集 {frame_count}/{TOTAL_FRAMES} 帧 | 速度: {current_speed:.2f} m/s")

                except Exception as e:
                    print(f"采集循环内部错误: {e}")
                    time.sleep(0.1)
                    continue

        except KeyboardInterrupt:
            print("\n>>> 用户停止采集")

        finally:
            labels_file.close()
            print(f">>> 数据已保存至: {DATA_DIR}")

    except Exception as e:
        print(f"❌ 致命错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # --- 8. 清理资源 ---
        print("正在清理...")
        if client:
            try:
                world = client.get_world()
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
            except:
                pass

        if vehicle:
            vehicle.destroy()

        print("✅ 清理完成")


if __name__ == '__main__':
    main()