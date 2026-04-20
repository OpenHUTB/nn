import os
import csv
import time
import random
import shutil
import numpy as np
import carla
import util.camera as cs
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
sensor_data = {'Front': None, 'Rear': None, 'Left': None, 'Right': None}


def pygame_callback(image, side):
    try:
        # 图像格式转换
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        img = img[:, :, :3]  # 去除 Alpha
        img = img[:, :, ::-1]  # RGB -> BGR (OpenCV 格式)
        sensor_data[side] = img
    except Exception as e:
        print(f"回调错误: {e}")


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

        # --- 3. 配置交通管理器 (TM) ---
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

        # --- 5. 强制唤醒车辆 (关键修复) ---
        vehicle.set_target_velocity(carla.Vector3D(x=1.0, y=0, z=0))
        time.sleep(0.1)  # 短暂等待物理引擎响应

        # --- 6. 设置摄像头 ---
        pygame_size = {"image_x": 1152, "image_y": 600}
        try:
            cameras = cs.cameraManage(world, vehicle, pygame_size).camaraGenarate()
            for cam_name, cam_actor in cameras.items():
                cam_actor.listen(lambda image, name=cam_name: pygame_callback(image, name))
                sensors.append(cam_actor)
            print("📷 摄像头已绑定")
        except Exception as e:
            print(f"⚠️ 摄像头绑定失败: {e}")

        # --- 7. 开启自动驾驶并提速 ---
        vehicle.set_autopilot(True)

        # 关键修复：直接设置目标速度
        tm.set_desired_speed(vehicle, 15.0)
        tm.auto_lane_change(vehicle, True)  # 允许变道

        print(">>> 🚀 开始采集数据...")

        frame_count = 0
        max_fail_count = 30  # 如果连续30帧没数据，强制退出
        fail_count = 0

        labels_file = open(LABELS_PATH, 'w', newline='')
        csv_writer = csv.writer(labels_file)
        csv_writer.writerow(['steer', 'throttle', 'brake'])

        try:
            while frame_count < TOTAL_FRAMES:
                try:
                    # --- 关键修复：只保留 tick，移除 wait_for_tick ---
                    world.tick()  # 这一行足以推进仿真
                    time.sleep(0.01)  # 给数据传输留一点点缓冲时间

                    # --- 检查传感器数据 ---
                    # 如果数据为空，跳过本次循环，不进行保存
                    if any(v is None for v in sensor_data.values()):
                        fail_count += 1
                        if fail_count > max_fail_count:
                            print(f"❌ 错误：连续 {max_fail_count} 帧未接收到传感器数据，退出。")
                            break
                        continue

                    # 重置失败计数器
                    fail_count = 0

                    # --- 数据处理 ---
                    # 拼接图像
                    img_front = np.concatenate((sensor_data['Front'], sensor_data['Rear']), axis=1)
                    img_rear = np.concatenate((sensor_data['Left'], sensor_data['Right']), axis=1)
                    img_combined = np.concatenate((img_front, img_rear), axis=0)

                    # 保存图像
                    img_path = os.path.join(IMG_DIR, f"{frame_count:06d}.jpg")
                    cv2.imwrite(img_path, img_combined)

                    # 保存标签
                    control = vehicle.get_control()
                    csv_writer.writerow([control.steer, control.throttle, control.brake])

                    frame_count += 1

                    # --- 调试输出 ---
                    if frame_count % 50 == 0:
                        velocity = vehicle.get_velocity()
                        current_speed = np.linalg.norm([velocity.x, velocity.y])
                        print(f"📸 已采集 {frame_count}/{TOTAL_FRAMES} 帧 | 当前速度: {current_speed:.2f} m/s")

                    # 每次处理完数据后，重置 sensor_data，防止下一帧用到旧数据
                    for k in sensor_data:
                        sensor_data[k] = None

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

        for sensor in sensors:
            if sensor.is_alive: sensor.destroy()

        if vehicle: vehicle.destroy()
        print("✅ 清理完成")


if __name__ == '__main__':
    main()