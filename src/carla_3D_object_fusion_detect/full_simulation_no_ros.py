import carla
import time
import random
import os
import cv2
import numpy as np
import threading

# ====================== 保存到【上3级目录】 ======================
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
p1 = os.path.dirname(current_dir)
p2 = os.path.dirname(p1)
p3 = os.path.dirname(p2)

image_folder = os.path.join(p3, "images")
lidar_folder = os.path.join(p3, "lidar")
os.makedirs(image_folder, exist_ok=True)
os.makedirs(lidar_folder, exist_ok=True)

# ====================== 5分钟保存一次 ======================
SAVE_INTERVAL = 5 * 60
last_save_time = time.time()
latest_image = None
latest_lidar = None

# 退出控制
stop_thread = False

# ====================== 连接 CARLA ======================
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()

# ====================== 雨天天气 ======================
weather = carla.WeatherParameters(
    cloudiness=90.0,
    precipitation=90.0,
    precipitation_deposits=90.0,
    wind_intensity=20.0,
    wetness=90.0
)
world.set_weather(weather)
print("✅ 雨天天气已设置")

# ====================== 生成车辆 ======================
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
if vehicle is None:
    raise RuntimeError("车辆生成失败！请检查 spawn point 是否有效")
vehicle.set_autopilot(True)
print("✅ 车辆生成并开启自动驾驶!")

# ====================== 挂载相机 ======================
cam_bp = blueprint_library.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '800')
cam_bp.set_attribute('image_size_y', '600')
cam_bp.set_attribute('fov', '110')
camera = world.spawn_actor(
    cam_bp,
    carla.Transform(carla.Location(x=1.5, z=2.4)),
    attach_to=vehicle
)

# ====================== 挂载激光雷达 ======================
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '100')
lidar_bp.set_attribute('points_per_second', '100000')
lidar_bp.set_attribute('rotation_frequency', '10')
lidar = world.spawn_actor(
    lidar_bp,
    carla.Transform(carla.Location(x=0, z=2.5)),
    attach_to=vehicle
)

print("✅ 相机与雷达挂载成功")
print(f"⏱️  每 {SAVE_INTERVAL//60} 分钟自动保存一次")
print("🎥 画面窗口已弹出，按 Ctrl+C 键可安全退出")

# ====================== 窗口显示 + 保存线程 ======================
def show_window():
    global latest_image, latest_lidar, last_save_time
    while not stop_thread:
        if latest_image is not None:
            cv2.imshow("CARLA Camera", latest_image)
        
        # 按 Q 退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # 5分钟自动保存
        current_time = time.time()
        if current_time - last_save_time >= SAVE_INTERVAL:
            if latest_image is not None and latest_lidar is not None:
                ts = str(int(current_time))
                cv2.imwrite(os.path.join(image_folder, f"{ts}.png"), latest_image)
                latest_lidar.save_to_disk(os.path.join(lidar_folder, f"{ts}.ply"))
                print(f"💾 已保存：{ts}")
                last_save_time = current_time

# ====================== 传感器回调 ======================
def handle_img(image):
    global latest_image
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))[:, :, :3]
    latest_image = img

def handle_lidar(lidar_data):
    global latest_lidar
    latest_lidar = lidar_data

camera.listen(handle_img)
lidar.listen(handle_lidar)

# ====================== 启动窗口线程 ======================
thread = threading.Thread(target=show_window, daemon=True)
thread.start()

# ====================== 主程序等待退出 ======================
try:
    while thread.is_alive():
        time.sleep(0.5)
except:
    pass
finally:
    stop_thread = True
    camera.stop()
    lidar.stop()
    cv2.destroyAllWindows()
    camera.destroy()
    lidar.destroy()
    vehicle.destroy()
    print("✅ 已安全退出！")