import carla
import time
import random
import os
import cv2
import numpy as np

# ====================== 路径（上3级目录） ======================
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
p1 = os.path.dirname(current_dir)
p2 = os.path.dirname(p1)
p3 = os.path.dirname(p2)

image_folder = os.path.join(p3, "images")
lidar_folder = os.path.join(p3, "lidar")
os.makedirs(image_folder, exist_ok=True)
os.makedirs(lidar_folder, exist_ok=True)

SAVE_INTERVAL = 5 * 60
last_save_time = time.time()

# 传感器数据（全局变量）
latest_camera = None       # 车载相机（前向）
latest_follow = None       # 跟随相机（车外后方）
latest_lidar = None

# ====================== 连接 CARLA ======================
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()

# 雨天天气
weather = carla.WeatherParameters(
    cloudiness=90.0, precipitation=90.0, precipitation_deposits=90.0,
    wind_intensity=20.0, wetness=90.0
)
world.set_weather(weather)
print("✅ 雨天天气已设置")

# 生成车辆
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_points = world.get_map().get_spawn_points()
if not spawn_points:
    raise RuntimeError("地图没有生成点")
spawn_point = random.choice(spawn_points)
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
if vehicle is None:
    raise RuntimeError("车辆生成失败")
vehicle.set_autopilot(True)
print("✅ 车辆生成并开启自动驾驶")

# ====================== 传感器创建 ======================
# 车载相机（前向）
cam_bp = blueprint_library.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '800')
cam_bp.set_attribute('image_size_y', '600')
cam_bp.set_attribute('fov', '110')
camera_front = world.spawn_actor(
    cam_bp,
    carla.Transform(carla.Location(x=1.5, z=2.4)),
    attach_to=vehicle
)

# 跟随相机（车外后方）
follow_bp = blueprint_library.find('sensor.camera.rgb')
follow_bp.set_attribute('image_size_x', '1024')
follow_bp.set_attribute('image_size_y', '768')
follow_bp.set_attribute('fov', '90')
camera_follow = world.spawn_actor(
    follow_bp,
    carla.Transform(carla.Location(x=-5.0, y=0, z=3.0), carla.Rotation(pitch=-10)),
    attach_to=vehicle
)

# 激光雷达
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '100')
lidar_bp.set_attribute('points_per_second', '100000')
lidar_bp.set_attribute('rotation_frequency', '10')
lidar = world.spawn_actor(
    lidar_bp,
    carla.Transform(carla.Location(x=0, z=2.5)),
    attach_to=vehicle
)

print("✅ 所有传感器已挂载")
print(f"⏱️  每 {SAVE_INTERVAL//60} 分钟自动保存（车载相机图像 + 雷达点云）")
print("🎥 按 Q 或 ESC 退出")

# ====================== 回调函数（注意 global 声明） ======================
def on_camera_front(data):
    global latest_camera
    img = np.frombuffer(data.raw_data, dtype=np.uint8)
    img = img.reshape((data.height, data.width, 4))[:, :, :3]
    latest_camera = img

def on_camera_follow(data):
    global latest_follow
    img = np.frombuffer(data.raw_data, dtype=np.uint8)
    img = img.reshape((data.height, data.width, 4))[:, :, :3]
    latest_follow = img

def on_lidar(data):
    global latest_lidar
    latest_lidar = data

# 订阅
camera_front.listen(on_camera_front)
camera_follow.listen(on_camera_follow)
lidar.listen(on_lidar)

# ====================== 主循环（无线程，安全显示） ======================
try:
    while True:
        # 显示画中画
        if latest_follow is not None:
            display = latest_follow.copy()
            if latest_camera is not None:
                h, w = display.shape[:2]
                small_w = max(160, w // 4)
                small_h = int(small_w * latest_camera.shape[0] / latest_camera.shape[1])
                small_img = cv2.resize(latest_camera, (small_w, small_h))
                x = w - small_w - 10
                y = h - small_h - 10
                if x > 0 and y > 0:
                    display[y:y+small_h, x:x+small_w] = small_img
                    cv2.rectangle(display, (x-1, y-1), (x+small_w+1, y+small_h+1), (255,255,255), 2)
            cv2.imshow("CARLA 驾驶视角 (右下角车载相机)", display)
        elif latest_camera is not None:
            cv2.imshow("CARLA (等待跟随相机)", latest_camera)
        else:
            placeholder = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for sensors...", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("CARLA", placeholder)

        # 按键退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        # 定时保存（车载相机图像 + 雷达点云）
        now = time.time()
        if now - last_save_time >= SAVE_INTERVAL:
            if latest_camera is not None and latest_lidar is not None:
                ts = str(int(now))
                cv2.imwrite(os.path.join(image_folder, f"{ts}.png"), latest_camera)
                latest_lidar.save_to_disk(os.path.join(lidar_folder, f"{ts}.ply"))
                print(f"💾 已保存：{ts}")
                last_save_time = now
            else:
                print(f"⏳ 等待传感器数据，暂未保存 ({now:.0f})")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n⚠️ 用户中断")
finally:
    cv2.destroyAllWindows()
    if camera_front:
        camera_front.stop()
        camera_front.destroy()
    if camera_follow:
        camera_follow.stop()
        camera_follow.destroy()
    if lidar:
        lidar.stop()
        lidar.destroy()
    if vehicle:
        vehicle.destroy()
    print("✅ 已安全退出")