import carla
import time
import random
import os
import cv2
import numpy as np
import sys

# ====================== 路径（上3级目录） ======================
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
p1 = os.path.dirname(current_dir)
p2 = os.path.dirname(p1)
p3 = os.path.dirname(p2)

image_folder = os.path.join(p3, "images")
lidar_folder = os.path.join(p3, "lidar")
collision_folder = os.path.join(p3, "collision")
os.makedirs(image_folder, exist_ok=True)
os.makedirs(lidar_folder, exist_ok=True)
os.makedirs(collision_folder, exist_ok=True)

SAVE_INTERVAL = 5 * 60
last_save_time = time.time()

# 传感器数据
latest_camera = None       # 车载相机（前向）
latest_follow = None       # 跟随相机（车外后方）
latest_lidar = None

# 碰撞冷却
collision_cooldown = False
collision_cooldown_time = 0
COLLISION_COOLDOWN_SEC = 3.0

# ====================== 连接 CARLA ======================
print("正在连接 CARLA 服务器 (localhost:2000)...")
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)   # 增加超时时间

try:
    client.get_server_version()  # 测试连接
    print(f"✅ CARLA 服务器版本: {client.get_server_version()}")
    print(f"✅ CARLA 客户端版本: {client.get_client_version()}")
except Exception as e:
    print(f"❌ 无法连接 CARLA 服务器: {e}")
    print("请确保 CARLA 模拟器已启动 (CarlaUE4.exe 或 ./CarlaUE4.sh)")
    sys.exit(1)

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
def spawn_camera(attach_to, transform, size_x, size_y, fov):
    bp = blueprint_library.find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(size_x))
    bp.set_attribute('image_size_y', str(size_y))
    bp.set_attribute('fov', str(fov))
    return world.spawn_actor(bp, transform, attach_to=attach_to)

# 车载相机（前向）
camera_front = spawn_camera(vehicle, carla.Transform(carla.Location(x=1.5, z=2.4)), 800, 600, 110)
# 跟随相机（车外后方）
camera_follow = spawn_camera(vehicle, carla.Transform(carla.Location(x=-5.0, y=0, z=3.0), carla.Rotation(pitch=-10)), 1024, 768, 90)

# 激光雷达
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '100')
lidar_bp.set_attribute('points_per_second', '100000')
lidar_bp.set_attribute('rotation_frequency', '10')
lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0, z=2.5)), attach_to=vehicle)

# 碰撞传感器
collision_bp = blueprint_library.find('sensor.other.collision')
collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

print("✅ 所有传感器已挂载（相机 x2，激光雷达，碰撞）")

# ====================== 回调函数 ======================
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

def on_collision(event):
    global collision_cooldown, collision_cooldown_time
    now = time.time()
    if collision_cooldown and (now - collision_cooldown_time) < COLLISION_COOLDOWN_SEC:
        return
    collision_cooldown = True
    collision_cooldown_time = now
    impulse = event.normal_impulse
    magnitude = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    if latest_camera is not None:
        img_path = os.path.join(collision_folder, f"collision_{ts}_camera.png")
        cv2.imwrite(img_path, latest_camera)
        print(f"💥 碰撞图像: {img_path} (力度={magnitude:.2f})")
    if latest_lidar is not None:
        lidar_path = os.path.join(collision_folder, f"collision_{ts}_lidar.ply")
        latest_lidar.save_to_disk(lidar_path)
        print(f"💥 碰撞点云: {lidar_path}")

camera_front.listen(on_camera_front)
camera_follow.listen(on_camera_follow)
lidar.listen(on_lidar)
collision_sensor.listen(on_collision)

# ====================== 辅助函数：车速表 ======================
def draw_speedometer(image, vehicle):
    velocity = vehicle.get_velocity()
    speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    max_speed = 120.0
    speed_ratio = min(speed_kmh / max_speed, 1.0)
    h, w = image.shape[:2]
    bar_width = 200
    bar_height = 20
    bar_x = 20
    bar_y = 20
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50,50,50), -1)
    fill_width = int(bar_width * speed_ratio)
    color = (0, 255, 0) if speed_kmh < 80 else (0, 165, 255) if speed_kmh < 120 else (0, 0, 255)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
    text = f"{int(speed_kmh)} km/h"
    cv2.putText(image, text, (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    control = vehicle.get_control()
    gear_text = f"Gear: {control.gear}" if control.gear != 0 else "Gear: D"
    cv2.putText(image, gear_text, (bar_x, bar_y + bar_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

# ====================== 等待传感器就绪 ======================
print("等待传感器数据...", end="", flush=True)
timeout = 10  # 最多等待10秒
start = time.time()
while (latest_follow is None or latest_camera is None or latest_lidar is None) and (time.time() - start < timeout):
    time.sleep(0.1)
    print(".", end="", flush=True)
if latest_follow is None and latest_camera is None:
    print("\n⚠️ 传感器数据未就绪，请检查 CARLA 是否正常运行")
else:
    print("\n✅ 传感器数据已就绪，开始显示画面")

# ====================== 主循环 ======================
print(f"⏱️  每 {SAVE_INTERVAL//60} 分钟自动保存车载图像+雷达点云")
print("💥 发生碰撞时自动保存碰撞瞬间数据")
print("🎥 按 Q 或 ESC 退出")

try:
    while True:
        # 重置碰撞冷却
        if collision_cooldown and (time.time() - collision_cooldown_time) >= COLLISION_COOLDOWN_SEC:
            collision_cooldown = False

        # 显示画中画 + 车速表
        if latest_follow is not None:
            display = latest_follow.copy()
            draw_speedometer(display, vehicle)
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
            cv2.imshow("CARLA 驾驶视角 (车速表+碰撞保存)", display)
        elif latest_camera is not None:
            display = latest_camera.copy()
            draw_speedometer(display, vehicle)
            cv2.imshow("CARLA (等待跟随相机)", display)
        else:
            placeholder = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for sensors...", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("CARLA", placeholder)

        # 按键退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        # 定时保存
        now = time.time()
        if now - last_save_time >= SAVE_INTERVAL:
            if latest_camera is not None and latest_lidar is not None:
                ts = str(int(now))
                cv2.imwrite(os.path.join(image_folder, f"{ts}.png"), latest_camera)
                latest_lidar.save_to_disk(os.path.join(lidar_folder, f"{ts}.ply"))
                print(f"💾 定时保存：{ts}")
                last_save_time = now

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n⚠️ 用户中断")
finally:
    cv2.destroyAllWindows()
    for actor in [camera_front, camera_follow, lidar, collision_sensor, vehicle]:
        if actor is not None:
            try:
                if hasattr(actor, 'stop'):
                    actor.stop()
                actor.destroy()
            except Exception as e:
                print(f"清理资源时出错: {e}")
    print("✅ 已安全退出")