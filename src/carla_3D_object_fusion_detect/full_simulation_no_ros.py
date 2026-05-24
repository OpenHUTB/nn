import carla
import time
import random
import os
import cv2
import numpy as np
import sys
import math
import traceback

# ====================== 路径 ======================
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

# ====================== 配置 ======================
SAVE_INTERVAL = 5 * 60
COLLISION_COOLDOWN_SEC = 3.0

# 动态交通配置（安全保守值）
MAX_VEHICLES = 4          # 同时最多4辆
MAX_PEDESTRIANS = 5       # 最多5个行人
SPAWN_RADIUS = 40         # 生成半径
REMOVE_DISTANCE = 80      # 移除距离
SPAWN_INTERVAL = 5.0      # 每5秒尝试生成一次
MAX_SPAWN_ATTEMPTS = 1    # 每次只尝试1个

# 全局变量
last_save_time = time.time()
latest_camera = None
latest_follow = None
latest_lidar = None
collision_cooldown = False
collision_cooldown_time = 0
error_count = 0
frame_count = 0
fps = 0
last_fps_time = time.time()

# 动态生成相关
spawned_vehicles = []
spawned_pedestrians = []
all_spawned_actors = []
last_spawn_time = time.time()
vehicle_blueprints = []
pedestrian_blueprints = []

# ====================== 连接 CARLA（带重试） ======================
def connect_carla(retries=3):
    for i in range(retries):
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(10.0)
            client.get_server_version()
            print(f"✅ 连接成功，版本: {client.get_server_version()}")
            return client
        except Exception as e:
            print(f"连接失败 ({i+1}/{retries}): {e}")
            time.sleep(2)
    print("❌ 无法连接 CARLA，请确保模拟器已启动")
    sys.exit(1)

client = connect_carla()
world = client.get_world()

# 雨天天气
weather = carla.WeatherParameters(
    cloudiness=90.0, precipitation=90.0, precipitation_deposits=90.0,
    wind_intensity=20.0, wetness=90.0
)
world.set_weather(weather)
print("✅ 雨天天气")

# 生成自车
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_points = world.get_map().get_spawn_points()
if not spawn_points:
    raise RuntimeError("地图无生成点")
spawn_point = random.choice(spawn_points)
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
if vehicle is None:
    raise RuntimeError("自车生成失败")
vehicle.set_autopilot(True)
print("✅ 自车已生成")

# 准备蓝图
def prepare_blueprints():
    global vehicle_blueprints, pedestrian_blueprints
    try:
        all_vehicles = blueprint_library.filter('vehicle.*')
        vehicle_blueprints = [v for v in all_vehicles 
                              if int(v.get_attribute('number_of_wheels')) == 4 
                              and 'tesla' not in v.id]
        pedestrian_blueprints = list(blueprint_library.filter('walker.pedestrian.*'))
        print(f"✅ 车辆蓝图: {len(vehicle_blueprints)}, 行人蓝图: {len(pedestrian_blueprints)}")
    except Exception as e:
        print(f"准备蓝图失败: {e}")
        vehicle_blueprints = []
        pedestrian_blueprints = []

prepare_blueprints()

# ====================== 安全生成辅助 ======================
def is_location_occupied(location, radius=2.5):
    try:
        actors = world.get_actors()
        for actor in actors:
            if actor.id == vehicle.id:
                continue
            if actor.get_location().distance(location) < radius:
                return True
        return False
    except:
        return True

def spawn_random_vehicle_near(ego_location):
    if len(spawned_vehicles) >= MAX_VEHICLES or not vehicle_blueprints:
        return None
    available = []
    for sp in spawn_points:
        if sp.location.distance(ego_location) < SPAWN_RADIUS:
            if not is_location_occupied(sp.location, radius=3.0):
                available.append(sp)
    if not available:
        return None
    chosen = random.choice(available)
    blueprint = random.choice(vehicle_blueprints)
    try:
        new_vehicle = world.spawn_actor(blueprint, chosen)
        if new_vehicle:
            new_vehicle.set_autopilot(True)
            spawned_vehicles.append(new_vehicle)
            all_spawned_actors.append(new_vehicle)
            # print(f"🚗 生成车辆: {blueprint.id}")
            return new_vehicle
    except Exception as e:
        print(f"生成车辆异常: {e}")
    return None

def spawn_random_pedestrian_near(ego_location):
    if len(spawned_pedestrians) >= MAX_PEDESTRIANS or not pedestrian_blueprints:
        return None
    angle = random.uniform(0, 2*math.pi)
    radius = random.uniform(12, SPAWN_RADIUS)
    x = ego_location.x + radius * math.cos(angle)
    y = ego_location.y + radius * math.sin(angle)
    z = ego_location.z + 0.5
    spawn_loc = carla.Location(x=x, y=y, z=z)
    if is_location_occupied(spawn_loc, radius=1.5):
        return None
    blueprint = random.choice(pedestrian_blueprints)
    try:
        new_walker = world.spawn_actor(blueprint, carla.Transform(spawn_loc))
        if new_walker:
            controller_bp = blueprint_library.find('controller.ai.walker')
            controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=new_walker)
            if controller:
                controller.start()
                target_angle = random.uniform(0, 2*math.pi)
                target_dist = random.uniform(10, 20)
                target_loc = spawn_loc + carla.Location(x=target_dist*math.cos(target_angle),
                                                        y=target_dist*math.sin(target_angle))
                controller.go_to_location(target_loc)
                all_spawned_actors.append(controller)
            spawned_pedestrians.append(new_walker)
            all_spawned_actors.append(new_walker)
            # print(f"🚶 生成行人")
            return new_walker
    except Exception as e:
        print(f"生成行人异常: {e}")
    return None

def remove_far_actors(ego_location):
    global spawned_vehicles, spawned_pedestrians, all_spawned_actors
    to_remove_v = [v for v in spawned_vehicles if v.get_location().distance(ego_location) > REMOVE_DISTANCE]
    for v in to_remove_v:
        try:
            v.destroy()
            spawned_vehicles.remove(v)
            all_spawned_actors.remove(v)
        except:
            pass
    to_remove_w = [w for w in spawned_pedestrians if w.get_location().distance(ego_location) > REMOVE_DISTANCE]
    for w in to_remove_w:
        try:
            for a in all_spawned_actors[:]:
                if hasattr(a, 'parent_id') and a.parent_id == w.id:
                    a.stop()
                    a.destroy()
                    all_spawned_actors.remove(a)
                    break
            w.destroy()
            spawned_pedestrians.remove(w)
            all_spawned_actors.remove(w)
        except:
            pass

# ====================== 传感器创建 ======================
def spawn_safe_sensor(bp_name, transform, attach_to):
    try:
        bp = blueprint_library.find(bp_name)
        if bp is None:
            print(f"找不到蓝图 {bp_name}")
            return None
        return world.spawn_actor(bp, transform, attach_to=attach_to)
    except Exception as e:
        print(f"传感器 {bp_name} 生成失败: {e}")
        return None

camera_front = spawn_safe_sensor('sensor.camera.rgb',
                                 carla.Transform(carla.Location(x=1.5, z=2.4)),
                                 vehicle)
camera_follow = spawn_safe_sensor('sensor.camera.rgb',
                                  carla.Transform(carla.Location(x=-5.0, y=0, z=3.0), carla.Rotation(pitch=-10)),
                                  vehicle)
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
if lidar_bp:
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('points_per_second', '50000')
    lidar_bp.set_attribute('rotation_frequency', '10')
lidar = spawn_safe_sensor('sensor.lidar.ray_cast',
                          carla.Transform(carla.Location(x=0, z=2.5)),
                          vehicle)
collision_sensor = spawn_safe_sensor('sensor.other.collision',
                                     carla.Transform(),
                                     vehicle)

# ====================== 回调函数 ======================
def safe_callback(func):
    def wrapper(data):
        try:
            func(data)
        except Exception as e:
            global error_count
            error_count += 1
            if error_count % 100 == 1:
                print(f"回调错误: {e}")
    return wrapper

@safe_callback
def on_camera_front(data):
    global latest_camera
    img = np.frombuffer(data.raw_data, dtype=np.uint8)
    img = img.reshape((data.height, data.width, 4))[:, :, :3]
    latest_camera = img

@safe_callback
def on_camera_follow(data):
    global latest_follow
    img = np.frombuffer(data.raw_data, dtype=np.uint8)
    img = img.reshape((data.height, data.width, 4))[:, :, :3]
    latest_follow = img

@safe_callback
def on_lidar(data):
    global latest_lidar
    latest_lidar = data

@safe_callback
def on_collision(event):
    global collision_cooldown, collision_cooldown_time
    now = time.time()
    if collision_cooldown and (now - collision_cooldown_time) < COLLISION_COOLDOWN_SEC:
        return
    collision_cooldown = True
    collision_cooldown_time = now
    try:
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
    except Exception as e:
        print(f"碰撞保存失败: {e}")

if camera_front: camera_front.listen(on_camera_front)
if camera_follow: camera_follow.listen(on_camera_follow)
if lidar: lidar.listen(on_lidar)
if collision_sensor: collision_sensor.listen(on_collision)

# ====================== 车速表绘制 ======================
def draw_speedometer(image, vehicle):
    try:
        vel = vehicle.get_velocity()
        speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        max_speed = 120.0
        ratio = min(speed/max_speed, 1.0)
        bar_w, bar_h = 200, 20
        x, y = 20, 20
        cv2.rectangle(image, (x, y), (x+bar_w, y+bar_h), (50,50,50), -1)
        fill = int(bar_w * ratio)
        color = (0,255,0) if speed<80 else (0,165,255) if speed<120 else (0,0,255)
        cv2.rectangle(image, (x, y), (x+fill, y+bar_h), color, -1)
        cv2.putText(image, f"{int(speed)} km/h", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(image, f"Veh:{len(spawned_vehicles)}  Ped:{len(spawned_pedestrians)}", 
                    (x, y+bar_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(image, f"FPS:{fps:.1f} Err:{error_count}", 
                    (x, y+bar_h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    except Exception as e:
        pass

# ====================== 等待传感器就绪 ======================
print("等待传感器数据...", end="")
timeout_start = time.time()
while (latest_follow is None or latest_camera is None or latest_lidar is None) and (time.time() - timeout_start < 12):
    time.sleep(0.2)
    print(".", end="", flush=True)
print()
if latest_follow is not None:
    print("✅ 传感器就绪")
else:
    print("⚠️ 部分传感器未就绪，继续运行")

# ====================== 主循环 ======================
print(f"每 {SAVE_INTERVAL//60} 分钟自动保存，碰撞自动保存，动态交通已启用（最多{MAX_VEHICLES}车/{MAX_PEDESTRIANS}人）")
print("按 Q/ESC 退出")

loop_counter = 0
try:
    while True:
        loop_counter += 1
        now = time.time()
        # 每 500 帧打印一次心跳，证明程序还在运行
        if loop_counter % 500 == 0:
            print(f"♥ 心跳: 已运行 {loop_counter} 帧, 车辆={len(spawned_vehicles)}, 行人={len(spawned_pedestrians)}")

        # 获取自车位置
        try:
            ego_loc = vehicle.get_location()
        except Exception as e:
            print(f"❌ 获取自车位置失败: {e}")
            # 尝试重新获取 world 和 vehicle（可能是服务器重启）
            try:
                world = client.get_world()
                vehicle = world.get_actor(vehicle.id)
                continue
            except:
                break

        # 动态生成
        if now - last_spawn_time >= SPAWN_INTERVAL:
            for _ in range(MAX_SPAWN_ATTEMPTS):
                if len(spawned_vehicles) < MAX_VEHICLES and random.random() < 0.6:
                    spawn_random_vehicle_near(ego_loc)
                if len(spawned_pedestrians) < MAX_PEDESTRIANS and random.random() < 0.4:
                    spawn_random_pedestrian_near(ego_loc)
            last_spawn_time = now

        # 移除远处 actor
        remove_far_actors(ego_loc)

        # 重置碰撞冷却
        if collision_cooldown and (now - collision_cooldown_time) >= COLLISION_COOLDOWN_SEC:
            collision_cooldown = False

        # FPS 计算
        frame_count += 1
        if now - last_fps_time >= 1.0:
            fps = frame_count / (now - last_fps_time)
            frame_count = 0
            last_fps_time = now

        # 画面合成
        display = None
        if latest_follow is not None:
            display = latest_follow.copy()
            draw_speedometer(display, vehicle)
            if latest_camera is not None:
                h, w = display.shape[:2]
                sw = max(160, w//4)
                sh = int(sw * latest_camera.shape[0] / latest_camera.shape[1])
                small = cv2.resize(latest_camera, (sw, sh))
                x = w - sw - 10
                y = h - sh - 10
                if x > 0 and y > 0:
                    display[y:y+sh, x:x+sw] = small
                    cv2.rectangle(display, (x-1,y-1), (x+sw+1,y+sh+1), (255,255,255), 2)
        elif latest_camera is not None:
            display = latest_camera.copy()
            draw_speedometer(display, vehicle)
        else:
            display = np.zeros((600,800,3), dtype=np.uint8)
            cv2.putText(display, "Waiting for sensors...", (50,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        if display is not None:
            cv2.imshow("CARLA", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("用户主动退出")
            break

        # 定时保存
        if now - last_save_time >= SAVE_INTERVAL:
            if latest_camera is not None and latest_lidar is not None:
                ts = str(int(now))
                cv2.imwrite(os.path.join(image_folder, f"{ts}.png"), latest_camera)
                latest_lidar.save_to_disk(os.path.join(lidar_folder, f"{ts}.ply"))
                print(f"💾 定时保存 {ts}")
                last_save_time = now

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n用户中断")
except Exception as e:
    print(f"主循环异常: {e}")
    traceback.print_exc()
finally:
    cv2.destroyAllWindows()
    # 清理所有动态生成的 actor
    for actor in all_spawned_actors:
        if actor:
            try:
                if hasattr(actor, 'stop'): actor.stop()
                actor.destroy()
            except:
                pass
    # 清理传感器和自车
    for actor in [camera_front, camera_follow, lidar, collision_sensor, vehicle]:
        if actor:
            try:
                if hasattr(actor, 'stop'): actor.stop()
                actor.destroy()
            except:
                pass
    print("✅ 已清理退出")