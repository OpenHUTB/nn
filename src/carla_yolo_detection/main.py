import carla
import random
import time
import numpy as np
import cv2
import torch
import warnings

warnings.filterwarnings("ignore")

print("正在加载 YOLOv5 神经网络模型...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
print(f"模型加载完毕！当前使用的计算设备是: {device.upper()}")

# 全局变量，只保存"最新的一帧"
latest_image = None
latest_depth = None  # 新增：最新深度帧

# AEB 距离阈值（米）
DIST_WARN   = 15.0   # 超过15m：正常行驶
DIST_BRAKE  = 5.0    # 低于5m：紧急制动

# 当前 AEB 状态（避免每帧重复打印）
aeb_state = "NORMAL"

def camera_callback(image):
    """RGB摄像头只负责把最新画面存起来"""
    global latest_image
    latest_image = image

def depth_callback(image):
    """深度相机只负责把最新深度帧存起来"""
    global latest_depth
    latest_depth = image

def decode_depth(depth_image):
    """
    把CARLA深度相机的原始数据解码成以米为单位的二维数组
    官方公式：depth = (R + G*256 + B*256*256) / (256^3 - 1) * 1000
    """
    raw = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    raw = raw.reshape((depth_image.height, depth_image.width, 4))  # BGRA
    R = raw[:, :, 2].astype(np.float32)
    G = raw[:, :, 1].astype(np.float32)
    B = raw[:, :, 0].astype(np.float32)
    depth_m = (R + G * 256.0 + B * 65536.0) / 16777215.0 * 1000.0
    return depth_m

def get_box_depth(depth_map, x1, y1, x2, y2):
    """
    取检测框中心区域的深度中位数（比单点采样更稳定，能过滤边缘噪声）
    """
    h, w = depth_map.shape
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bw = max(1, (x2 - x1) // 5)
    bh = max(1, (y2 - y1) // 5)
    px1, px2 = max(0, cx - bw), min(w - 1, cx + bw)
    py1, py2 = max(0, cy - bh), min(h - 1, cy + bh)
    patch = depth_map[py1:py2, px1:px2]
    if patch.size == 0:
        return -1.0
    return float(np.median(patch))

def apply_aeb(vehicle, min_dist):
    """
    三段式AEB控制：根据最近目标距离决定油门/刹车
    返回当前状态字符串，供画面叠加使用
    """
    global aeb_state
    ctrl = carla.VehicleControl()

    if min_dist > DIST_WARN:
        new_state = "NORMAL"
        # autopilot 自主行驶，不干预
    elif min_dist > DIST_BRAKE:
        new_state = "WARN"
        # 仅预警，路径还是交给 autopilot，不强行改速度
    else:
        new_state = "BRAKE"
        # 危险距离：强行覆盖 autopilot，紧急制动
        ctrl.throttle = 0.0
        ctrl.brake    = 1.0
        vehicle.apply_control(ctrl)

    # 只有状态切换时才打印，避免刷屏
    if new_state != aeb_state:
        aeb_state = new_state
        if new_state == "WARN":
            print(f"\n[⚠ AEB预警] 前方目标 {min_dist:.1f}m，注意！")
        elif new_state == "BRAKE":
            print(f"\n[🚨 紧急制动] 前方目标 {min_dist:.1f}m，已刹车！")
        else:
            print(f"\n[✅ AEB] 解除制动，恢复 autopilot")
            # 解除制动后把控制权还给 autopilot
            vehicle.set_autopilot(True, 8000)

    return aeb_state

def spawn_traffic(client, world, number_of_vehicles=30):
    bp_lib = world.get_blueprint_library()
    tm = client.get_trafficmanager(8000)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_synchronous_mode(False)

    vehicle_bps = bp_lib.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    temp_actors = []
    for i in range(min(number_of_vehicles, len(spawn_points))):
        bp = random.choice(vehicle_bps)
        npc = world.try_spawn_actor(bp, spawn_points[i])
        if npc:
            npc.set_autopilot(True, tm.get_port())
            temp_actors.append(npc)
    return temp_actors

def collision_handler(event):
    print(f"\n[💥碰撞预警] 发生碰撞! 撞到了: {event.other_actor.type_id}")

def main():
    global latest_image, latest_depth
    actor_list = []

    # 需要计算距离的目标类别（COCO类别名）
    vehicle_classes = {'car', 'truck', 'bus', 'motorbike', 'bicycle', 'person'}

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()

        # 启动时先清理上次残留的所有车辆和传感器，防止重复运行时乱撞
        print("[INFO] 正在清理地图上的残留 actor...")
        all_actors = world.get_actors()
        vehicles = all_actors.filter('vehicle.*')
        sensors = all_actors.filter('sensor.*')
        for a in list(sensors) + list(vehicles):
            a.destroy()
        print(f"[INFO] 清理完成：{len(list(vehicles))} 辆车，{len(list(sensors))} 个传感器")

        # 生成自车（固定出生点，每次位置一样，方便在地图上找）
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        # 从第0个出生点开始找，跳过被占用的，找到第一个能用的
        vehicle = None
        used_index = 0
        for idx, sp in enumerate(spawn_points):
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle:
                used_index = idx
                break
        if vehicle is None:
            raise RuntimeError("所有出生点都被占用，请重启模拟器后再试")
        actor_list.append(vehicle)
        vehicle.set_autopilot(True)
        print(f"[INFO] 自车出生点索引: {used_index}, 位置: {spawn_points[used_index].location}")

        # 启动时把CARLA视角对准自车一次，之后用户可自由移动
        spectator = world.get_spectator()
        t0 = vehicle.get_transform()
        spectator.set_transform(carla.Transform(
            t0.location + carla.Location(z=50),
            carla.Rotation(pitch=-90)
        ))
        print("[INFO] CARLA视角已对准自车，可自由移动视角")

        # 生成背景车流
        traffic_actors = spawn_traffic(client, world, 30)
        actor_list.extend(traffic_actors)

        # ── RGB 摄像头（与原来保持一致）──────────────────────────────
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '640')
        cam_bp.set_attribute('image_size_y', '480')
        cam_bp.set_attribute('fov', '90')
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        actor_list.append(camera)
        camera.listen(camera_callback)

        # ── 深度相机（与RGB相机同位置、同FOV，像素才能对应）─────────
        depth_bp = bp_lib.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '640')
        depth_bp.set_attribute('image_size_y', '480')
        depth_bp.set_attribute('fov', '90')
        depth_cam = world.spawn_actor(depth_bp, cam_transform, attach_to=vehicle)
        actor_list.append(depth_cam)
        depth_cam.listen(depth_callback)

        # ── 碰撞传感器（与原来保持一致）──────────────────────────────
        col_bp = bp_lib.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        actor_list.append(collision_sensor)
        collision_sensor.listen(collision_handler)

        print("\n✅ 系统启动！按 Ctrl+C 退出...")

        # 主循环：做YOLO推理 + 深度测距 + AEB控制
        while True:
            if latest_image is not None:
                start_time = time.time()

                # 取最新RGB帧
                img_data = latest_image
                latest_image = None

                # 取最新深度帧（可能为None，没关系）
                depth_data = latest_depth

                # 图像格式转换
                i = np.array(img_data.raw_data)
                i2 = i.reshape((img_data.height, img_data.width, 4))
                img_bgr = i2[:, :, :3]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # 解码深度图
                depth_map = decode_depth(depth_data) if depth_data is not None else None

                # YOLO推理
                results = model(img_rgb)
                detections = results.xyxy[0]  # [x1,y1,x2,y2,conf,cls]

                min_dist = float('inf')   # 本帧所有目标中的最近距离
                img_display = img_bgr.copy()

                for *xyxy, conf, cls in detections:
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = results.names[int(cls)]
                    conf_val = float(conf)

                    # ── 深度查询 ──────────────────────────────────────
                    dist = -1.0
                    if depth_map is not None and label in vehicle_classes:
                        dist = get_box_depth(depth_map, x1, y1, x2, y2)
                        if dist > 0:
                            min_dist = min(min_dist, dist)

                    # ── 框颜色随距离变化 ──────────────────────────────
                    if dist <= 0:
                        color = (0, 255, 0)      # 绿：无距离信息
                    elif dist > DIST_WARN:
                        color = (0, 255, 0)      # 绿：安全
                    elif dist > DIST_BRAKE:
                        color = (0, 200, 255)    # 黄：预警
                    else:
                        color = (0, 0, 255)      # 红：危险

                    cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)

                    # ── 标签：有距离就显示距离，否则显示置信度 ─────────
                    if dist > 0 and label in vehicle_classes:
                        text = f"{label}: {dist:.1f}m"
                    else:
                        text = f"{label} {conf_val:.0%}"

                    cv2.putText(img_display, text, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # ── AEB 控制 ──────────────────────────────────────────
                state = apply_aeb(vehicle, min_dist)

                # ── 画面叠加AEB状态 ───────────────────────────────────
                if state == "WARN":
                    cv2.putText(img_display,
                                f"WARNING: {min_dist:.1f}m",
                                (10, img_data.height - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                elif state == "BRAKE":
                    # 红色半透明覆盖
                    overlay = img_display.copy()
                    cv2.rectangle(overlay, (0, 0), (img_data.width, img_data.height),
                                  (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.15, img_display, 0.85, 0, img_display)
                    cv2.putText(img_display,
                                f"EMERGENCY BRAKE! {min_dist:.1f}m",
                                (10, img_data.height - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # ── FPS + 坐标（方便在地图上找车）─────────────────
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(img_display, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                loc = vehicle.get_transform()
                coord_text = f"X:{loc.location.x:.1f} Y:{loc.location.y:.1f} Yaw:{loc.rotation.yaw:.0f}deg"
                cv2.putText(img_display, coord_text, (10, img_data.height - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                # AEB状态（右上角）
                state_color = {'NORMAL': (0,255,0), 'WARN': (0,200,255), 'BRAKE': (0,0,255)}
                cv2.putText(img_display, f"AEB: {state}",
                            (img_data.width - 180, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            state_color.get(state, (255,255,255)), 2)

                cv2.imshow("CARLA YOLO + Depth AEB", img_display)
                cv2.waitKey(1)
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n正在关闭系统...")
    finally:
        for actor in actor_list:
            if actor is not None:
                actor.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()