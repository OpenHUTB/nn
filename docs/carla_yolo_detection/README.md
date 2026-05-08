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
latest_depth = None

# AEB 距离阈值（米）
DIST_WARN   = 15.0
DIST_BRAKE  = 5.0

# 当前 AEB 状态
aeb_state = "NORMAL"

# LDW 偏移阈值（像素）：车道中心偏离画面中心超过这个值才报警
LDW_THRESHOLD = 60

def camera_callback(image):
    global latest_image
    latest_image = image

def depth_callback(image):
    global latest_depth
    latest_depth = image

def decode_depth(depth_image):
    raw = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    raw = raw.reshape((depth_image.height, depth_image.width, 4))
    R = raw[:, :, 2].astype(np.float32)
    G = raw[:, :, 1].astype(np.float32)
    B = raw[:, :, 0].astype(np.float32)
    depth_m = (R + G * 256.0 + B * 65536.0) / 16777215.0 * 1000.0
    return depth_m

def get_box_depth(depth_map, x1, y1, x2, y2):
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
    global aeb_state
    ctrl = carla.VehicleControl()

    if min_dist > DIST_WARN:
        new_state = "NORMAL"
    elif min_dist > DIST_BRAKE:
        new_state = "WARN"
    else:
        new_state = "BRAKE"
        ctrl.throttle = 0.0
        ctrl.brake    = 1.0
        vehicle.apply_control(ctrl)

    if new_state != aeb_state:
        aeb_state = new_state
        if new_state == "WARN":
            print(f"\n[⚠ AEB预警] 前方目标 {min_dist:.1f}m，注意！")
        elif new_state == "BRAKE":
            print(f"\n[🚨 紧急制动] 前方目标 {min_dist:.1f}m，已刹车！")
        else:
            print(f"\n[✅ AEB] 解除制动，恢复 autopilot")
            vehicle.set_autopilot(True, 8000)

    return aeb_state

# ── LDW 车道线检测 ────────────────────────────────────────────────────────
def detect_lanes(img_bgr):
    """
    输入：BGR图像
    输出：
      - img_lane: 叠加了车道线和引导区的图像（直接在副本上画）
      - offset: 车道中心相对画面中心的偏移像素（正=偏右，负=偏左）
      - ldw_state: "NORMAL" / "LEFT" / "RIGHT"
    """
    h, w = img_bgr.shape[:2]
    img_lane = img_bgr.copy()

    # 1. 灰度 + 高斯模糊 + Canny 边缘
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    # 2. ROI 梯形掩膜（只保留画面下半部分的路面）
    roi_vertices = np.array([[
        (0,          h),
        (w * 0.1,    h * 0.55),
        (w * 0.9,    h * 0.55),
        (w,          h),
    ]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 3. 霍夫变换找线段
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1, theta=np.pi/180,
        threshold=30,
        minLineLength=30,
        maxLineGap=100
    )

    left_lines, right_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # 斜率过滤：太平的线不是车道线
            if abs(slope) < 0.3:
                continue
            if slope < 0:          # 斜率为负 → 左车道线
                left_lines.append(line[0])
            else:                  # 斜率为正 → 右车道线
                right_lines.append(line[0])

    def average_line(line_group, img_h):
        """把一组线段拟合成一条从画面底部到ROI顶部的直线"""
        if not line_group:
            return None
        xs, ys = [], []
        for x1, y1, x2, y2 in line_group:
            xs += [x1, x2]
            ys += [y1, y2]
        # 一次多项式拟合
        try:
            fit = np.polyfit(ys, xs, 1)  # x = f(y)，更稳定
        except Exception:
            return None
        y_bottom = img_h
        y_top    = int(img_h * 0.55)
        x_bottom = int(np.polyval(fit, y_bottom))
        x_top    = int(np.polyval(fit, y_top))
        return (x_bottom, y_bottom, x_top, y_top, fit)

    left  = average_line(left_lines,  h)
    right = average_line(right_lines, h)

    # 4. 计算车道中心偏移
    img_cx = w // 2   # 画面中心（车头正前方）
    offset = 0
    ldw_state = "NORMAL"

    left_x_bottom  = left[0]  if left  else None
    right_x_bottom = right[0] if right else None

    if left_x_bottom is not None and right_x_bottom is not None:
        lane_cx = (left_x_bottom + right_x_bottom) // 2
        offset  = lane_cx - img_cx          # 正=车道中心在右=车偏左，负反之
    elif left_x_bottom is not None:
        offset = left_x_bottom - (img_cx - 160)   # 只有左线，估算偏移
    elif right_x_bottom is not None:
        offset = (img_cx + 160) - right_x_bottom

    if offset > LDW_THRESHOLD:
        ldw_state = "RIGHT"   # 车道中心在右边 → 车辆偏左
    elif offset < -LDW_THRESHOLD:
        ldw_state = "LEFT"    # 车道中心在左边 → 车辆偏右

    # 5. 绘制引导区（半透明蓝色填充）
    if left is not None and right is not None:
        lx_b, ly_b, lx_t, ly_t, _ = left
        rx_b, ry_b, rx_t, ry_t, _ = right
        pts = np.array([[lx_b, ly_b], [lx_t, ly_t],
                         [rx_t, ry_t], [rx_b, ry_b]], dtype=np.int32)
        overlay = img_lane.copy()
        cv2.fillPoly(overlay, [pts], (255, 180, 0))   # 蓝色填充
        cv2.addWeighted(overlay, 0.25, img_lane, 0.75, 0, img_lane)

    # 6. 画车道线
    lane_color = (255, 200, 0)   # 亮蓝色
    if ldw_state != "NORMAL":
        lane_color = (0, 80, 255)  # 偏离时变红

    if left is not None:
        lx_b, ly_b, lx_t, ly_t, _ = left
        cv2.line(img_lane, (lx_b, ly_b), (lx_t, ly_t), lane_color, 4)
    if right is not None:
        rx_b, ry_b, rx_t, ry_t, _ = right
        cv2.line(img_lane, (rx_b, ry_b), (rx_t, ry_t), lane_color, 4)

    # 7. 画车道中心线（虚线效果）
    if left is not None and right is not None:
        lx_b, ly_b, lx_t, ly_t, _ = left
        rx_b, ry_b, rx_t, ry_t, _ = right
        mid_b = ((lx_b + rx_b) // 2, h)
        mid_t = ((lx_t + rx_t) // 2, int(h * 0.55))
        # 虚线：每隔20px画一段
        seg = 20
        for i in range(0, 10):
            t0 = i / 10
            t1 = (i + 0.5) / 10
            p0 = (int(mid_b[0] + (mid_t[0] - mid_b[0]) * t0),
                  int(mid_b[1] + (mid_t[1] - mid_b[1]) * t0))
            p1 = (int(mid_b[0] + (mid_t[0] - mid_b[0]) * t1),
                  int(mid_b[1] + (mid_t[1] - mid_b[1]) * t1))
            cv2.line(img_lane, p0, p1, (0, 255, 255), 2)

    return img_lane, offset, ldw_state

# ─────────────────────────────────────────────────────────────────────────

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

    vehicle_classes = {'car', 'truck', 'bus', 'motorbike', 'bicycle', 'person'}

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()

        # 启动时先清理上次残留的所有车辆和传感器
        print("[INFO] 正在清理地图上的残留 actor...")
        all_actors = world.get_actors()
        vehicles = all_actors.filter('vehicle.*')
        sensors  = all_actors.filter('sensor.*')
        for a in list(sensors) + list(vehicles):
            a.destroy()
        print(f"[INFO] 清理完成：{len(list(vehicles))} 辆车，{len(list(sensors))} 个传感器")

        # 生成自车
        vehicle_bp   = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
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

        # 启动时把CARLA视角对准自车一次
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

        # RGB 摄像头
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '640')
        cam_bp.set_attribute('image_size_y', '480')
        cam_bp.set_attribute('fov', '90')
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        actor_list.append(camera)
        camera.listen(camera_callback)

        # 深度相机
        depth_bp = bp_lib.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '640')
        depth_bp.set_attribute('image_size_y', '480')
        depth_bp.set_attribute('fov', '90')
        depth_cam = world.spawn_actor(depth_bp, cam_transform, attach_to=vehicle)
        actor_list.append(depth_cam)
        depth_cam.listen(depth_callback)

        # 碰撞传感器
        col_bp = bp_lib.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        actor_list.append(collision_sensor)
        collision_sensor.listen(collision_handler)

        print("\n✅ 系统启动！按 Ctrl+C 退出...")

        while True:
            if latest_image is not None:
                start_time = time.time()

                img_data   = latest_image
                latest_image = None
                depth_data = latest_depth

                # 图像转换
                i  = np.array(img_data.raw_data)
                i2 = i.reshape((img_data.height, img_data.width, 4))
                img_bgr = i2[:, :, :3]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # 深度解码
                depth_map = decode_depth(depth_data) if depth_data is not None else None

                # ── LDW 车道线检测（车速过低时跳过，避免路口误检）──
                spd = vehicle.get_velocity()
                speed_ms = (spd.x**2 + spd.y**2 + spd.z**2) ** 0.5  # m/s
                if speed_ms > 1.5:  # 约5.4km/h，停车/低速不检测
                    img_display, lane_offset, ldw_state = detect_lanes(img_bgr)
                else:
                    img_display = img_bgr.copy()
                    lane_offset = 0
                    ldw_state   = "NORMAL" 

                # YOLO推理
                results    = model(img_rgb)
                detections = results.xyxy[0]

                min_dist = float('inf')

                for *xyxy, conf, cls in detections:
                    x1, y1, x2, y2 = map(int, xyxy)
                    label    = results.names[int(cls)]
                    conf_val = float(conf)

                    dist = -1.0
                    if depth_map is not None and label in vehicle_classes:
                        dist = get_box_depth(depth_map, x1, y1, x2, y2)
                        if dist > 0:
                            min_dist = min(min_dist, dist)

                    if dist <= 0:
                        color = (0, 255, 0)
                    elif dist > DIST_WARN:
                        color = (0, 255, 0)
                    elif dist > DIST_BRAKE:
                        color = (0, 200, 255)
                    else:
                        color = (0, 0, 255)

                    cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)

                    if dist > 0 and label in vehicle_classes:
                        text = f"{label}: {dist:.1f}m"
                    else:
                        text = f"{label} {conf_val:.0%}"

                    cv2.putText(img_display, text, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # AEB
                state = apply_aeb(vehicle, min_dist)

                if state == "WARN":
                    cv2.putText(img_display,
                                f"WARNING: {min_dist:.1f}m",
                                (10, img_data.height - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                elif state == "BRAKE":
                    overlay = img_display.copy()
                    cv2.rectangle(overlay, (0, 0), (img_data.width, img_data.height),
                                  (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.15, img_display, 0.85, 0, img_display)
                    cv2.putText(img_display,
                                f"EMERGENCY BRAKE! {min_dist:.1f}m",
                                (10, img_data.height - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # ── LDW 状态叠加 ──────────────────────────────────────
                if ldw_state == "LEFT":
                    cv2.putText(img_display, "⚠ LDW: 偏右！",
                                (img_data.width // 2 - 80, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
                elif ldw_state == "RIGHT":
                    cv2.putText(img_display, "⚠ LDW: 偏左！",
                                (img_data.width // 2 - 80, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

                # 偏移量数值显示（左上角第二行）
                offset_color = (0, 255, 0) if ldw_state == "NORMAL" else (0, 80, 255)
                cv2.putText(img_display,
                            f"Lane offset: {lane_offset:+d}px",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, offset_color, 2)

                # FPS
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(img_display, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 坐标
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

                cv2.imshow("CARLA YOLO + Depth AEB + LDW", img_display)
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