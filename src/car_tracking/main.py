"""
CARLA 0.9.14 低画质版专用脚本
- 适配0.9.14 API差异（移除road_type、修正Vector2D.is_zero、修正异常处理）
- 实现严格的红灯停、绿灯行逻辑
- 车辆沿道路正常行驶，路口自然拐弯（兼容0.9.14）
- 支持地图随机生成
- 新增：3D车辆转2D边界框可视化（含车辆ID、距离筛选）
"""
import sys
import os
import carla
import cv2
import numpy as np
import queue
import math
import random
import colorsys  # 用于生成边界框颜色

# 全局变量
IMAGE_QUEUE = queue.Queue(maxsize=2)  # 增大队列，避免图像丢失
LATEST_IMAGE = None  # 存储最新图像，防止黑屏
# 替换为你的低画质CARLA实际路径
CARLA_ROOT = 'D:/123/apps/CARLA_0.9.14/WindowsNoEditor'

# ===================== 3D转2D可视化核心工具函数（适配0.9.14） =====================
def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """构建相机投影矩阵（将3D世界坐标投影到2D图像）"""
    focal = w / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    if is_behind_camera:
        K[2, 2] = -1  # 反转z轴处理相机后方的点
    return K

def get_image_point(loc, K, w2c):
    """将Carla的3D位置转换为2D图像坐标（适配0.9.14）"""
    point = np.array([loc.x, loc.y, loc.z, 1.0])
    point_camera = np.dot(w2c, point)  # 世界→相机
    point_img = np.dot(K, point_camera[:3])  # 相机→图像
    if point_img[2] != 0:  # 避免除零错误（适配低画质版）
        point_img = point_img / point_img[2]  # 归一化
    return (point_img[0], point_img[1])

def get_2d_box_from_3d_edges(points_2d, h, w):
    """从3D点的2D投影生成最小包围框"""
    x_coords = [p[0] for p in points_2d]
    y_coords = [p[1] for p in points_2d]
    x_min = max(0, min(x_coords))
    x_max = min(w, max(x_coords))
    y_min = max(0, min(y_coords))
    y_max = min(h, max(y_coords))
    return x_min, y_min, x_max, y_max  # 调整返回顺序，匹配cv2.rectangle

def point_in_canvas(point, h, w):
    """检查2D点是否在图像画布范围内"""
    x, y = point
    return 0 <= x < w and 0 <= y < h

def generate_color(class_id, num_classes=1):
    """为类别生成唯一颜色（这里仅车辆，固定颜色）"""
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), colors))
    return colors[class_id % num_classes]

def draw_bounding_boxes(image, boxes, ids, class_names):
    """在图像上绘制边界框（含车辆ID）"""
    image_copy = image.copy()
    color = generate_color(0)  # 车辆类别固定为0，颜色为红色系
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        # 绘制矩形框
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        # 准备文本（车辆ID）
        text = f"Car (ID: {ids[i]})" if ids and i < len(ids) else "Car"
        # 绘制文本背景和文字
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10
        cv2.rectangle(image_copy, (x1, text_y - text_size[1] - 2),
                      (x1 + text_size[0], text_y + 2), color, -1)
        cv2.putText(image_copy, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    return image_copy

# ===================== 原脚本核心功能函数 =====================
# 摄像头回调函数（低画质适配，新增图像存储）
def image_callback(image):
    global LATEST_IMAGE
    try:
        img_bgra = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_bgra = img_bgra.reshape((image.height, image.width, 4))
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
        img_bgr = cv2.resize(img_bgr, (640, 360))  # 统一分辨率

        # 存储最新图像，防止黑屏
        LATEST_IMAGE = img_bgr.copy()

        # 处理队列，避免满溢
        if IMAGE_QUEUE.full():
            IMAGE_QUEUE.get_nowait()
        IMAGE_QUEUE.put(img_bgr, timeout=0.1)
    except Exception as e:
        print(f"⚠️ 图像回调出错：{e}")

# 计算两个向量的夹角（用于转向控制，平滑转向）
# 适配0.9.14：移除Vector2D.is_zero()，改用长度判断
def calculate_angle(current_transform, target_location):
    # 获取车辆的前进方向向量（仅平面，忽略z轴）
    forward = current_transform.get_forward_vector()
    forward_flat = carla.Vector2D(forward.x, forward.y)
    # 计算车辆到目标点的方向向量
    target_flat = carla.Vector2D(
        target_location.x - current_transform.location.x,
        target_location.y - current_transform.location.y
    )
    # 归一化向量（避免长度影响夹角计算）
    if forward_flat.length() > 0:
        forward_flat = forward_flat / forward_flat.length()
    if target_flat.length() > 0:
        target_flat = target_flat / target_flat.length()
    # 计算夹角（弧度），范围[-π, π]
    dot = forward_flat.x * target_flat.x + forward_flat.y * target_flat.y
    cross = forward_flat.x * target_flat.y - forward_flat.y * target_flat.x
    angle = math.atan2(cross, dot)
    return angle

# 选择路口的路径点（适配0.9.14：不使用road_type，优先选直走/主方向）
def choose_main_waypoint(waypoint):
    # 获取下一组路径点（间距5米，更远的距离能更好识别路口）
    next_waypoints = waypoint.next(5.0)
    if not next_waypoints:
        return waypoint
    # 优先选择第一个路径点（主方向，避免拐入小巷）
    main_waypoint = next_waypoints[0]
    return main_waypoint

def main():
    camera = None
    vehicle = None
    current_waypoint = None  # 动态更新的当前目标路径点

    # 检查CARLA进程是否运行
    def check_carla_running():
        try:
            import psutil
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] == 'CarlaUE4.exe':
                    return True
            return False
        except ImportError:
            print("⚠️ 未安装psutil，默认认为CARLA进程已运行")
            return True

    # 前置检查
    print("=" * 60)
    print("--- [低画质CARLA环境检查] ---")
    if not check_carla_running():
        print("❌ 错误：未检测到CARLA服务器运行！")
        print(f"   请先启动：{os.path.join(CARLA_ROOT, 'CarlaUE4.exe')}")
        print("   （建议使用低画质快捷方式启动）")
        return
    print("✅ 检测到CARLA服务器运行")
    print("--- [环境检查完成] ---")
    print("=" * 60)

    try:
        # 1. 连接CARLA服务器（低画质版超时延长）
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(60.0)

        # 随机选择地图
        available_maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']
        random_map = random.choice(available_maps)
        world = client.load_world(random_map)
        carla_map = world.get_map()
        world.wait_for_tick()
        print(f"✅ 随机加载地图成功！当前地图：{carla_map.name}（随机选择：{random_map}）")

        # 2. 获取蓝图和生成点
        blueprint_library = world.get_blueprint_library()
        spawn_points = carla_map.get_spawn_points()
        if not spawn_points:
            print("❌ 无可用生成点，退出")
            return

        # 3. 生成车辆（低画质选轻量化车型）
        vehicle_bps = blueprint_library.filter('vehicle.seat.leon')
        if not vehicle_bps:
            vehicle_bps = blueprint_library.filter('vehicle.*')[0:1]
        vehicle_bp = vehicle_bps[0]
        vehicle_bp.set_attribute('role_name', 'autopilot')

        # 选择生成点
        spawn_idx = random.choice([5, 12, 15, 20]) if len(spawn_points) > 20 else random.randint(0, len(spawn_points)-1)
        spawn_point = spawn_points[spawn_idx]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(False)
        print(f"✅ 生成车辆：{vehicle.type_id} | 生成点：{spawn_point.location}")

        # 4. 生成NPC车辆（可选，用于测试边界框可视化）
        npc_count = 10  # 生成10辆NPC车辆
        spawned_npcs = 0
        for spawn_point in spawn_points:
            if spawned_npcs >= npc_count:
                break
            npc_bps = [bp for bp in blueprint_library.filter('vehicle.*') if int(bp.get_attribute('number_of_wheels')) == 4]
            if not npc_bps:
                continue
            npc_bp = random.choice(npc_bps)
            npc = world.try_spawn_actor(npc_bp, spawn_point)
            if npc:
                npc.set_autopilot(True)
                spawned_npcs += 1
        print(f"✅ 生成{spawned_npcs}辆NPC车辆")

        # 5. 初始化第一个路径点
        current_waypoint = carla_map.get_waypoint(vehicle.get_location(), project_to_road=True)
        current_waypoint = choose_main_waypoint(current_waypoint)
        print(f"✅ 初始化路径点完成")

        # 6. 挂载摄像头（低画质参数）
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '360')
        camera_bp.set_attribute('fov', '80')
        camera_bp.set_attribute('sensor_tick', '0.1')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.8))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        camera.listen(image_callback)
        print("✅ 摄像头挂载成功")

        # 7. 初始化3D转2D投影参数（适配0.9.14）
        image_w = 640
        image_h = 360
        fov = camera_bp.get_attribute('fov').as_float()
        K = build_projection_matrix(image_w, image_h, fov)
        K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)
        # 车辆包围盒边缘（简化版，不影响边界框生成）
        edges = [[0,1],[1,3],[3,2],[2,0],[0,4],[4,5],[5,1],[5,7],[7,6],[6,4],[6,2],[7,3]]
        # COCO类别名称（仅保留车辆）
        COCO_CLASS_NAMES = ['car']

        # 8. 核心逻辑：红灯停+绿灯行+沿道路行驶+3D可视化
        print("\n📌 按 'q' 退出 | 红灯停绿灯行+沿道路行驶+3D车辆边界框可视化")
        cv2.namedWindow('CARLA Low-Quality View (3D Bounding Box)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('CARLA Low-Quality View (3D Bounding Box)', 640, 360)

        # 初始化控制参数
        vehicle_control = carla.VehicleControl()
        TARGET_SPEED = 30  # 目标车速（km/h）
        TARGET_SPEED_MS = TARGET_SPEED / 3.6
        STANDSTILL_THRESHOLD = 0.1
        last_traffic_light_state = None

        while True:
            # ========== 步骤1：获取车辆当前状态 ==========
            current_location = vehicle.get_location()
            current_transform = vehicle.get_transform()
            current_speed = vehicle.get_velocity().length()
            is_standstill = current_speed < STANDSTILL_THRESHOLD
            need_brake = False
            traffic_light_info = "无交通灯"

            # ========== 步骤2：红绿灯识别（原逻辑保留） ==========
            traffic_light = vehicle.get_traffic_light()
            if traffic_light is not None:
                tl_state = traffic_light.state
                traffic_light_info = f"交通灯状态：{tl_state}"
                if tl_state in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
                    need_brake = True
                elif tl_state == carla.TrafficLightState.Green:
                    need_brake = False

            if vehicle.is_at_traffic_light():
                tl_state_alt = vehicle.get_traffic_light_state()
                if tl_state_alt in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
                    need_brake = True
                    traffic_light_info = f"备用检测：{tl_state_alt}"

            # ========== 步骤3：车辆控制逻辑（原逻辑保留） ==========
            if need_brake:
                if traffic_light_info != last_traffic_light_state:
                    print(f"🚦 {traffic_light_info} | 速度：{current_speed:.2f}m/s → 刹车停车")
                    last_traffic_light_state = traffic_light_info

                if not is_standstill:
                    vehicle_control.brake = 1.0
                    vehicle_control.throttle = 0.0
                    vehicle_control.steer = 0.0
                    vehicle_control.hand_brake = False
                    vehicle_control.gear = 1
                else:
                    vehicle_control.brake = 1.0
                    vehicle_control.throttle = 0.0
                    vehicle_control.steer = 0.0
                    vehicle_control.hand_brake = True
                    vehicle_control.gear = 0
            else:
                if last_traffic_light_state is not None:
                    print(f"🚦 {traffic_light_info} | 速度：{current_speed:.2f}m/s → 沿道路行驶")
                    last_traffic_light_state = None

                # 动态更新路径点
                distance_to_waypoint = math.hypot(
                    current_location.x - current_waypoint.transform.location.x,
                    current_location.y - current_waypoint.transform.location.y
                )

                if distance_to_waypoint < 2.0:
                    current_waypoint = choose_main_waypoint(current_waypoint)
                    TARGET_SPEED_MS = (TARGET_SPEED - 5) / 3.6 if current_waypoint.is_junction else TARGET_SPEED / 3.6

                # 转向控制
                angle = calculate_angle(current_transform, current_waypoint.transform.location)
                vehicle_control.steer = np.clip(angle * 0.8, -1.0, 1.0)

                # 速度控制
                vehicle_control.hand_brake = False
                vehicle_control.brake = 0.0
                if current_speed < TARGET_SPEED_MS:
                    vehicle_control.throttle = min(0.7, (TARGET_SPEED_MS - current_speed) / 3 + 0.2)
                else:
                    vehicle_control.throttle = 0.1 if current_speed < TARGET_SPEED_MS + 1 else 0.0
                    vehicle_control.brake = 0.1 if current_speed > TARGET_SPEED_MS + 1 else 0.0
                vehicle_control.gear = 2

            # 应用控制指令
            vehicle.apply_control(vehicle_control)

            # ========== 步骤4：3D车辆转2D边界框可视化（新增核心逻辑） ==========
            # 获取摄像头图像
            current_img = None
            if not IMAGE_QUEUE.empty():
                try:
                    current_img = IMAGE_QUEUE.get(timeout=0.5)
                except queue.Empty:
                    pass
            # 容错：使用最新图像或黑色占位图
            if current_img is None and LATEST_IMAGE is not None:
                current_img = LATEST_IMAGE.copy()
            if current_img is None:
                current_img = np.zeros((image_h, image_w, 3), dtype=np.uint8)

            # 获取世界→相机的变换矩阵
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # 遍历所有车辆，计算2D边界框
            boxes = []
            ids = []
            for npc in world.get_actors().filter('*vehicle*'):
                # 跳过自车
                if npc.id == vehicle.id:
                    continue

                # 筛选50米内、自车前方的车辆
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - vehicle.get_transform().location
                if dist < 50 and forward_vec.dot(ray) > 0:
                    # 获取车辆包围盒的3D顶点（适配0.9.14）
                    bb_verts = [v for v in npc.bounding_box.get_world_vertices(npc.get_transform())]
                    points_2d = []
                    # 将每个3D顶点投影到2D图像
                    for vert in bb_verts:
                        ray_cam = vert - camera.get_transform().location
                        cam_forward = camera.get_transform().get_forward_vector()
                        if cam_forward.dot(ray_cam) > 0:
                            p = get_image_point(vert, K, world_2_camera)
                        else:
                            p = get_image_point(vert, K_b, world_2_camera)
                        points_2d.append(p)
                    # 生成2D边界框
                    if points_2d:
                        x1, y1, x2, y2 = get_2d_box_from_3d_edges(points_2d, image_h, image_w)
                        # 过滤过小的边界框
                        if (y2 - y1) * (x2 - x1) > 100 and (x2 - x1) > 20:
                            if point_in_canvas((x1, y1), image_h, image_w) and point_in_canvas((x2, y2), image_h, image_w):
                                boxes.append(np.array([x1, y1, x2, y2]))
                                ids.append(npc.id)

            # 绘制边界框
            if boxes:
                boxes = np.array(boxes)
                output_image = draw_bounding_boxes(current_img, boxes, ids, COCO_CLASS_NAMES)
            else:
                output_image = current_img

            # ========== 步骤5：显示图像 ==========
            cv2.imshow('CARLA Low-Quality View (3D Bounding Box)', output_image)

            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 异常处理
    except Exception as e:
        if "Connection" in str(e):
            print("\n❌ 连接失败！")
            print("   解决：1. 确认CarlaUE4.exe已启动 2. 关闭防火墙 3. 检查端口2000")
        elif "Spawn" in str(e):
            print("\n❌ 车辆生成失败！")
            print("   解决：换生成点或重启CARLA")
        elif "AttributeError" in str(e):
            print(f"\n❌ API属性错误：{e}")
            print("   解决：确认CARLA版本为0.9.14，重新安装对应版本的whl包")
        else:
            print(f"\n❌ 未知错误：{e}")
            import traceback
            traceback.print_exc()

    # 清理资源
    finally:
        print("\n--- [清理资源] ---")
        # 销毁所有车辆（包括NPC）
        if world:
            for actor in world.get_actors().filter('*vehicle*'):
                try:
                    actor.destroy()
                except:
                    pass
        if camera:
            try:
                camera.stop()
                camera.destroy()
                print("✅ 销毁摄像头")
            except:
                print("⚠️ 摄像头销毁失败")
        if vehicle:
            try:
                vehicle.destroy()
                print("✅ 销毁车辆")
            except:
                print("⚠️ 车辆销毁失败")
        cv2.destroyAllWindows()
        print("✅ 程序结束")

if __name__ == '__main__':
    # 导入psutil
    try:
        import psutil
    except ImportError:
        print("⚠️ 未安装psutil，跳过CARLA进程检查")
        def check_carla_running():
            return True
    main()