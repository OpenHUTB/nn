import os
import sys
import traceback
import queue
import random
import time
import numpy as np
import cv2
import carla
from collections import deque

# ====================== 关键：添加src/2d-carla-tracking-master到Python搜索路径 ======================
CARLA_TRACKING_ROOT = r"D:\nn\src\2d-carla-tracking-master"
sys.path.append(CARLA_TRACKING_ROOT)

# ====================== 导入所需模块 ======================
try:
    from utils.projection import build_projection_matrix, get_image_point, point_in_canvas
    from utils.world import clear_npc, clear_static_vehicle, clear
except ImportError as e:
    print(f"导入模块失败：{e}")
    sys.exit(1)

# ====================== 全局配置变量 ======================
# Carla连接配置
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0
SYNC_DELTA_SECONDS = 0.05  # 20FPS

# 摄像头配置 - 减小视图尺寸
CAMERA_WIDTH = 960  # 减小分辨率
CAMERA_HEIGHT = 540
CAMERA_FOV = 90
CAMERA_POSITION = carla.Transform(carla.Location(x=1, z=2))

# NPC车辆配置
NPC_VEHICLE_NUM = 25

# 3D边界框配置
EDGES = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
         [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
DISTANCE_THRESHOLD = 80  # 减小显示距离

# 红绿灯配置
SHOW_TRAFFIC_LIGHTS = True  # 是否显示红绿灯
TRAFFIC_LIGHT_DISTANCE = 60  # 减小红绿灯显示距离

# 显示控制
SHOW_INFO_PANEL = True  # 是否显示信息面板
SHOW_VEHICLES = True  # 是否显示车辆
SHOW_TRAFFIC_LIGHTS_STATE = True  # 是否显示红绿灯状态文字

# ====================== 颜色定义 ======================
# 车辆边界框颜色
VEHICLE_COLOR = (0, 255, 0)  # 绿色（BGR格式）

# 红绿灯状态颜色（BGR格式）
TRAFFIC_LIGHT_COLORS = {
    0: (0, 255, 0),  # 绿色
    1: (0, 255, 255),  # 黄色
    2: (0, 0, 255),  # 红色
    3: (255, 255, 255)  # 白色
}

TRAFFIC_LIGHT_STATE_NAMES = {
    0: "GREEN",
    1: "YELLOW",
    2: "RED",
    3: "UNKNOWN"
}

# 信息面板颜色
PANEL_BG_COLOR = (40, 40, 40)  # 深灰色背景
PANEL_BORDER_COLOR = (0, 200, 0)  # 浅绿色边框
TEXT_COLOR = (240, 240, 240)  # 浅灰色文字
HIGHLIGHT_COLOR = (0, 255, 255)  # 黄色高亮文字


# ====================== 性能监控器 ======================
class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.frame_times = deque(maxlen=window_size)

    def start_frame(self):
        self.frame_start = time.time()

    def end_frame(self):
        current_time = time.time()
        frame_time = (current_time - self.frame_start) * 1000

        self.timestamps.append(current_time)
        self.frame_times.append(frame_time)

        if len(self.timestamps) > 1:
            fps = len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0])
            avg_frame_time = np.mean(self.frame_times) if self.frame_times else 0
            return fps, avg_frame_time, frame_time
        return 0, 0, 0


# ====================== Carla环境初始化 ======================
def init_carla_environment():
    """初始化Carla环境"""
    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(CARLA_TIMEOUT)

        # 连接
        world = client.get_world()
        print("✅ Carla模拟器连接成功！")

        # 设置同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = SYNC_DELTA_SECONDS
        world.apply_settings(settings)
        print(f"✅ 同步模式已启用，帧率: {1 / SYNC_DELTA_SECONDS:.1f} FPS")

        # 获取蓝图和生成点
        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        if not spawn_points:
            print("⚠️ 警告：未找到生成点")
            spawn_points = [carla.Transform()]

        # 生成主车辆
        print("🚗 生成主车辆...")
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
        vehicle = None
        for _ in range(10):
            vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            if vehicle:
                break

        if not vehicle:
            print("⚠️ 警告：使用默认车辆")
            vehicle_bp = random.choice(list(bp_lib.filter('vehicle.*')))
            vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))

        print("✅ 主车辆生成成功")

        # 生成摄像头
        print("📷 生成摄像头...")
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(CAMERA_FOV))

        camera = world.spawn_actor(camera_bp, CAMERA_POSITION, attach_to=vehicle)
        print(f"✅ 摄像头生成成功: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

        # 图像队列
        image_queue = queue.Queue(maxsize=1)

        def camera_callback(image):
            try:
                img = np.reshape(np.copy(image.raw_data),
                                 (image.height, image.width, 4))
                img = img[:, :, :3].astype(np.uint8)

                # 如果尺寸不匹配，调整尺寸
                if CAMERA_WIDTH != image.width or CAMERA_HEIGHT != image.height:
                    img = cv2.resize(img, (CAMERA_WIDTH, CAMERA_HEIGHT))

                if not image_queue.full():
                    image_queue.put_nowait(img)
            except Exception as e:
                pass

        camera.listen(camera_callback)

        # 清理现有NPC
        print("🧹 清理现有NPC...")
        try:
            clear_npc(world)
            clear_static_vehicle(world)
            print("✅ 现有NPC已清理")
        except:
            print("⚠️ 清理NPC失败（可能已清理）")

        # 生成NPC车辆
        print(f"🚗 生成 {NPC_VEHICLE_NUM} 辆NPC车辆...")
        car_bps = []

        # 获取四轮车辆蓝图
        for bp in bp_lib.filter('vehicle.*'):
            try:
                wheels = bp.get_attribute('number_of_wheels')
                if wheels and int(wheels.as_int()) == 4:
                    car_bps.append(bp)
            except:
                car_bps.append(bp)

        if not car_bps:
            car_bps = list(bp_lib.filter('vehicle.*'))

        spawned = 0
        max_attempts = min(NPC_VEHICLE_NUM * 3, len(spawn_points) * 2)

        for attempt in range(max_attempts):
            if spawned >= NPC_VEHICLE_NUM:
                break

            bp = random.choice(car_bps)
            spawn_point = random.choice(spawn_points)
            npc = world.try_spawn_actor(bp, spawn_point)

            if npc:
                try:
                    npc.set_autopilot(True)
                    spawned += 1
                    if spawned % 10 == 0:
                        print(f"  已生成 {spawned} 辆NPC车辆")
                except:
                    npc.destroy()

        print(f"✅ 成功生成 {spawned} 辆NPC车辆")
        vehicle.set_autopilot(True)
        print("✅ 主车辆自动驾驶已开启")

        return world, client, vehicle, camera, image_queue

    except Exception as e:
        print(f"❌ Carla初始化失败: {e}")
        traceback.print_exc()
        raise


# ====================== 3D边界框和红绿灯绘制函数 ======================
def draw_3d_bounding_boxes(image, world, camera, vehicle):
    """在图像上绘制3D真值边界框和红绿灯"""
    try:
        img = image.copy()
        height, width = img.shape[:2]

        # 构建投影矩阵
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        K = build_projection_matrix(width, height, CAMERA_FOV)
        K_b = build_projection_matrix(width, height, CAMERA_FOV, is_behind_camera=True)

        vehicle_count = 0
        traffic_light_count = 0

        # 绘制车辆
        if SHOW_VEHICLES:
            vehicles = list(world.get_actors().filter('*vehicle*'))

            for npc in vehicles:
                if npc.id == vehicle.id:
                    continue

                # 计算距离
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                if dist >= DISTANCE_THRESHOLD:
                    continue

                # 检查是否在相机前方
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - vehicle.get_transform().location

                if forward_vec.dot(ray) <= 0:
                    continue

                # 获取边界框顶点
                bb = npc.bounding_box
                verts = bb.get_world_vertices(npc.get_transform())

                # 投影到2D
                points_2d = []
                for vert in verts:
                    ray0 = vert - camera.get_transform().location
                    cam_forward_vec = camera.get_transform().get_forward_vector()

                    if cam_forward_vec.dot(ray0) > 0:
                        p = get_image_point(vert, K, world_2_camera)
                    else:
                        p = get_image_point(vert, K_b, world_2_camera)

                    points_2d.append(p)

                # 绘制3D边界框
                for edge in EDGES:
                    p1 = points_2d[edge[0]]
                    p2 = points_2d[edge[1]]

                    if point_in_canvas(p1, height, width) or point_in_canvas(p2, height, width):
                        thickness = max(1, int(2 - dist / 50))
                        color_intensity = max(50, int(255 - dist))
                        color = (0, color_intensity, 0)

                        cv2.line(img, (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])), color, thickness)

                vehicle_count += 1

        # 绘制红绿灯
        if SHOW_TRAFFIC_LIGHTS:
            traffic_lights = list(world.get_actors().filter('*traffic_light*'))

            for light in traffic_lights:
                # 计算距离
                dist = light.get_transform().location.distance(vehicle.get_transform().location)
                if dist >= TRAFFIC_LIGHT_DISTANCE:
                    continue

                # 检查是否在相机前方
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = light.get_transform().location - vehicle.get_transform().location

                if forward_vec.dot(ray) <= 0:
                    continue

                # 获取红绿灯位置
                location = light.get_transform().location

                # 投影到2D
                ray0 = location - camera.get_transform().location
                cam_forward_vec = camera.get_transform().get_forward_vector()

                if cam_forward_vec.dot(ray0) > 0:
                    point_2d = get_image_point(location, K, world_2_camera)
                else:
                    point_2d = get_image_point(location, K_b, world_2_camera)

                # 检查点是否在画布内
                if not point_in_canvas(point_2d, height, width):
                    continue

                x, y = int(point_2d[0]), int(point_2d[1])

                # 获取红绿灯状态
                light_state = light.get_state()
                state_mapping = {
                    carla.TrafficLightState.Green: 0,  # 绿色
                    carla.TrafficLightState.Yellow: 1,  # 黄色
                    carla.TrafficLightState.Red: 2,  # 红色
                }
                state_idx = state_mapping.get(light_state, 3)  # 默认白色
                light_color = TRAFFIC_LIGHT_COLORS[state_idx]

                # 绘制红绿灯
                radius = max(6, int(15 - dist / 20))
                cv2.circle(img, (x, y), radius, light_color, -1)
                cv2.circle(img, (x, y), radius, (255, 255, 255), 1)

                # 添加文字标签
                if SHOW_TRAFFIC_LIGHTS_STATE and radius > 8:
                    state_name = TRAFFIC_LIGHT_STATE_NAMES[state_idx]
                    text_size = cv2.getTextSize(state_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_x = x - text_size[0] // 2
                    text_y = y - radius - 5

                    # 文字背景
                    cv2.rectangle(img, (text_x - 3, text_y - text_size[1] - 3),
                                  (text_x + text_size[0] + 3, text_y + 3),
                                  (40, 40, 40), -1)

                    # 文字
                    cv2.putText(img, state_name, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                traffic_light_count += 1

        return img, vehicle_count, traffic_light_count

    except Exception as e:
        print(f"❌ 3D边界框绘制错误: {e}")
        return image, 0, 0


# ====================== 绘制清晰的信息面板（修复乱码） ======================
def draw_info_panel(image, fps, avg_frame_time, frame_count, vehicle_count, traffic_light_count):
    """在图像上绘制清晰的信息面板"""
    try:
        img = image.copy()

        # 面板尺寸和位置（根据图像大小调整）
        panel_width = 320
        panel_height = 180
        panel_x = 10
        panel_y = 10

        # 确保面板不会超出图像边界
        if panel_x + panel_width > img.shape[1]:
            panel_x = img.shape[1] - panel_width - 10
        if panel_y + panel_height > img.shape[0]:
            panel_y = img.shape[0] - panel_height - 10

        # 创建半透明背景
        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      PANEL_BG_COLOR, -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # 绘制边框
        cv2.rectangle(img, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      PANEL_BORDER_COLOR, 2)

        # 标题 - 使用英文避免乱码
        title = "CARLA 3D VISUALIZATION"
        cv2.putText(img, title, (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, HIGHLIGHT_COLOR, 2)

        # 分隔线
        line_y = panel_y + 45
        cv2.line(img, (panel_x + 10, line_y), (panel_x + panel_width - 10, line_y),
                 (100, 200, 100), 1)

        # 信息区域
        info_start_y = line_y + 10
        line_spacing = 25

        # 第1行：FPS和帧数
        row1_y = info_start_y
        fps_text = f"FPS: {fps:.1f}"
        frame_text = f"Frame: {frame_count}"

        cv2.putText(img, fps_text, (panel_x + 15, row1_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
        cv2.putText(img, frame_text, (panel_x + 150, row1_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

        # 第2行：帧时间
        row2_y = row1_y + line_spacing
        frame_time_text = f"Frame Time: {avg_frame_time:.1f}ms"
        cv2.putText(img, frame_time_text, (panel_x + 15, row2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

        # 第3行：车辆和红绿灯数量
        row3_y = row2_y + line_spacing
        vehicles_text = f"Vehicles: {vehicle_count}"
        lights_text = f"Lights: {traffic_light_count}"

        cv2.putText(img, vehicles_text, (panel_x + 15, row3_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
        cv2.putText(img, lights_text, (panel_x + 150, row3_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

        # 第4行：显示状态
        row4_y = row3_y + line_spacing
        status_text = f"Display: V{'ON' if SHOW_VEHICLES else 'OFF'} T{'ON' if SHOW_TRAFFIC_LIGHTS else 'OFF'}"
        cv2.putText(img, status_text, (panel_x + 15, row4_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)

        # 操作提示（最后一行）
        hint_y = panel_y + panel_height - 10
        hint_text = "V:Vehicles T:Traffic I:Info Q:Quit"
        cv2.putText(img, hint_text, (panel_x + 10, hint_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 250, 150), 1)

        return img

    except Exception as e:
        print(f"❌ 信息面板绘制错误: {e}")
        return image


# ====================== 主函数 ======================
def main():
    """主函数"""
    # 声明全局变量
    global SHOW_INFO_PANEL, SHOW_VEHICLES, SHOW_TRAFFIC_LIGHTS, SHOW_TRAFFIC_LIGHTS_STATE

    world = None
    camera = None
    perf_monitor = None

    try:
        print("=" * 60)
        print("CARLA 3D Visualization System")
        print("=" * 60)
        print(f"Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        print(f"View Distance: {DISTANCE_THRESHOLD}m")
        print(f"NPC Vehicles: {NPC_VEHICLE_NUM}")
        print("=" * 60)

        # 1. 初始化Carla
        print("Initializing Carla environment...")
        world, client, vehicle, camera, image_queue = init_carla_environment()

        # 2. 性能监控
        perf_monitor = PerformanceMonitor()

        # 3. 获取spectator
        spectator = world.get_spectator()

        # 4. 主循环
        print("\nStarting main loop...")
        print("Controls:")
        print("  V: Toggle vehicles display")
        print("  T: Toggle traffic lights display")
        print("  I: Toggle info panel")
        print("  S: Save screenshot")
        print("  R: Reset statistics")
        print("  Q/ESC: Quit program")

        frame_count = 0

        # 创建可调整大小的窗口
        window_title = "CARLA 3D Visualization"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, CAMERA_WIDTH, CAMERA_HEIGHT)

        # 设置窗口位置（可选）
        cv2.moveWindow(window_title, 100, 100)

        while True:
            # 性能监控
            perf_monitor.start_frame()

            # 同步Carla世界
            world.tick()

            # 更新观察者视角
            try:
                vehicle_transform = vehicle.get_transform()
                spectator_transform = carla.Transform(
                    vehicle_transform.transform(carla.Location(x=-6, z=50)),  # 更近的视角
                    carla.Rotation(yaw=-180, pitch=-75)
                )
                spectator.set_transform(spectator_transform)
            except:
                pass

            # 获取图像
            if image_queue.empty():
                time.sleep(0.001)
                continue

            origin_image = image_queue.get()
            frame_count += 1

            # 绘制3D边界框和红绿灯
            result_image, vehicle_count, traffic_light_count = draw_3d_bounding_boxes(
                origin_image, world, camera, vehicle
            )

            # 获取性能数据
            fps, avg_frame_time, current_frame_time = perf_monitor.end_frame()

            # 绘制信息面板
            if SHOW_INFO_PANEL:
                result_image = draw_info_panel(
                    result_image, fps, avg_frame_time, frame_count,
                    vehicle_count, traffic_light_count
                )

            # 显示图像
            cv2.imshow(window_title, result_image)

            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Quitting...")
                break
            elif key == ord('v'):
                SHOW_VEHICLES = not SHOW_VEHICLES
                status = "ON" if SHOW_VEHICLES else "OFF"
                print(f"Vehicles display: {status}")
            elif key == ord('t'):
                SHOW_TRAFFIC_LIGHTS = not SHOW_TRAFFIC_LIGHTS
                status = "ON" if SHOW_TRAFFIC_LIGHTS else "OFF"
                print(f"Traffic lights display: {status}")
            elif key == ord('i'):
                SHOW_INFO_PANEL = not SHOW_INFO_PANEL
                status = "ON" if SHOW_INFO_PANEL else "OFF"
                print(f"Info panel: {status}")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"carla_{timestamp}.png"
                cv2.imwrite(filename, result_image)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                frame_count = 0
                perf_monitor = PerformanceMonitor()
                print("Statistics reset")

    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        print("\nCleaning up resources...")

        if camera is not None:
            try:
                camera.stop()
                camera.destroy()
                print("Camera destroyed")
            except:
                pass

        if world is not None:
            try:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
                print("Carla sync mode disabled")
            except:
                pass

        cv2.destroyAllWindows()

        # 清理车辆
        try:
            if vehicle is not None:
                vehicle.destroy()
                print("Ego vehicle destroyed")

            # 清理NPC
            npc_count = 0
            for actor in world.get_actors().filter('vehicle.*'):
                if actor.is_alive:
                    try:
                        actor.destroy()
                        npc_count += 1
                    except:
                        pass
            print(f"NPC vehicles destroyed: {npc_count}")
        except:
            pass

        print("Program exited")


if __name__ == '__main__':
    main()