import carla
import queue
import random
import cv2
import numpy as np

# 注意：移除了所有deep_sort相关导入
# 保留必要的工具函数导入（如果utils模块不存在，我会替换为内置实现）
from utils.projection import build_projection_matrix, get_image_point, get_2d_box_from_3d_edges, point_in_canvas
from utils.world import clear_npc, clear_static_vehicle

# ---------------------------
# 全局配置（可根据需要调整）
# ---------------------------
MAP_NAME = "Town07"  # 指定要加载的地图，可替换为Town01-Town12
NPC_VEHICLE_NUM = 50  # NPC车辆数量
CAMERA_RESOLUTION = (640, 640)  # 相机分辨率
SYNC_DELTA_SECONDS = 0.05  # 同步模式的固定时间步长
VEHICLE_DISTANCE_THRESHOLD = 50  # 检测车辆的距离阈值（米）
SPEED_DISPLAY_UNIT = "km/h"  # 速度单位：m/s 或 km/h

# ---------------------------
# 替代draw_bounding_boxes的内置函数（如果utils.box_utils不存在）
# 如果你有utils.box_utils，可注释这部分，保留原导入
# ---------------------------
def draw_bounding_boxes(image, bboxes, labels, class_names, ids):
    """
    绘制包围盒和ID标签
    :param image: 输入图像（BGR格式）
    :param bboxes: 包围盒列表，格式为[[x1,y1,x2,y2], ...]
    :param labels: 类别标签列表
    :param class_names: 类别名称列表
    :param ids: 跟踪ID列表
    :return: 绘制后的图像
    """
    for i, (bbox, id) in enumerate(zip(bboxes, ids)):
        x1, y1, x2, y2 = bbox.astype(int)
        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制标签（类别 + ID）
        label = f"{class_names[labels[i]]} | ID: {id}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def camera_callback(image, rgb_image_queue):
    """相机回调函数，将图像数据存入队列"""
    rgb_image_queue.put(np.reshape(np.copy(image.raw_data),
                        (image.height, image.width, 4)))

def calculate_vehicle_speed(velocity, unit="km/h"):
    """计算车辆速度，转换为指定单位"""
    speed_m_s = np.linalg.norm([velocity.x, velocity.y, velocity.z])
    if unit == "km/h":
        return speed_m_s * 3.6  # 1 m/s = 3.6 km/h
    return speed_m_s

def main():
    # ---------------------------
    # 1. 初始化Carla客户端，加快地图加载
    # ---------------------------
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(30.0)  # 增加超时时间，避免地图加载时超时

        # 方案3：查看所有可用地图
        print("Carla支持的地图列表：")
        available_maps = client.get_available_maps()
        print(available_maps)

        # 方案2：使用当前已加载的地图，避免map not found
        world = client.get_world()
        # 如果想指定地图，可从available_maps中选一个，比如：
        # world = client.load_world("Town01")

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("警告：未获取到地图生成点，程序将退出")
            return
    # 后续代码不变...

        # ---------------------------
        # 2. 设置同步模式，优化仿真效率
        # ---------------------------
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = SYNC_DELTA_SECONDS
        world.apply_settings(settings)

        # ---------------------------
        # 3. 生成Ego车辆
        # ---------------------------
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
        ego_vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        if not ego_vehicle:
            print("警告：未能生成Ego车辆，程序将退出")
            return

        # ---------------------------
        # 4. 生成相机传感器
        # ---------------------------
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(CAMERA_RESOLUTION[0]))
        camera_bp.set_attribute('image_size_y', str(CAMERA_RESOLUTION[1]))
        camera_init_trans = carla.Transform(carla.Location(x=1, z=2))
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

        # 图像队列（设置最大长度，避免内存溢出）
        image_queue = queue.Queue(maxsize=1)
        camera.listen(lambda image: camera_callback(image, image_queue))

        # ---------------------------
        # 5. 清理现有NPC和静态车辆
        # ---------------------------
        clear_npc(world)
        clear_static_vehicle(world)

        # ---------------------------
        # 6. 预计算相机相关参数
        # ---------------------------
        edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
                 [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
        image_w, image_h = CAMERA_RESOLUTION
        fov = camera_bp.get_attribute("fov").as_float()
        K = build_projection_matrix(image_w, image_h, fov)
        K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

        # ---------------------------
        # 7. 生成NPC车辆
        # ---------------------------
        car_bp = [bp for bp in bp_lib.filter('vehicle') if int(bp.get_attribute('number_of_wheels')) == 4]
        if car_bp:
            for i in range(NPC_VEHICLE_NUM):
                npc = world.try_spawn_actor(random.choice(car_bp), random.choice(spawn_points))
                if npc:
                    npc.set_autopilot(True)

        ego_vehicle.set_autopilot(True)

        # ---------------------------
        # 8. 主循环（核心：用Actor ID实现跟踪）
        # ---------------------------
        while True:
            world.tick()

            # 移动spectator到车辆上方
            spectator = world.get_spectator()
            transform = carla.Transform(ego_vehicle.get_transform().transform(
                carla.Location(x=-4, z=50)), carla.Rotation(yaw=-180, pitch=-90))
            spectator.set_transform(transform)

            # 获取相机图像
            if not image_queue.empty():
                image = image_queue.get()
            else:
                continue

            # 图像预处理（RGBA转BGR，减少一次颜色转换）
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

            # 更新相机的世界到相机矩阵
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # 存储：包围盒、车辆ID、速度
            bboxes = []
            vehicle_ids = []
            vehicle_speeds = []

            # 遍历所有车辆Actor
            vehicles = world.get_actors().filter('*vehicle*')
            for npc in vehicles:
                if npc.id == ego_vehicle.id:
                    # 输出Ego车辆速度
                    ego_speed = calculate_vehicle_speed(npc.get_velocity(), SPEED_DISPLAY_UNIT)
                    print(f"Ego车辆速度：{ego_speed:.1f} {SPEED_DISPLAY_UNIT}")
                    continue

                # 过滤距离过远的车辆
                dist = npc.get_transform().location.distance(ego_vehicle.get_transform().location)
                if dist > VEHICLE_DISTANCE_THRESHOLD:
                    continue

                # 只检测车辆前方的目标
                forward_vec = ego_vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - ego_vehicle.get_transform().location
                if forward_vec.dot(ray) <= 0:
                    continue

                # 计算NPC车辆速度
                npc_speed = calculate_vehicle_speed(npc.get_velocity(), SPEED_DISPLAY_UNIT)

                # 计算3D包围盒的2D投影
                bb = npc.bounding_box
                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                points_2d = []

                for vert in verts:
                    ray0 = vert - camera.get_transform().location
                    cam_forward_vec = camera.get_transform().get_forward_vector()
                    if cam_forward_vec.dot(ray0) > 0:
                        p = get_image_point(vert, K, world_2_camera)
                    else:
                        p = get_image_point(vert, K_b, world_2_camera)
                    points_2d.append(p)

                # 获取2D框
                x_min, x_max, y_min, y_max = get_2d_box_from_3d_edges(points_2d, edges, image_h, image_w)

                # 过滤过小的框和超出画布的框
                box_area = (y_max - y_min) * (x_max - x_min)
                box_width = x_max - x_min
                if box_area > 100 and box_width > 20 and point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):
                    bboxes.append(np.array([x_min, y_min, x_max, y_max]))
                    vehicle_ids.append(npc.id)  # 用Actor ID作为跟踪ID
                    vehicle_speeds.append(npc_speed)

            # ---------------------------
            # 绘制包围盒、ID和速度（核心替代deep_sort的部分）
            # ---------------------------
            if len(bboxes) > 0:
                # 绘制基础包围盒和ID（类别固定为car，对应COCO的2号类别）
                labels = np.array([2] * len(bboxes))
                COCO_CLASS_NAMES = {2: 'car'}  # 简化的类别映射，替代原导入
                image = draw_bounding_boxes(image, bboxes, labels, COCO_CLASS_NAMES, vehicle_ids)

                # 绘制速度标签
                for i, (bbox, speed) in enumerate(zip(bboxes, vehicle_speeds)):
                    x1, y1, x2, y2 = bbox.astype(int)
                    speed_label = f"Speed: {speed:.1f} {SPEED_DISPLAY_UNIT}"
                    # 在框下方绘制速度
                    cv2.putText(image, speed_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 显示图像
            cv2.imshow('2D Tracking with Actor ID (no deep_sort)', image)

            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序出错：{e}")
    finally:
        # ---------------------------
        # 资源清理
        # ---------------------------
        print("正在清理资源...")
        if 'world' in locals():
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        if 'camera' in locals():
            camera.destroy()
        if 'ego_vehicle' in locals():
            ego_vehicle.destroy()
        # 清理所有NPC车辆
        if 'world' in locals():
            for vehicle in world.get_actors().filter('*vehicle*'):
                vehicle.destroy()
        cv2.destroyAllWindows()
        print("资源清理完成")

if __name__ == "__main__":
    main()