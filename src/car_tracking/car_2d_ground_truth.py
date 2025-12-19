import carla
import queue
import random
import cv2
import numpy as np
import math
import colorsys

# ===================== 核心工具函数（整合原utils的核心功能，无需外部依赖） =====================
# 1. 绘制边界框（原utils.box_utils.draw_bounding_boxes）
def draw_bounding_boxes(image, boxes, labels, class_names, ids=None, scores=None):
    """在图像上绘制边界框，包含类别、ID、置信度"""
    num_classes = len(class_names)
    # 为每个类别生成唯一颜色
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), colors))

    image_copy = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        label = labels[i]
        class_name = class_names[label] if label < len(class_names) else 'unknown'
        color = colors[label % num_classes]

        # 绘制矩形框
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        # 准备文本（类别+ID）
        text = f"{class_name} (ID: {ids[i]})" if ids and i < len(ids) else class_name
        if scores and i < len(scores):
            text += f" {scores[i]:.2f}"

        # 绘制文本背景和文字
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10
        cv2.rectangle(image_copy, (x1, text_y - text_size[1] - 2),
                      (x1 + text_size[0], text_y + 2), color, -1)
        cv2.putText(image_copy, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    return image_copy

# 2. 构建相机投影矩阵（原utils.projection.build_projection_matrix）
def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """将3D世界坐标投影到2D图像的矩阵"""
    focal = w / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    if is_behind_camera:
        K[2, 2] = -1  # 反转z轴处理相机后方的点
    return K

# 3. 3D点转2D图像点（原utils.projection.get_image_point）
def get_image_point(loc, K, w2c):
    """将Carla的3D位置转换为2D图像坐标"""
    point = np.array([loc.x, loc.y, loc.z, 1.0])
    point_camera = np.dot(w2c, point)  # 世界→相机
    point_img = np.dot(K, point_camera[:3])  # 相机→图像
    point_img = point_img / point_img[2]  # 归一化
    return (point_img[0], point_img[1])

# 4. 从3D边缘生成2D边界框（原utils.projection.get_2d_box_from_3d_edges）
def get_2d_box_from_3d_edges(points_2d, edges, h, w):
    """从3D点的2D投影生成最小包围框"""
    x_coords = [p[0] for p in points_2d]
    y_coords = [p[1] for p in points_2d]
    x_min = max(0, min(x_coords))
    x_max = min(w, max(x_coords))
    y_min = max(0, min(y_coords))
    y_max = min(h, max(y_coords))
    return x_min, x_max, y_min, y_max

# 5. 判断点是否在图像内（原utils.projection.point_in_canvas）
def point_in_canvas(point, h, w):
    """检查2D点是否在图像画布范围内"""
    x, y = point
    return 0 <= x < w and 0 <= y < h

# 6. 清理Carla中的车辆（原utils.world.clear_npc/clear_static_vehicle/clear）
def clear_actors(world, camera=None):
    """销毁所有车辆和相机传感器"""
    if camera:
        camera.destroy()
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()

# ===================== COCO类别名称（对应车辆类别） =====================
# 无需依赖what库，直接定义核心类别
COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light'
]

# ===================== 相机回调函数 =====================
def camera_callback(image, rgb_image_queue):
    """将相机图像存入队列"""
    rgb_image_queue.put(np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)))

# ===================== 主程序 =====================
if __name__ == '__main__':
    # 1. 连接Carla客户端
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    # 2. 设置Carla同步模式（稳定控制帧率）
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    # 3. 获取核心资源：蓝图库、出生点、观察者
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    spectator = world.get_spectator()

    # 4. 生成自车（林肯MKZ）
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    # 防止出生点冲突，重试生成
    while not vehicle:
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

    # 5. 生成相机传感器（挂载在自车前方）
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '640')
    camera_transform = carla.Transform(carla.Location(x=1.0, z=2.0))  # 前视相机
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # 6. 初始化图像队列，监听相机数据
    image_queue = queue.Queue()
    camera.listen(lambda img: camera_callback(img, image_queue))

    # 7. 清理原有车辆，生成50辆NPC车辆（开启自动驾驶）
    clear_actors(world)
    for i in range(50):
        # 筛选4轮车辆，排除自行车
        vehicle_bps = [bp for bp in bp_lib.filter('vehicle') if int(bp.get_attribute('number_of_wheels')) == 4]
        npc = world.try_spawn_actor(random.choice(vehicle_bps), random.choice(spawn_points))
        if npc:
            npc.set_autopilot(True)
    vehicle.set_autopilot(True)  # 自车也开启自动驾驶

    # 8. 初始化3D→2D投影的参数
    edges = [[0,1],[1,3],[3,2],[2,0],[0,4],[4,5],[5,1],[5,7],[7,6],[6,4],[6,2],[7,3]]  # 车辆包围盒边缘
    image_w = 640
    image_h = 640
    fov = camera_bp.get_attribute('fov').as_float()
    K = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

    # ===================== 主循环：实时可视化地面真值 =====================
    print("程序已启动，按q键退出...")
    try:
        while True:
            # 推进Carla仿真一帧
            world.tick()

            # 移动观察者到自车上方（俯视视角）
            spectator_transform = carla.Transform(
                vehicle.get_transform().transform(carla.Location(x=-4, z=50)),
                carla.Rotation(yaw=-180, pitch=-90)
            )
            spectator.set_transform(spectator_transform)

            # 获取相机图像
            image = image_queue.get()
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
                    # 获取车辆包围盒的3D顶点
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
                    x_min, x_max, y_min, y_max = get_2d_box_from_3d_edges(points_2d, edges, image_h, image_w)
                    # 过滤过小的边界框
                    if (y_max - y_min)*(x_max - x_min) > 100 and (x_max - x_min) > 20:
                        if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):
                            boxes.append(np.array([x_min, y_min, x_max, y_max]))
                            ids.append(npc.id)

            # 绘制边界框（标签2对应COCO的car类别）
            boxes = np.array(boxes)
            labels = np.array([2] * len(boxes))  # 2 = car（对应COCO_CLASS_NAMES[2]）
            output_image = draw_bounding_boxes(image, boxes, labels, COCO_CLASS_NAMES, ids)

            # 显示图像
            cv2.imshow('Carla 2D Ground Truth (Vehicles)', output_image)
            # 按q键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        # 清理资源
        clear_actors(world, camera)
        cv2.destroyAllWindows()
        print("程序已退出，资源已清理")