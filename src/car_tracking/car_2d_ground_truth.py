# 导入所需库
import carla  # Carla仿真平台的Python API
import queue  # 队列，用于线程安全地存储相机图像数据
import random  # 随机数，用于随机选择出生点和车辆蓝图
import cv2  # OpenCV库，用于图像处理和显示
import numpy as np  # 数值计算库，处理矩阵和数组
import math  # 数学库，用于三角函数、矩阵计算等
import colorsys  # 颜色空间转换库，生成类别的唯一颜色

# ===================== 核心工具函数（整合原utils的核心功能，无需外部依赖） =====================
# 1. 绘制边界框（原utils.box_utils.draw_bounding_boxes）
def draw_bounding_boxes(image, boxes, labels, class_names, ids=None, scores=None):
    """
    在图像上绘制边界框，包含类别名称、实例ID、置信度（可选）
    参数说明：
        image: 输入的原始图像（numpy数组）
        boxes: 边界框坐标数组，形状为(N,4)，每个元素是[x1,y1,x2,y2]
        labels: 每个边界框对应的类别标签数组，形状为(N,)
        class_names: 类别名称列表，标签对应列表的索引
        ids: 可选，每个边界框对应的实例ID数组（如车辆ID）
        scores: 可选，每个边界框对应的置信度分数数组
    返回：
        image_copy: 绘制了边界框的图像副本
    """
    # 获取类别总数，用于生成唯一颜色
    num_classes = len(class_names)
    # 为每个类别生成HSV颜色（色调均匀分布，饱和度和亮度为1）
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    # 将HSV颜色转换为RGB颜色
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    # 将RGB值转换为0-255的整数（符合OpenCV的颜色格式）
    colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), colors))

    # 复制原始图像，避免直接修改原图像
    image_copy = image.copy()
    # 遍历每个边界框
    for i, box in enumerate(boxes):
        # 将边界框坐标转换为整数（图像像素坐标为整数）
        x1, y1, x2, y2 = box.astype(int)
        # 获取当前边界框的类别标签
        label = labels[i]
        # 根据标签获取类别名称，若标签超出范围则为unknown
        class_name = class_names[label] if label < len(class_names) else 'unknown'
        # 获取当前类别对应的颜色（取模处理避免标签超出类别数）
        color = colors[label % num_classes]

        # 绘制矩形边界框（线条宽度为2）
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        # 准备文本内容：类别名称 + 实例ID（可选）
        text = f"{class_name} (ID: {ids[i]})" if ids and i < len(ids) else class_name
        # 若有置信度，添加到文本中（保留2位小数）
        if scores and i < len(scores):
            text += f" {scores[i]:.2f}"

        # 计算文本的尺寸，用于绘制文本背景框
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        # 确定文本的y坐标（避免文本超出图像顶部，若顶部空间不足则显示在框下方）
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10
        # 绘制文本背景框（填充颜色为类别颜色）
        cv2.rectangle(image_copy, (x1, text_y - text_size[1] - 2),
                      (x1 + text_size[0], text_y + 2), color, -1)
        # 绘制文本（白色字体，线条宽度为1）
        cv2.putText(image_copy, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    return image_copy

# 2. 构建相机投影矩阵（原utils.projection.build_projection_matrix）
def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """
    构建相机的内参矩阵K，用于将3D相机坐标投影到2D图像坐标
    参数说明：
        w: 图像宽度（像素）
        h: 图像高度（像素）
        fov: 相机的视场角（度数）
        is_behind_camera: 是否处理相机后方的点（反转z轴）
    返回：
        K: 3x3的相机内参矩阵
    """
    # 计算相机的焦距（根据视场角和图像宽度）
    focal = w / (2.0 * math.tan(fov * math.pi / 360.0))
    # 初始化内参矩阵为单位矩阵
    K = np.identity(3)
    # 设置x和y方向的焦距
    K[0, 0] = K[1, 1] = focal
    # 设置图像的主点（图像中心）
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    # 若处理相机后方的点，反转z轴（使投影后的坐标有效）
    if is_behind_camera:
        K[2, 2] = -1
    return K

# 3. 3D点转2D图像点（原utils.projection.get_image_point）
def get_image_point(loc, K, w2c):
    """
    将Carla的3D世界坐标转换为2D图像坐标
    参数说明：
        loc: Carla的Location对象（3D世界坐标）
        K: 相机内参矩阵（3x3）
        w2c: 世界到相机的外参矩阵（4x4，逆矩阵）
    返回：
        (x, y): 2D图像坐标（像素）
    """
    # 将3D坐标转换为齐次坐标（4维）
    point = np.array([loc.x, loc.y, loc.z, 1.0])
    # 世界坐标→相机坐标（乘以外参矩阵）
    point_camera = np.dot(w2c, point)
    # 相机坐标→图像坐标（乘以内参矩阵，取前3维）
    point_img = np.dot(K, point_camera[:3])
    # 归一化（除以z坐标，得到像素坐标）
    point_img = point_img / point_img[2]
    return (point_img[0], point_img[1])

# 4. 从3D边缘生成2D边界框（原utils.projection.get_2d_box_from_3d_edges）
def get_2d_box_from_3d_edges(points_2d, edges, h, w):
    """
    从3D点的2D投影坐标生成最小包围矩形框
    参数说明：
        points_2d: 2D投影点列表，每个元素是(x,y)
        edges: 3D物体的边缘列表（此处未使用，保留兼容）
        h: 图像高度（像素）
        w: 图像宽度（像素）
    返回：
        x_min, x_max, y_min, y_max: 边界框的最小/最大坐标（限制在图像范围内）
    """
    # 提取所有点的x和y坐标
    x_coords = [p[0] for p in points_2d]
    y_coords = [p[1] for p in points_2d]
    # 计算x的最小值（不小于0）和最大值（不大于图像宽度）
    x_min = max(0, min(x_coords))
    x_max = min(w, max(x_coords))
    # 计算y的最小值（不小于0）和最大值（不大于图像高度）
    y_min = max(0, min(y_coords))
    y_max = min(h, max(y_coords))
    return x_min, x_max, y_min, y_max

# 5. 判断点是否在图像内（原utils.projection.point_in_canvas）
def point_in_canvas(point, h, w):
    """
    检查2D点是否在图像画布的有效范围内
    参数说明：
        point: 2D点坐标(x,y)
        h: 图像高度（像素）
        w: 图像宽度（像素）
    返回：
        bool: 点在图像内返回True，否则返回False
    """
    x, y = point
    return 0 <= x < w and 0 <= y < h

# 6. 清理Carla中的车辆（原utils.world.clear_npc/clear_static_vehicle/clear）
def clear_actors(world, camera=None):
    """
    销毁Carla世界中的所有车辆和相机传感器，释放资源
    参数说明：
        world: Carla的World对象
        camera: 可选，需要销毁的相机传感器对象
    """
    # 销毁相机传感器（若存在）
    if camera:
        camera.destroy()
    # 遍历并销毁所有车辆Actor
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()

# ===================== COCO类别名称（对应车辆类别） =====================
# 无需依赖what库，直接定义核心类别（标签对应索引）
COCO_CLASS_NAMES = [
    'person',  # 0
    'bicycle', # 1
    'car',     # 2（车辆类别，对应本文档的核心检测目标）
    'motorcycle', # 3
    'airplane',   # 4
    'bus',        # 5
    'train',      # 6
    'truck',      # 7
    'boat',       # 8
    'traffic light' # 9
]

# ===================== 相机回调函数 =====================
def camera_callback(image, rgb_image_queue):
    """
    相机传感器的回调函数，将原始图像数据转换为numpy数组并存入队列
    参数说明：
        image: Carla的Image对象（原始相机数据）
        rgb_image_queue: 存储图像的队列（线程安全）
    """
    # Carla的图像数据是BGRA格式（4通道），转换为(height, width, 4)的numpy数组
    rgb_image = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    # 将图像存入队列
    rgb_image_queue.put(rgb_image)

# ===================== 主程序 =====================
if __name__ == '__main__':
    # 1. 连接Carla客户端（本地主机，端口2000）
    client = carla.Client('localhost', 2000)
    # 设置客户端超时时间（20秒），避免连接卡死
    client.set_timeout(20.0)
    # 获取Carla的世界对象（核心交互对象）
    world = client.get_world()

    # 2. 设置Carla同步模式（稳定控制仿真帧率，避免图像和物理不同步）
    settings = world.get_settings()
    # 启用同步模式（需要手动调用world.tick()推进仿真）
    settings.synchronous_mode = True
    # 设置固定的仿真步长（0.05秒，对应20 FPS）
    settings.fixed_delta_seconds = 0.05
    # 应用设置
    world.apply_settings(settings)

    # 3. 获取核心资源：蓝图库、出生点、观察者
    bp_lib = world.get_blueprint_library()  # 蓝图库，存储所有可生成的Actor蓝图（车辆、相机、传感器等）
    spawn_points = world.get_map().get_spawn_points()  # 地图的预设出生点列表（车辆生成的位置和姿态）
    spectator = world.get_spectator()  # 观察者（用于调整仿真画面的视角）

    # 4. 生成自车（林肯MKZ 2020款）
    # 从蓝图库中查找指定车辆的蓝图
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    # 随机选择一个出生点生成车辆（try_spawn_actor避免出生点冲突返回None）
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    # 防止出生点冲突，循环重试生成自车，直到成功
    while not vehicle:
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

    # 5. 生成相机传感器（挂载在自车前方，前视视角）
    # 从蓝图库中查找RGB相机的蓝图
    camera_bp = bp_lib.find('sensor.camera.rgb')
    # 设置相机的图像分辨率（640x640像素）
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '640')
    # 设置相机的变换（相对于自车的位置：前方1.0米，上方2.0米）
    camera_transform = carla.Transform(carla.Location(x=1.0, z=2.0))
    # 生成相机Actor，并挂载到自车上
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # 6. 初始化图像队列，监听相机数据
    # 创建队列，用于存储相机回调函数的图像数据（线程安全）
    image_queue = queue.Queue()
    # 注册相机的回调函数，当相机产生图像时自动调用
    camera.listen(lambda img: camera_callback(img, image_queue))

    # 7. 清理原有车辆，生成50辆NPC车辆（开启自动驾驶）
    # 销毁世界中已存在的所有车辆（避免场景混乱）
    clear_actors(world)
    # 循环生成50辆NPC车辆
    for i in range(50):
        # 筛选蓝图库中的4轮车辆（排除自行车、摩托车等2轮车辆）
        vehicle_bps = [bp for bp in bp_lib.filter('vehicle') if int(bp.get_attribute('number_of_wheels')) == 4]
        # 随机选择车辆蓝图和出生点生成NPC车辆
        npc = world.try_spawn_actor(random.choice(vehicle_bps), random.choice(spawn_points))
        # 若生成成功，开启NPC车辆的自动驾驶模式
        if npc:
            npc.set_autopilot(True)
    # 自车也开启自动驾驶模式
    vehicle.set_autopilot(True)

    # 8. 初始化3D→2D投影的参数
    # 车辆包围盒的边缘连接列表（3D物体的12条边缘，此处未使用，保留兼容）
    edges = [[0,1],[1,3],[3,2],[2,0],[0,4],[4,5],[5,1],[5,7],[7,6],[6,4],[6,2],[7,3]]
    # 图像的宽度和高度（与相机设置一致）
    image_w = 640
    image_h = 640
    # 获取相机的视场角（从相机蓝图的属性中读取）
    fov = camera_bp.get_attribute('fov').as_float()
    # 构建相机内参矩阵（处理相机前方的点）
    K = build_projection_matrix(image_w, image_h, fov)
    # 构建相机内参矩阵（处理相机后方的点，反转z轴）
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

    # ===================== 主循环：实时可视化地面真值 =====================
    print("程序已启动，按q键退出...")
    try:
        while True:
            # 推进Carla仿真一帧（同步模式下必须调用，否则仿真不会运行）
            world.tick()

            # 移动观察者到自车上方（俯视视角，方便查看全局场景）
            # 设置观察者的位置：自车后方4米，上方50米；姿态：偏航角-180度，俯仰角-90度（垂直向下看）
            spectator_transform = carla.Transform(
                vehicle.get_transform().transform(carla.Location(x=-4, z=50)),
                carla.Rotation(yaw=-180, pitch=-90)
            )
            # 应用观察者的变换
            spectator.set_transform(spectator_transform)

            # 从队列中获取相机图像（阻塞等待，直到有图像）
            image = image_queue.get()
            # 获取世界到相机的变换矩阵（外参矩阵的逆矩阵，用于坐标转换）
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # 初始化边界框和实例ID列表
            boxes = []
            ids = []
            # 遍历世界中所有的车辆Actor
            for npc in world.get_actors().filter('*vehicle*'):
                # 跳过自车（只处理NPC车辆）
                if npc.id == vehicle.id:
                    continue

                # 筛选条件：距离自车50米内，且在自车前方的车辆
                # 计算NPC车辆与自车的直线距离
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                # 获取自车的前向向量
                forward_vec = vehicle.get_transform().get_forward_vector()
                # 计算自车到NPC车辆的向量
                ray = npc.get_transform().location - vehicle.get_transform().location
                # 点积大于0表示NPC车辆在自车前方（前向向量与射线向量同向）
                if dist < 50 and forward_vec.dot(ray) > 0:
                    # 获取NPC车辆包围盒的3D顶点（世界坐标）
                    bb_verts = [v for v in npc.bounding_box.get_world_vertices(npc.get_transform())]
                    points_2d = []
                    # 遍历每个3D顶点，投影到2D图像
                    for vert in bb_verts:
                        # 计算顶点到相机的向量
                        ray_cam = vert - camera.get_transform().location
                        # 获取相机的前向向量
                        cam_forward = camera.get_transform().get_forward_vector()
                        # 判断顶点是否在相机前方：使用对应内参矩阵投影
                        if cam_forward.dot(ray_cam) > 0:
                            # 相机前方的点：使用普通内参矩阵K
                            p = get_image_point(vert, K, world_2_camera)
                        else:
                            # 相机后方的点：使用反转z轴的内参矩阵K_b
                            p = get_image_point(vert, K_b, world_2_camera)
                        # 将2D投影点加入列表
                        points_2d.append(p)
                    # 从2D投影点生成最小包围边界框
                    x_min, x_max, y_min, y_max = get_2d_box_from_3d_edges(points_2d, edges, image_h, image_w)
                    # 过滤条件：边界框的面积大于100像素，宽度大于20像素（排除过小的无效框）
                    if (y_max - y_min)*(x_max - x_min) > 100 and (x_max - x_min) > 20:
                        # 过滤条件：边界框的四个角点都在图像范围内
                        if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):
                            # 将边界框坐标加入列表（格式：[x1,y1,x2,y2]）
                            boxes.append(np.array([x_min, y_min, x_max, y_max]))
                            # 将NPC车辆的ID加入列表
                            ids.append(npc.id)

            # 绘制边界框：转换为numpy数组，设置类别标签为2（对应COCO的car类别）
            boxes = np.array(boxes)
            labels = np.array([2] * len(boxes))  # 2 = car（对应COCO_CLASS_NAMES[2]）
            # 调用绘制函数，生成带边界框的图像
            output_image = draw_bounding_boxes(image, boxes, labels, COCO_CLASS_NAMES, ids)

            # 显示图像（窗口名称：Carla 2D Ground Truth (Vehicles)）
            cv2.imshow('Carla 2D Ground Truth (Vehicles)', output_image)
            # 按q键退出循环（等待1ms，处理键盘事件）
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 捕获键盘中断（Ctrl+C）
    except KeyboardInterrupt:
        pass
    # 最终清理资源（无论是否异常，都会执行）
    finally:
        # 销毁车辆和相机传感器
        clear_actors(world, camera)
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        # 打印退出信息
        print("程序已退出，资源已清理")