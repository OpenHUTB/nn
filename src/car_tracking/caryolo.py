import os
import sys
import traceback  # 用于打印错误堆栈
# ====================== 关键：添加src/2d-carla-tracking-master到Python搜索路径 ======================
# 替换为你本地的src/2d-carla-tracking-master绝对路径（例如：D:\nn\src\2d-carla-tracking-master）
CARLA_TRACKING_ROOT = r"D:\nn\src\2d-carla-tracking-master"  # 必须修改为你的实际路径
sys.path.append(CARLA_TRACKING_ROOT)

import queue
import random
import time
import numpy as np
import cv2
import carla  # 确保carla库已安装且版本与模拟器匹配

# ====================== 直接导入utils模块（保留原名，去掉多余路径） ======================
# 1. 导入oc_sort（在src/2d-carla-tracking-master下）
from oc_sort.ocsort import OCSort
# 2. 导入utils模块中的函数（模块名就是utils，完全保留原名，确保utils在2d-carla-tracking-master下）
from utils.box_utils import draw_bounding_boxes
from utils.projection import build_projection_matrix
from utils.world import clear_npc, clear_static_vehicle, clear

# ====================== 全局配置项（可根据需求修改） ======================
# Carla连接配置
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0  # 单次连接超时时间（秒）
CARLA_RETRY_MAX = 3   # Carla连接最大重试次数
CARLA_RETRY_DELAY = 2  # 重试间隔（秒）
SYNC_DELTA_SECONDS = 0.05  # 同步模式的固定时间步长

# 摄像头配置
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 640
CAMERA_FOV = 90  # 摄像头视场角
CAMERA_POSITION = carla.Transform(carla.Location(x=1, z=2))  # 摄像头挂载位置

# 目标检测配置
YOLO_MODEL_INDEX = 0  # 0: YOLOv4, 1: YOLOv4 Tiny
DETECT_CLASSES = [2, 5, 7]  # COCO类别：2-汽车，5-公交车，7-卡车
DETECT_CLASS_ID = 2  # 统一标注为汽车类别用于绘制

# OC-SORT跟踪器配置
OCSORT_DET_THRESH = 0.6  # 检测置信度阈值
OCSORT_IOU_THRESH = 0.3  # IOU匹配阈值
OCSORT_USE_BYTE = False  # 是否使用ByteTrack

# NPC车辆配置
NPC_VEHICLE_NUM = 50  # 生成的NPC车辆数量

# ====================== 抑制TensorFlow冗余警告（可选） ======================
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # 只显示ERROR级别日志，屏蔽WARNING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽TensorFlow的GPU库警告

# ====================== 目标检测模型加载 ======================
def load_yolov4_model():
    """加载YOLOv4目标检测模型（包含异常处理）"""
    try:
        from what.models.detection.datasets.coco import COCO_CLASS_NAMES
        from what.models.detection.yolo.yolov4 import YOLOV4
        from what.cli.model import (
            what_model_list, WHAT_MODEL_FILE_INDEX, WHAT_MODEL_URL_INDEX,
            WHAT_MODEL_HASH_INDEX, WHAT_MODEL_PATH
        )
        from what.utils.file import get_file

        # 获取模型列表（YOLOv4和YOLOv4 Tiny）
        what_yolov4_model_list = what_model_list[4:6]
        model_info = what_yolov4_model_list[YOLO_MODEL_INDEX]

        # 下载模型（如果不存在）
        model_file = model_info[WHAT_MODEL_FILE_INDEX]
        model_url = model_info[WHAT_MODEL_URL_INDEX]
        model_hash = model_info[WHAT_MODEL_HASH_INDEX]
        model_path = os.path.join(WHAT_MODEL_PATH, model_file)

        if not os.path.isfile(model_path):
            print(f"正在下载YOLOv4模型：{model_file}")
            get_file(model_file, WHAT_MODEL_PATH, model_url, model_hash)
            print("模型下载完成")

        # 初始化YOLOv4模型
        model = YOLOV4(COCO_CLASS_NAMES, model_path)
        return model, COCO_CLASS_NAMES
    except ImportError as e:
        raise ImportError(f"缺少what库或相关依赖：{e}")
    except Exception as e:
        raise RuntimeError(f"模型加载失败：{e}")

# ====================== Carla环境初始化（含重试逻辑） ======================
def init_carla_environment():
    """初始化Carla环境：连接服务器、设置同步模式、生成主车辆和摄像头"""
    # 连接Carla服务器（带重试逻辑）
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CARLA_TIMEOUT)
    world = None
    retry_count = 0

    while retry_count < CARLA_RETRY_MAX and world is None:
        try:
            print(f"尝试连接Carla模拟器（{retry_count + 1}/{CARLA_RETRY_MAX}）...")
            world = client.get_world()
        except RuntimeError as e:
            retry_count += 1
            print(f"连接失败：{e}")
            if retry_count < CARLA_RETRY_MAX:
                print(f"{CARLA_RETRY_DELAY}秒后重试...")
                time.sleep(CARLA_RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"多次重试后仍无法连接Carla模拟器，请检查：\n"
                    f"1. Carla模拟器是否已启动（运行CarlaUE4.exe）\n"
                    f"2. 模拟器端口是否为{CARLA_PORT}\n"
                    f"3. Python的carla库版本是否与模拟器一致"
                )

    print("Carla模拟器连接成功！")

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = SYNC_DELTA_SECONDS
    world.apply_settings(settings)
    print("Carla同步模式已启用")

    # 获取蓝图库和生成点
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("未找到地图生成点，请检查Carla地图是否加载正常")

    # 生成主车辆（林肯MKZ 2020，确保生成成功）
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = None
    vehicle_retry = 0
    while not vehicle and vehicle_retry < 10:  # 最多重试10次
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        vehicle_retry += 1
    if not vehicle:
        raise RuntimeError("无法生成主车辆，请检查地图生成点是否可用")
    print("主车辆生成成功")

    # 生成RGB摄像头传感器
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    camera_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    camera_bp.set_attribute('fov', str(CAMERA_FOV))

    camera = world.spawn_actor(camera_bp, CAMERA_POSITION, attach_to=vehicle)
    print("摄像头传感器生成成功")

    # 创建图像队列，用于存储摄像头数据
    image_queue = queue.Queue(maxsize=1)  # 限制队列大小，避免堆积
    def camera_callback(image, queue_obj):
        """摄像头回调函数：将图像数据存入队列（确保数据连续且类型正确）"""
        try:
            # 转换原始数据为RGB图像（HWC，去除alpha通道）
            img_data = np.reshape(
                np.copy(image.raw_data),
                (image.height, image.width, 4)
            )[:, :, :3]  # 去掉第四个alpha通道
            # 关键：转换为uint8的连续数组，避免OpenCV布局不兼容
            img_data = np.ascontiguousarray(img_data, dtype=np.uint8)
            # 队列满时丢弃旧数据
            if not queue_obj.full():
                queue_obj.put(img_data)
        except Exception as e:
            print(f"摄像头回调函数出错：{e}")
            traceback.print_exc()

    camera.listen(lambda image: camera_callback(image, image_queue))

    # 清除现有NPC和静态车辆
    clear_npc(world)
    clear_static_vehicle(world)
    print("现有NPC和静态车辆已清除")

    # 生成NPC车辆
    spawn_npc_vehicles(world, bp_lib, spawn_points)
    print(f"成功生成{NPC_VEHICLE_NUM}辆NPC车辆（尽可能）")

    # 开启主车辆自动驾驶
    vehicle.set_autopilot(True)
    print("主车辆自动驾驶已开启")

    return world, client, vehicle, camera, image_queue

def spawn_npc_vehicles(world, bp_lib, spawn_points):
    """生成指定数量的NPC车辆并开启自动驾驶（容错处理）"""
    npc_count = 0
    # 过滤四轮车辆（排除自行车、摩托车）
    car_bps = [
        bp for bp in bp_lib.filter('vehicle')
        if int(bp.get_attribute('number_of_wheels')) == 4
    ]
    if not car_bps:
        print("警告：未找到四轮车辆蓝图")
        return

    # 循环生成，直到达到数量或无生成点
    max_spawn_retry = len(spawn_points) * 2  # 最大生成重试次数
    spawn_retry = 0
    while npc_count < NPC_VEHICLE_NUM and spawn_retry < max_spawn_retry:
        bp = random.choice(car_bps)
        spawn_point = random.choice(spawn_points)
        npc = world.try_spawn_actor(bp, spawn_point)
        if npc:
            npc.set_autopilot(True)
            npc_count += 1
        spawn_retry += 1

    if npc_count < NPC_VEHICLE_NUM:
        print(f"警告：仅生成了{npc_count}辆NPC车辆，未达到目标数量{NPC_VEHICLE_NUM}")

# ====================== 目标检测与跟踪处理（修复数组索引+OpenCV图像兼容问题） ======================
def process_detection_and_tracking(image, model, tracker, coco_names, detect_classes, detect_class_id):
    """处理单帧图像的目标检测和OC-SORT跟踪（修复数组索引+OpenCV图像兼容问题）"""
    try:
        # ====================== 修复1：确保图像是连续的uint8数组，维度正确 ======================
        # 转换为连续数组，避免OpenCV布局不兼容
        img = np.ascontiguousarray(image, dtype=np.uint8)
        # 校验图像维度：必须是(H, W, 3)的RGB图像
        if len(img.shape) != 3 or img.shape[2] != 3:
            print(f"图像维度错误：{img.shape}，预期(H, W, 3)")
            return image
        # 图像预处理：BGR转RGB（YOLO模型输入要求，注意OpenCV默认是BGR，这里确认转换）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # YOLOv4推理
        _, boxes, labels, probs = model.predict(img_rgb)

        # ====================== 核心修复：数据维度与类型校验 ======================
        # 1. 确保labels是一维数组（处理模型返回的多维labels情况）
        labels = np.ravel(labels)  # 展平为一维数组
        # 2. 确保boxes是二维数组（形状为(n, 4)，n为检测框数量）
        if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
            # 若维度异常，直接返回原始图像（无检测结果）
            return img
        # 3. 确保probs是一维数组，且长度与labels/boxes一致
        probs = np.ravel(probs)
        # 过滤掉长度不匹配的情况
        min_len = min(len(boxes), len(labels), len(probs))
        boxes = boxes[:min_len]
        labels = labels[:min_len]
        probs = probs[:min_len]

        # 过滤目标：只保留汽车、公交车、卡车（用整数索引替代np.isin，避免掩码问题）
        keep_indices = [i for i, label in enumerate(labels) if label in detect_classes]
        # 按整数索引过滤数组（确保索引是整数标量）
        boxes = boxes[keep_indices]
        probs = probs[keep_indices]
        # 确保labels的长度与检测框一致
        labels = np.full(len(boxes), detect_class_id, dtype=np.int32)  # 统一标注为汽车类别，指定int32类型

        # 转换检测框格式：(xc, yc, w, h) -> (x1, y1, x2, y2)（OC-SORT要求）
        dets = np.empty((0, 5), dtype=np.float32)  # 初始化检测结果：(x1, y1, x2, y2, score)
        if len(boxes) > 0:
            height, width = img.shape[:2]
            # 反归一化：将0-1的坐标转换为像素坐标（转为float32避免整数溢出）
            boxes = boxes.astype(np.float32)
            boxes[:, 0] *= width   # xc（中心x）
            boxes[:, 1] *= height  # yc（中心y）
            boxes[:, 2] *= width   # w（宽度）
            boxes[:, 3] *= height  # h（高度）

            # 中心坐标转左上角、右下角坐标（显式重塑维度，避免拼接错误）
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            scores = probs.reshape(-1, 1)  # 形状为(n, 1)

            # 拼接为(n, 5)的检测结果
            dets = np.hstack((
                x1.reshape(-1, 1),
                y1.reshape(-1, 1),
                x2.reshape(-1, 1),
                y2.reshape(-1, 1),
                scores
            )).astype(np.float32)

        # 更新OC-SORT跟踪器
        height, width = img.shape[:2]
        track_results = tracker.update(dets, [height, width], (height, width))

        # 绘制检测框和跟踪ID
        if len(track_results) > 0:
            # ====================== 修复2：确保跟踪结果和标签长度匹配 ======================
            # 提取检测框和跟踪ID
            track_boxes = track_results[:, :4].astype(np.int32)  # 转换为整数坐标（OpenCV需要）
            track_ids = track_results[:, 4].astype(np.int32)     # 跟踪ID转为整数
            # 确保标签长度与跟踪框一致（若不足则补全）
            if len(labels) < len(track_boxes):
                labels = np.pad(labels, (0, len(track_boxes) - len(labels)), mode='constant', constant_values=detect_class_id)
            else:
                labels = labels[:len(track_boxes)]
            # 调用绘制函数，传入正确的参数
            img = draw_bounding_boxes(img, track_boxes, labels, coco_names, track_ids)

        return img
    except Exception as e:
        print(f"检测与跟踪处理出错：{e}")
        traceback.print_exc()  # 打印完整错误堆栈，便于定位具体行
        return image  # 出错时返回原始图像

# ====================== 主函数 ======================
def main():
    """主函数：整合所有流程，包含完整的异常处理"""
    # 初始化资源变量
    world = None
    camera = None
    try:
        # 1. 加载YOLOv4模型
        print("开始加载YOLOv4模型...")
        model, coco_class_names = load_yolov4_model()
        print("YOLOv4模型加载成功")

        # 2. 初始化Carla环境
        print("开始初始化Carla环境...")
        world, client, vehicle, camera, image_queue = init_carla_environment()

        # 3. 初始化OC-SORT跟踪器
        mot_tracker = OCSort(
            det_thresh=OCSORT_DET_THRESH,
            iou_threshold=OCSORT_IOU_THRESH,
            use_byte=OCSORT_USE_BYTE
        )
        print("OC-SORT跟踪器初始化成功")

        # 获取spectator（用于视角跟随）
        spectator = world.get_spectator()

        # 4. 主循环
        print("进入主循环，按'q'退出...")
        while True:
            # 同步步进到下一帧
            world.tick()

            # 移动spectator到车辆上方的俯视视角
            vehicle_transform = vehicle.get_transform()
            spectator_transform = carla.Transform(
                vehicle_transform.transform(carla.Location(x=-4, z=50)),
                carla.Rotation(yaw=-180, pitch=-90)
            )
            spectator.set_transform(spectator_transform)

            # 获取摄像头图像（非阻塞，避免队列堆积）
            if not image_queue.empty():
                origin_image = image_queue.get()
            else:
                continue

            # 处理检测和跟踪
            output_image = process_detection_and_tracking(
                origin_image, model, mot_tracker, coco_class_names,
                DETECT_CLASSES, DETECT_CLASS_ID
            )

            # 显示结果
            cv2.imshow('2D YOLOv4 OC-SORT Vehicle Tracking', output_image)

            # 按q退出（cv2.waitKey不能省略，否则窗口卡死）
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户按下'q'，退出主循环")
                break

    except KeyboardInterrupt:
        print("\n用户中断程序（Ctrl+C）")
    except Exception as e:
        print(f"\n程序运行出错：{e}")
        traceback.print_exc()
    finally:
        # 释放资源：恢复Carla设置、销毁演员、关闭窗口
        print("开始释放资源...")
        if world is not None:
            # 恢复Carla异步模式
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            print("Carla同步模式已关闭")
        if camera is not None:
            clear(world, camera)  # 销毁摄像头等演员
            print("Carla演员已销毁")
        # 关闭OpenCV窗口
        cv2.destroyAllWindows()
        print("资源释放完成，程序退出")

if __name__ == '__main__':
    main()