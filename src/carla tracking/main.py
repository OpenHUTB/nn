import argparse
import carla
import os
import queue
import random
import cv2
import numpy as np
import torch
from collections import deque


# -------------------------- 内置 SORT 跟踪器（无需安装任何依赖） --------------------------
class KalmanFilter:
    def __init__(self):
        self.dt = 1.0
        self.x = np.zeros((4, 1))  # [x, y, vx, vy]
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.P = np.eye(4) * 1000
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 10

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:2]

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x[:2]


class Track:
    def __init__(self, box, track_id):
        self.id = track_id
        self.kf = KalmanFilter()
        self.x1, self.y1, self.x2, self.y2 = box
        self.center = np.array([[(self.x1 + self.x2) / 2], [(self.y1 + self.y2) / 2]])
        self.kf.x[:2] = self.center
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.hits = 1
        self.age = 0

    def predict(self):
        self.age += 1
        center = self.kf.predict()
        self.x1 = center[0, 0] - self.width / 2
        self.y1 = center[1, 0] - self.height / 2
        self.x2 = self.x1 + self.width
        self.y2 = self.y1 + self.height
        return [self.x1, self.y1, self.x2, self.y2]

    def update(self, box):
        self.x1, self.y1, self.x2, self.y2 = box
        self.center = np.array([[(self.x1 + self.x2) / 2], [(self.y1 + self.y2) / 2]])
        self.kf.update(self.center)
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.hits += 1
        self.age = 0

    def get_box(self):
        return [self.x1, self.y1, self.x2, self.y2]


class Sort:
    def __init__(self, max_age=3, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        if len(detections) == 0:
            for track in self.tracks:
                track.age += 1
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            return np.array([[t.x1, t.y1, t.x2, t.y2, t.id] for t in self.tracks if t.hits >= self.min_hits])

        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(det[:4], self.next_id))
                self.next_id += 1
            return np.array([[t.x1, t.y1, t.x2, t.y2, t.id] for t in self.tracks])

        track_boxes = np.array([t.get_box() for t in self.tracks])
        iou_matrix = self._iou_batch(track_boxes, detections[:, :4])

        matches, unmatched_tracks, unmatched_dets = self._hungarian_algorithm(iou_matrix)

        for track_idx, det_idx in matches:
            if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                self.tracks[track_idx].update(detections[det_idx][:4])

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].age += 1

        for det_idx in unmatched_dets:
            self.tracks.append(Track(detections[det_idx][:4], self.next_id))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return np.array([[t.x1, t.y1, t.x2, t.y2, t.id] for t in self.tracks if t.hits >= self.min_hits])

    def _iou_batch(self, b1, b2):
        b1_x1, b1_y1, b1_x2, b1_y2 = b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

        inter_x1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
        inter_y1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
        inter_x2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
        inter_y2 = np.minimum(b1_y2[:, None], b2_y2[None, :])

        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        return inter_area / (b1_area[:, None] + b2_area[None, :] - inter_area + 1e-6)

    def _hungarian_algorithm(self, cost_matrix):
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        matches = np.array(list(zip(row_ind, col_ind)))
        unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in matches[:, 0]]
        unmatched_dets = [i for i in range(cost_matrix.shape[1]) if i not in matches[:, 1]]
        return matches, unmatched_tracks, unmatched_dets


# -------------------------- YOLOv5 检测模型（需安装 ultralytics） --------------------------
from ultralytics import YOLO


# -------------------------- 核心工具函数（内置，无需创建文件夹） --------------------------
def draw_bounding_boxes(image, boxes, labels, class_names, track_ids=None, probs=None):
    """绘制检测/跟踪框（含类别、置信度、跟踪ID）"""
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 拼接文本信息
        conf_text = f"{probs[i]:.2f}" if (probs is not None and i < len(probs)) else ""
        label_text = f"{class_names[label]} {conf_text}".strip()
        if track_ids and i < len(track_ids):
            label_text += f"_ID:{track_ids[i]}"
        # 绘制文本背景（避免遮挡）
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label_text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image


def build_projection_matrix(w, h, fov):
    """构建相机投影矩阵"""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    return np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])


def clear_npc(world):
    """清理所有NPC车辆"""
    if not world:
        return
    for actor in world.get_actors().filter('vehicle.*'):
        try:
            # 老版本CARLA无role_name属性，直接销毁非自车车辆
            actor.destroy()
        except:
            pass


def clear_static_vehicle(world):
    """清理静态车辆"""
    if not world:
        return
    for actor in world.get_actors().filter('static.vehicle.*'):
        try:
            actor.destroy()
        except:
            pass


def clear(world, camera):
    """清理所有资源"""
    if camera:
        try:
            camera.destroy()
        except:
            pass
    clear_npc(world)
    clear_static_vehicle(world)


# -------------------------- CARLA 相关核心函数 --------------------------
def camera_callback(image, rgb_image_queue):
    """相机回调函数：将图像存入队列"""
    rgb_image_queue.put(np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)))


def setup_carla_client(host='localhost', port=2000):
    """连接CARLA服务器并设置同步模式"""
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    # 设置同步模式（保证帧率稳定）
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    return world, client


def spawn_ego_vehicle(world):
    """生成主车辆（自车）- 兼容老版本CARLA"""
    if not world:
        return None
    bp_lib = world.get_blueprint_library()
    # 兼容不同CARLA版本的车辆蓝图
    try:
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    except:
        vehicle_bp = random.choice(
            [bp for bp in bp_lib.filter('vehicle') if int(bp.get_attribute('number_of_wheels')) == 4])

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("警告：没有可用的车辆生成点！")
        return None

    # 尝试生成车辆（多次尝试避免生成失败）
    vehicle = None
    for _ in range(5):
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        if vehicle:
            break
    if vehicle:
        vehicle.set_autopilot(True)
        print("主车辆生成成功！")
    else:
        print("警告：无法生成主车辆！")
    return vehicle


def spawn_camera(world, vehicle):
    """生成相机传感器（挂载在自车）"""
    if not world or not vehicle:
        return None, None, None
    bp_lib = world.get_blueprint_library()
    camera_bp = bp_lib.find('sensor.camera.rgb')
    # 设置相机参数（降低分辨率提升CPU性能）
    camera_bp.set_attribute('image_size_x', '480')
    camera_bp.set_attribute('image_size_y', '360')
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('sensor_tick', '0.05')  # 与世界帧率一致

    # 相机安装位置（自车前方1.5米，高度2米）
    camera_init_trans = carla.Transform(carla.Location(x=1.5, z=2.0))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    # 图像队列（异步获取图像）
    image_queue = queue.Queue()
    camera.listen(lambda image: camera_callback(image, image_queue))
    return camera, image_queue, camera_bp


def spawn_npcs(world, count=10):
    """生成NPC车辆（数量适中，避免CPU卡顿）"""
    if not world:
        return
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle')
    car_bp = [bp for bp in vehicle_bp if int(bp.get_attribute('number_of_wheels')) == 4]
    spawn_points = world.get_map().get_spawn_points()
    if not car_bp or not spawn_points:
        print("警告：无法生成NPC车辆！")
        return

    spawned_count = 0
    for _ in range(count):
        npc = world.try_spawn_actor(random.choice(car_bp), random.choice(spawn_points))
        if npc:
            npc.set_autopilot(True)
            spawned_count += 1
    print(f"成功生成 {spawned_count} 辆NPC车辆")


def load_detection_model(model_type):
    """加载YOLOv5检测模型（自动下载权重）"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载YOLOv5模型（自动缓存权重）
    if model_type == 'yolov5s':
        model = YOLO("yolov5s.pt")
    elif model_type == 'yolov5m':
        model = YOLO("yolov5m.pt")
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")
    model.to(device)
    print(f"检测模型加载成功（设备：{device}）")
    return model, model.names


def setup_tracker(tracker_type):
    """初始化跟踪器（默认SORT，稳定高效）"""
    if tracker_type == 'sort':
        return Sort(max_age=3, min_hits=2, iou_threshold=0.3), None
    else:
        raise ValueError(f"不支持的跟踪器类型：{tracker_type}")


# -------------------------- 主函数（最终稳定版本） --------------------------
def main():
    parser = argparse.ArgumentParser(description='CARLA 目标检测与跟踪（兼容老版本，零额外依赖）')
    parser.add_argument('--model', type=str, default='yolov5s', choices=['yolov5s', 'yolov5m'],
                        help='检测模型（yolov5s更轻量）')
    parser.add_argument('--tracker', type=str, default='sort', choices=['sort'], help='跟踪器（仅保留稳定的SORT）')
    parser.add_argument('--host', type=str, default='localhost', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口')
    args = parser.parse_args()

    # 提前初始化所有变量，避免未定义错误
    world = None
    camera = None
    vehicle = None
    image_queue = None

    try:
        # 1. 初始化设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"当前使用设备：{device}")

        # 2. 连接CARLA服务器（必须先启动CARLA）
        print(f"正在连接CARLA服务器 {args.host}:{args.port}...")
        world, client = setup_carla_client(args.host, args.port)
        spectator = world.get_spectator()
        print("CARLA服务器连接成功！")

        # 3. 清理环境（避免残留车辆）
        clear_npc(world)
        clear_static_vehicle(world)
        print("环境清理完成！")

        # 4. 生成主车辆
        vehicle = spawn_ego_vehicle(world)
        if not vehicle:
            print("无法生成主车辆，程序退出！")
            return

        # 5. 生成相机传感器
        camera, image_queue, camera_bp = spawn_camera(world, vehicle)
        if not camera:
            print("无法生成相机，程序退出！")
            return
        print("相机传感器生成成功！")

        # 6. 生成NPC车辆
        spawn_npcs(world, count=10)

        # 7. 加载检测模型和跟踪器
        model, class_names = load_detection_model(args.model)
        tracker, _ = setup_tracker(args.tracker)

        # 8. 主循环（检测+跟踪）
        print("开始目标检测与跟踪（按 'q' 键退出程序）")
        while True:
            # 同步CARLA世界时钟
            world.tick()

            # 移动视角到自车上方（方便观察）
            ego_transform = vehicle.get_transform()
            spectator_transform = carla.Transform(
                ego_transform.transform(carla.Location(x=-5, z=8)),
                carla.Rotation(yaw=ego_transform.rotation.yaw - 180, pitch=-30)
            )
            spectator.set_transform(spectator_transform)

            # 获取相机图像（跳过空队列）
            if image_queue.empty():
                continue
            origin_image = image_queue.get()
            # CARLA图像默认BGRA格式，转换为RGB
            image = cv2.cvtColor(origin_image, cv2.COLOR_BGRA2RGB)
            height, width, _ = image.shape

            # -------------------------- 目标检测（YOLOv5） --------------------------
            results = model(image, conf=0.5)  # 置信度阈值0.5，过滤低置信度结果
            boxes, labels, probs = [], [], []
            for r in results:
                for box in r.boxes:
                    # 解析边界框（x1,y1,x2,y2）
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # 解析置信度和类别
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    # 只保留车辆相关类别（COCO数据集：car=2, motorcycle=3, bus=5, truck=7）
                    if cls in [2, 3, 5, 7]:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls)
                        probs.append(conf)

            # 转换为numpy数组（避免空列表报错）
            boxes = np.array(boxes) if boxes else np.array([])
            labels = np.array(labels) if labels else np.array([])
            probs = np.array(probs) if probs else np.array([])

            # -------------------------- 目标跟踪（SORT） --------------------------
            if args.tracker == 'sort' and len(boxes) > 0:
                # 转换为SORT需要的格式：[x1,y1,x2,y2,conf]
                dets = np.hstack([boxes, probs.reshape(-1, 1)]) if len(probs) > 0 else boxes
                # 更新跟踪器
                track_results = tracker.update(dets)
                # 解析跟踪结果
                track_boxes = []
                track_ids = []
                for track in track_results:
                    x1, y1, x2, y2, track_id = track
                    track_boxes.append([x1, y1, x2, y2])
                    track_ids.append(int(track_id))
                # 绘制跟踪框（所有跟踪目标统一标注为car类别）
                if track_boxes:
                    image = draw_bounding_boxes(
                        image, track_boxes,
                        labels=[2] * len(track_boxes),  # 2对应COCO的car类别
                        class_names=class_names,
                        track_ids=track_ids,
                        probs=[0.9] * len(track_boxes)  # 跟踪结果默认高置信度
                    )
            elif len(boxes) > 0:
                # 无跟踪时，仅绘制检测结果
                image = draw_bounding_boxes(image, boxes, labels, class_names, probs=probs)

            # -------------------------- 显示结果 --------------------------
            cv2.imshow(f'CARLA {args.model} + {args.tracker}', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # 按'q'退出（必须加cv2.waitKey，否则窗口卡死）
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户触发退出程序...")
                break

    except Exception as e:
        print(f"程序运行出错：{str(e)}")
        print("报错提示：1. 确保CARLA服务器已启动；2. 确保carla包版本与服务器一致；3. 确保已安装所有依赖")
    finally:
        # 安全清理所有资源
        print("正在清理资源...")
        if camera:
            try:
                camera.destroy()
            except:
                pass
        if world:
            # 恢复CARLA异步模式
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            # 清理车辆
            clear_npc(world)
            clear_static_vehicle(world)
        # 关闭图像窗口
        cv2.destroyAllWindows()
        print("资源清理完成，程序正常退出！")


if __name__ == "__main__":
    main()

