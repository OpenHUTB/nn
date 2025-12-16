import argparse
import carla
import queue
import random
import cv2
import numpy as np
from ultralytics import YOLO

# 内置简易SORT跟踪器（无第三方跟踪库依赖）
class KalmanFilter:
    def __init__(self):
        self.dt = 0.05  # 与CARLA同步步长一致
        self.x = np.zeros(8)  # [x1, y1, x2, y2, vx, vy, vw, vh]
        self.F = np.array([
            [1, 0, 0, 0, self.dt, 0, 0, 0],
            [0, 1, 0, 0, 0, self.dt, 0, 0],
            [0, 0, 1, 0, 0, 0, self.dt, 0],
            [0, 0, 0, 1, 0, 0, 0, self.dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        self.H = np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0]])
        self.Q = np.diag([1,1,1,1,10,10,10,10])
        self.R = np.diag([10,10,10,10])
        self.P = np.eye(8) * 100

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.x[:4]

class Track:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.kf = KalmanFilter()
        self.bbox = bbox
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        self.bbox = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox):
        self.bbox = self.kf.update(bbox)
        self.hits += 1
        self.time_since_update = 0

class SimpleSORT:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        for track in self.tracks:
            track.predict()

        track_boxes = np.array([t.bbox for t in self.tracks])
        det_boxes = np.array([d[:4] for d in detections])
        iou_matrix = self._iou_batch(det_boxes, track_boxes)

        matches = []
        used_dets = set()
        used_tracks = set()

        for t_idx in range(len(self.tracks)):
            if t_idx in used_tracks:
                continue
            d_idx = np.argmax(iou_matrix[:, t_idx])
            if iou_matrix[d_idx, t_idx] > self.iou_threshold and d_idx not in used_dets:
                matches.append((t_idx, d_idx))
                used_dets.add(d_idx)
                used_tracks.add(t_idx)

        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(detections[d_idx][:4])

        for d_idx in range(len(detections)):
            if d_idx not in used_dets:
                self.tracks.append(Track(self.next_id, detections[d_idx][:4]))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        confirmed = [(t.bbox, t.track_id) for t in self.tracks if t.hits >= self.min_hits]
        return np.array([b for b, _ in confirmed]), np.array([i for _, i in confirmed])

    def _iou_batch(self, boxes1, boxes2):
        ious = np.zeros((len(boxes1), len(boxes2)))
        for i, b1 in enumerate(boxes1):
            for j, b2 in enumerate(boxes2):
                ious[i, j] = self._iou(b1, b2)
        return ious

    def _iou(self, b1, b2):
        x1, y1, x2, y2 = b1
        a1, b1, a2, b2 = b2
        inter = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (a2 - a1) * (b2 - b1)
        return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0

# 工具函数（无需外部utils库）
def draw_bounding_boxes(image, boxes, ids):
    """在图像上绘制跟踪框和ID"""
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制跟踪ID
        cv2.putText(image, f"ID: {ids[i]}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def clear_actors(world):
    """清理CARLA中的所有车辆和传感器"""
    for actor in world.get_actors():
        if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('sensor.'):
            actor.destroy()

def camera_callback(image, queue):
    """相机数据回调函数"""
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))[:, :, :3]  # 去除Alpha通道
    queue.put(img)

def main():
    parser = argparse.ArgumentParser(description="CARLA目标跟踪（YOLOv5 + 内置SORT）")
    parser.add_argument("--host", default="localhost", help="CARLA服务器地址")
    parser.add_argument("--port", type=int, default=2000, help="CARLA服务器端口")
    parser.add_argument("--num_npcs", type=int, default=30, help="NPC车辆数量")
    args = parser.parse_args()

    # 初始化CARLA客户端
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    # 开启同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 清理现有Actor
    clear_actors(world)

    # 生成主车辆
    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    spawn_points = world.get_map().get_spawn_points()
    ego_vehicle = world.try_spawn_actor(ego_bp, random.choice(spawn_points))
    if not ego_vehicle:
        print("无法生成主车辆，退出程序")
        return
    ego_vehicle.set_autopilot(True)

    # 生成相机（挂载到主车辆）
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))  # 相机位置
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

    # 图像队列
    image_queue = queue.Queue()
    camera.listen(lambda img: camera_callback(img, image_queue))

    # 生成NPC车辆
    for _ in range(args.num_npcs):
        npc_bp = random.choice([bp for bp in bp_lib.filter('vehicle') 
                              if int(bp.get_attribute('number_of_wheels')) == 4])
        npc = world.try_spawn_actor(npc_bp, random.choice(spawn_points))
        if npc:
            npc.set_autopilot(True)

    # 初始化YOLOv5检测器和SORT跟踪器
    detector = YOLO("yolov5s.pt")  # 自动下载轻量版模型
    tracker = SimpleSORT(max_age=2, min_hits=2, iou_threshold=0.3)

    # 主循环
    try:
        while True:
            world.tick()  # 推进仿真

            # 调整 spectator 视角到主车辆上方
            ego_transform = ego_vehicle.get_transform()
            world.get_spectator().set_transform(carla.Transform(
                ego_transform.location + carla.Location(x=-8, z=12),
                carla.Rotation(pitch=-45, yaw=ego_transform.rotation.yaw)
            ))

            # 处理相机图像
            if not image_queue.empty():
                image = image_queue.get()
                height, width, _ = image.shape

                # YOLOv5目标检测（仅保留车辆类）
                results = detector.predict(image, conf=0.4)  # 置信度阈值0.4
                boxes, scores = [], []
                for result in results:
                    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                        cls_id = int(cls)
                        # COCO数据集车辆类：2=car, 5=bus, 7=truck
                        if cls_id in [2, 5, 7]:
                            boxes.append(box.numpy())  # [x1, y1, x2, y2]
                            scores.append(conf.numpy())

                # 转换为numpy数组
                boxes = np.array(boxes) if boxes else np.array([])
                scores = np.array(scores) if scores else np.array([])

                # 目标跟踪
                tracked_boxes, track_ids = [], []
                if len(boxes) > 0:
                    # 跟踪器输入格式：[x1, y1, x2, y2, score]
                    detections = np.hstack((boxes, scores.reshape(-1, 1)))
                    tracked_boxes, track_ids = tracker.update(detections)

                # 绘制跟踪结果
                if len(tracked_boxes) > 0:
                    display_img = image.copy()
                    display_img = draw_bounding_boxes(display_img, tracked_boxes, track_ids)
                    cv2.imshow("CARLA Vehicle Tracking (YOLOv5 + SORT)", display_img)

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("用户中断程序")
    finally:
        # 清理资源
        camera.stop()
        clear_actors(world)
        # 关闭同步模式
        settings.synchronous_mode = False
        world.apply_settings(settings)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()