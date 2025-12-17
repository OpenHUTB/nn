#目标跟踪
import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import random

class VirtualEnv:
    """虚拟环境：生成随机移动的目标"""
    def __init__(self, width=800, height=600, num_objects=3):
        self.width = width
        self.height = height
        self.objects = []  # 存储目标信息：(x, y, w, h, velocity_x, velocity_y, color)
        self._init_objects(num_objects)

    def _init_objects(self, num_objects):
        """初始化随机移动的目标"""
        for _ in range(num_objects):
            # 随机初始位置（确保在画面内）
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            w, h = random.randint(30, 60), random.randint(30, 60)
            # 随机速度（像素/帧）
            vx = random.uniform(-2, 2)
            vy = random.uniform(-2, 2)
            # 随机颜色
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            self.objects.append([x, y, w, h, vx, vy, color])

    def update(self):
        """更新目标位置（模拟随机移动）"""
        for obj in self.objects:
            x, y, w, h, vx, vy, color = obj
            # 更新位置
            x += vx
            y += vy
            # 边界反弹（模拟环境约束）
            if x <= 0 or x >= self.width - w:
                vx = -vx * random.uniform(0.8, 1.2)  # 随机调整速度
            if y <= 0 or y >= self.height - h:
                vy = -vy * random.uniform(0.8, 1.2)
            # 随机小幅度改变速度（模拟不规则运动）
            vx += random.uniform(-0.3, 0.3)
            vy += random.uniform(-0.3, 0.3)
            # 更新目标信息
            obj[:] = [x, y, w, h, vx, vy, color]

    def render(self):
        """渲染环境画面"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for obj in self.objects:
            x, y, w, h, _, _, color = obj
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        return frame


class Tracker:
    """目标跟踪器：结合YOLO检测和卡尔曼滤波预测"""
    def __init__(self, model_path="yolov8n.pt"):
        self.detector = YOLO(model_path)  # 加载YOLOv8模型
        self.trackers = {}  # 跟踪器字典：{id: 卡尔曼滤波器}
        self.next_id = 0  # 下一个目标ID

    def _init_kalman(self, bbox):
        """初始化卡尔曼滤波器（针对边界框[x, y, w, h]）"""
        kf = KalmanFilter(dim_x=8, dim_z=4)
        # 状态转移矩阵（假设匀速运动模型）
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        # 测量矩阵
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        # 噪声协方差矩阵
        kf.R *= 10.0  # 测量噪声
        kf.P *= 1000.0  # 初始状态协方差
        kf.Q[-4:, -4:] *= 0.5  # 过程噪声（速度部分）
        # 初始化状态
        kf.x[:4] = np.array(bbox).reshape(4, 1)
        return kf

    def update(self, frame):
        """更新跟踪：检测目标 -> 匹配跟踪器 -> 预测位置"""
        # 1. YOLO目标检测（这里简化为检测所有目标，实际可指定类别）
        results = self.detector(frame, classes=[0])  # 假设跟踪"人"（类别0），虚拟目标可适配
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 边界框坐标
                w, h = x2 - x1, y2 - y1
                detections.append([x1, y1, w, h])  # 存储[x, y, w, h]

        # 2. 卡尔曼滤波预测与匹配（简化版：距离最近匹配）
        new_trackers = {}
        for det in detections:
            min_dist = float('inf')
            best_id = None
            # 与现有跟踪器匹配
            for track_id, kf in self.trackers.items():
                # 预测当前位置
                kf.predict()
                pred = kf.x[:4].flatten()  # 预测的[x, y, w, h]
                # 计算检测与预测的距离（欧氏距离）
                dist = np.linalg.norm(np.array(det) - pred)
                if dist < min_dist and dist < 50:  # 阈值控制匹配
                    min_dist = dist
                    best_id = track_id
            # 匹配成功：更新跟踪器
            if best_id is not None:
                self.trackers[best_id].update(np.array(det).reshape(4, 1))
                new_trackers[best_id] = self.trackers[best_id]
            # 新目标：初始化跟踪器
            else:
                new_trackers[self.next_id] = self._init_kalman(det)
                self.next_id += 1
        self.trackers = new_trackers

        # 3. 返回跟踪结果（ID + 预测边界框）
        tracks = []
        for track_id, kf in self.trackers.items():
            x, y, w, h = kf.x[:4].flatten()
            tracks.append((track_id, int(x), int(y), int(x + w), int(y + h)))
        return tracks


def main():
    # 初始化虚拟环境和跟踪器
    env = VirtualEnv(width=1024, height=768, num_objects=4)
    tracker = Tracker()

    while True:
        # 更新虚拟环境（目标移动）
        env.update()
        frame = env.render()

        # 无人机视角跟踪（调用跟踪器）
        tracks = tracker.update(frame)

        # 绘制跟踪结果
        for track_id, x1, y1, x2, y2 in tracks:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示画面
        cv2.imshow("Drone Tracking (Virtual Objects)", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()