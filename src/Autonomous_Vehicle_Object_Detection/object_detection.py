import carla
import cv2
import numpy as np
import pygame
import time
import os
from typing import List, Tuple


class YOLOv3Detector:
    """YOLOv3目标检测器（适配用于CARLA场景）"""

    def __init__(self, cfg_path: str, weights_path: str, classes_path: str):
        # 检查模型文件
        self._check_file_exists(cfg_path, weights_path, classes_path)

        # 加载模型
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # 加载类别
        with open(classes_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # 输出层（固定值）
        self.output_layers = ["yolo_82", "yolo_94", "yolo_106"]

        # 降低CARLA场景优化的阈值（适合远处目标）
        self.conf_threshold = 0.25  # 置信度阈值（较低，提高检测灵敏度）
        self.nms_threshold = 0.3  # NMS阈值（控制重叠框过滤）

    def _check_file_exists(self, *paths: str):
        """检查文件是否存在"""
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在：{path}")

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        """检测图像中的目标，返回边界框和类别"""
        height, width = image.shape[:2]

        # 预处理图像
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416),
            mean=(0, 0, 0), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        # 解析检测结果
        boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_threshold:
                    # 转换为图像坐标
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # NMS处理（兼容各种索引格式）
        indices_arr = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        indices = []
        if len(indices_arr) > 0:
            indices_arr = np.squeeze(indices_arr).astype(int)
            # 处理单个索引的情况（避免int不可迭代错误）
            indices = [indices_arr] if isinstance(indices_arr, np.integer) else indices_arr.tolist()

        # 整理结果
        results = []
        for i in indices:
            i = int(i)  # 强制转换为整数
            if 0 <= i < len(boxes):
                x, y, w, h = boxes[i]
                x1, y1, x2, y2 = x, y, x + w, y + h
                class_name = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown"
                results.append((x1, y1, x2, y2, class_name, confidences[i]))

        return results


class CarlaObjectDetector:
    """CARLA场景目标检测主控制器"""

    def __init__(self, yolo_cfg: str, yolo_weights: str, yolo_classes: str):
        # 初始化CARLA客户端
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # 初始化检测器和组件
        self.detector = YOLOv3Detector(yolo_cfg, yolo_weights, yolo_classes)
        self.vehicle = None
        self.camera = None
        self.display = None
        self.running = True

    def _spawn_actors(self):
        """生成车辆和相机（第一人称视角）"""
        # 生成车辆（Model3）
        vehicle_bp = self.blueprint_library.filter("model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        self.vehicle.set_autopilot(True)  # 自动行驶

        # 第一人称相机设置（贴近驾驶员视角）
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "100")  # 广角，适合第一人称
        # 相机位置：驾驶员视角（x=0.8m在车中心前方，z=1.2m高度）
        camera_transform = carla.Transform(
            carla.Location(x=0.8, y=0, z=1.2),  # 贴近真实驾驶视角
            carla.Rotation(pitch=-1)  # 轻微低头
        )
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )

        # 注册图像回调
        self.camera.listen(self._process_image)

    def _process_image(self, image: carla.Image):
        """处理相机图像并绘制检测结果"""
        if not self.running:
            return

        # 1. 转换CARLA图像格式（BGRA→BGR）并确保内存连续
        img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )[:, :, :3]  # 去除Alpha通道
        img = np.ascontiguousarray(img)  # 修复OpenCV格式兼容问题

        # 2. 目标检测（带异常捕获）
        try:
            detections = self.detector.detect(img)
        except Exception as e:
            print(f"检测出错：{e}")
            detections = []

        # 3. 绘制检测框
        for (x1, y1, x2, y2, class_name, conf) in detections:
            # 绘制矩形框（绿色，线宽2）
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制标签
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(
                img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # 4. 转换为Pygame格式并显示
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
        self.display.blit(surf, (0, 0))
        pygame.display.flip()

    def run(self):
        """启动主循环"""
        try:
            # 初始化Pygame显示
            pygame.init()
            self.display = pygame.display.set_mode(
                (800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            pygame.display.set_caption("CARLA 第一人称目标检测")

            # 生成车辆和相机
            self._spawn_actors()
            print("程序启动成功！按ESC键退出...")

            # 主循环
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                time.sleep(0.01)  # 降低CPU占用

        except Exception as e:
            print(f"运行错误：{e}")
        finally:
            # 清理资源
            self.running = False
            if self.camera:
                self.camera.destroy()
                print("相机已销毁")
            if self.vehicle:
                self.vehicle.destroy()
                print("车辆已销毁")
            pygame.quit()
            print("程序退出")


if __name__ == "__main__":
    # 模型文件路径（确保与脚本同目录）
    YOLO_CFG = "yolov3.cfg"
    YOLO_WEIGHTS = "yolov3.weights"
    YOLO_CLASSES = "coco.names"

    # 启动检测
    detector = CarlaObjectDetector(YOLO_CFG, YOLO_WEIGHTS, YOLO_CLASSES)
    detector.run()