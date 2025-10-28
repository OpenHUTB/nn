import carla
import cv2
import numpy as np
import pygame
import time
import os
from typing import List, Tuple


class YOLOv3Detector:
    """YOLOv3目标检测器，封装模型加载和检测逻辑"""
    def __init__(self, cfg_path: str, weights_path: str, classes_path: str):
        # 检查模型文件是否存在
        self._check_file_exists(cfg_path, weights_path, classes_path)

        # 加载模型
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # 加载类别名称
        with open(classes_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # 手动指定YOLOv3输出层（避免自动获取错误）
        self.output_layers = ["yolo_82", "yolo_94", "yolo_106"]

        # 检测参数（可调整）
        self.conf_threshold = 0.5  # 置信度阈值
        self.nms_threshold = 0.4   # NMS重叠阈值

    def _check_file_exists(self, *paths: str):
        """检查文件是否存在，不存在则报错"""
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"模型文件不存在：{path}，请检查路径")

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        """
        对输入图像进行目标检测
        返回：[(x1, y1, x2, y2, 类别名称, 置信度), ...]
        """
        height, width = image.shape[:2]

        # 图像预处理（转换为YOLO输入格式）
        blob = cv2.dnn.blobFromImage(
            image, 1/255.0, (416, 416),
            mean=(0, 0, 0), swapRB=True, crop=False
        )
        self.net.setInput(blob)

        # 前向传播获取检测结果
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
                    # 转换为图像实际坐标
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 非极大值抑制（NMS）去除重叠框
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold
        )
        # 处理NMS返回的索引（兼容嵌套数组）
        indices = np.squeeze(indices).astype(int).tolist() if len(indices) > 0 else []

        # 整理最终结果
        results = []
        for i in indices:
            i = i if isinstance(i, int) else i[0]  # 确保索引为整数
            if 0 <= i < len(boxes):
                x, y, w, h = boxes[i]
                x1, y1, x2, y2 = x, y, x + w, y + h
                class_name = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown"
                results.append((x1, y1, x2, y2, class_name, confidences[i]))

        return results


class CarlaObjectDetector:
    """CARLA目标检测主类，整合车辆、相机和YOLO检测器"""
    def __init__(self, yolo_cfg: str, yolo_weights: str, yolo_classes: str):
        # 初始化CARLA客户端
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # 初始化YOLO检测器
        self.detector = YOLOv3Detector(yolo_cfg, yolo_weights, yolo_classes)

        # CARLA Actors（车辆、相机）
        self.vehicle = None
        self.camera = None

        # Pygame显示
        self.display = None
        self.running = True

    def _spawn_actors(self):
        """生成车辆和相机传感器"""
        # 生成车辆（Tesla Model3）
        vehicle_bp = self.blueprint_library.filter("model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        self.vehicle.set_autopilot(True)  # 开启自动驾驶（可改为False手动控制）

        # 生成RGB相机（安装在车辆前方）
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")  # 图像宽度
        camera_bp.set_attribute("image_size_y", "600")  # 图像高度
        camera_bp.set_attribute("fov", "90")           # 视野角度
        # 相机位置：车辆前方1.5m，上方2.4m
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )

        # 注册相机回调（处理图像并检测）
        self.camera.listen(self._process_image)

    def _process_image(self, image: carla.Image):
        """相机图像回调：转换格式→目标检测→可视化"""
        if not self.running:
            return

        # 将CARLA图像（BGRA）转换为OpenCV格式（BGR）
        img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )[:, :, :3]  # 去除Alpha通道

        # 目标检测
        detections = self.detector.detect(img)

        # 绘制检测框
        for (x1, y1, x2, y2, class_name, conf) in detections:
            # 绘制矩形框（绿色，线宽2）
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制标签（类别+置信度）
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(
                img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # 转换为Pygame格式（BGR→RGB + 维度交换）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
        self.display.blit(surf, (0, 0))
        pygame.display.flip()

    def run(self):
        """启动主循环"""
        try:
            # 初始化Pygame
            pygame.init()
            self.display = pygame.display.set_mode(
                (800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            pygame.display.set_caption("CARLA + YOLOv3 目标检测")

            # 生成车辆和相机
            self._spawn_actors()
            print("启动成功！按ESC键退出...")

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
            print(f"运行出错：{e}")
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
            print("程序已退出")


if __name__ == "__main__":
    # YOLO模型文件路径（确保与脚本在同一目录）
    YOLO_CFG = "yolov3.cfg"
    YOLO_WEIGHTS = "yolov3.weights"
    YOLO_CLASSES = "coco.names"

    # 启动检测
    detector = CarlaObjectDetector(YOLO_CFG, YOLO_WEIGHTS, YOLO_CLASSES)
    detector.run()