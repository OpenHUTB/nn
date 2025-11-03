import carla
import cv2
import numpy as np
import pygame
import time
import os
from typing import List, Tuple


class YOLOv3TinyDetector:
    def __init__(self, cfg_path: str, weights_path: str, classes_path: str):
        self._check_file_exists(cfg_path, weights_path, classes_path)
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with open(classes_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.output_layers = ["yolo_16", "yolo_23"]  # YOLOv3-Tiny输出层
        self.conf_threshold = 0.2  # 降低阈值以识别远处目标
        self.nms_threshold = 0.25

    def _check_file_exists(self, *paths: str):
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在：{path}")

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (320, 320),
            mean=(0, 0, 0), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 核心修复：强制转换为可迭代列表
        indices_arr = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        indices = []
        if len(indices_arr) > 0:
            indices = indices_arr.flatten().tolist()  # 确保是列表

        results = []
        for i in indices:
            i = int(i)
            if 0 <= i < len(boxes):
                x, y, w, h = boxes[i]
                x1, y1, x2, y2 = x, y, x + w, y + h
                class_name = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown"
                results.append((x1, y1, x2, y2, class_name, confidences[i]))
        return results


class CarlaObjectDetector:
    def __init__(self, detector):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.detector = detector
        self.vehicle = None
        self.camera = None
        self.display = None
        self.running = True

    def _spawn_actors(self):
        vehicle_bp = self.blueprint_library.filter("model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        self.vehicle.set_autopilot(True)

        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "480")
        camera_bp.set_attribute("fov", "100")  # 增大FOV以捕捉更多远处目标
        camera_transform = carla.Transform(
            carla.Location(x=0.8, y=0, z=1.2),
            carla.Rotation(pitch=-1)
        )
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera.listen(self._process_image)

    def _process_image(self, image: carla.Image):
        if not self.running:
            return
        img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )[:, :, :3]
        img = np.ascontiguousarray(img)

        try:
            detections = self.detector.detect(img)
        except Exception as e:
            print(f"检测出错：{e}")
            detections = []

        for (x1, y1, x2, y2, class_name, conf) in detections:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(
                img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1)).convert()
        self.display.blit(surf, (0, 0))
        pygame.display.flip()

    def run(self):
        pygame.init()
        self.display = pygame.display.set_mode(
            (640, 480), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA 目标检测（修复版）")
        self._spawn_actors()
        print("程序启动成功！按ESC键退出...")

        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                time.sleep(0.01)
        except Exception as e:
            print(f"运行错误：{e}")
        finally:
            self.running = False
            if self.camera:
                self.camera.destroy()
            if self.vehicle:
                self.vehicle.destroy()
            pygame.quit()


if __name__ == "__main__":
    YOLO_CFG = "yolov3-tiny.cfg"
    YOLO_WEIGHTS = "yolov3-tiny.weights"
    YOLO_CLASSES = "coco.names"
    detector = YOLOv3TinyDetector(YOLO_CFG, YOLO_WEIGHTS, YOLO_CLASSES)
    carla_detector = CarlaObjectDetector(detector)
    carla_detector.run()