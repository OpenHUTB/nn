# -*- coding: utf-8 -*-
import carla
import pygame
import logging
import numpy as np
import cv2
import time
from logging import StreamHandler
import sys
import threading
# 使用兼容 Python 3.7 的 ultralytics 8.0.151
from ultralytics import YOLO

# 日志初始化（兼容 Python 3.7）
def init_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
init_logger()


class CarlaDetector:
    def __init__(self, carla_host="localhost", carla_port=2000, yolo_model="yolov8n.pt"):
        # CARLA 客户端（适配 0.9.11）
        self.client = carla.Client(carla_host, carla_port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_lib = self.world.get_blueprint_library()
        self.vehicle = None
        self.camera = None
        self.camera_img = None

        # YOLOv8 模型（适配 8.0.151）
        self.yolo = YOLO(yolo_model)
        self.detect_classes = [0, 2, 3, 5, 7]  # 行人、汽车、摩托、公交、卡车
        self.conf_threshold = 0.5
        self.detections = []

        # Pygame 可视化（确保中文正常）
        pygame.init()
        self.screen_w, self.screen_h = 1024, 768
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("CARLA 0.9.11 目标检测")
        self.font = pygame.font.SysFont(["SimHei", "WenQuanYi Micro Hei"], 20)

        # 类别颜色（红框、绿框等）
        self.class_colors = {
            "person": (255, 0, 0),    # 行人-红框
            "car": (0, 255, 0),       # 汽车-绿框
            "motorcycle": (0, 0, 255),# 摩托-蓝框
            "bus": (255, 255, 0),     # 公交-黄框
            "truck": (255, 0, 255)    # 卡车-洋红框
        }
        self.default_color = (128, 128, 128)

        # 运行控制
        self.running = True
        self.detect_thread = threading.Thread(target=self._run_detection, daemon=True)


    def spawn_actors(self):
        # 生成车辆（0.9.11 兼容）
        vehicle_bp = self.blueprint_lib.filter("model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        self.vehicle.set_autopilot(True)
        logging.info("车辆生成成功")

        # 生成摄像头（0.9.11 兼容）
        camera_bp = self.blueprint_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.screen_w))
        camera_bp.set_attribute("image_size_y", str(self.screen_h))
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda img: self._process_image(img))
        logging.info("摄像头启动成功")


    def _process_image(self, img):
        # 转换 CARLA 0.9.11 图像格式（BGRA -> RGB）
        img_array = np.frombuffer(img.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((img.height, img.width, 4))
        self.camera_img = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)


    def _run_detection(self):
        # 适配 ultralytics 8.0.151 的推理逻辑
        while self.running:
            if self.camera_img is not None:
                # 模型推理（兼容 8.0.151 的参数）
                results = self.yolo(
                    self.camera_img,
                    conf=self.conf_threshold,
                    classes=self.detect_classes,
                    device="cpu"  # 避免 GPU 兼容问题
                )
                # 解析检测结果
                self.detections = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        self.detections.append({
                            "bbox": (x1, y1, x2 - x1, y2 - y1),
                            "class": result.names[cls],
                            "confidence": conf
                        })
            time.sleep(0.01)


    def _draw_elements(self):
        # 绘制摄像头画面
        if self.camera_img is not None:
            img_surf = pygame.surfarray.make_surface(np.swapaxes(self.camera_img, 0, 1))
            self.screen.blit(img_surf, (0, 0))

        # 绘制检测框（红框、绿框等）
        for det in self.detections:
            x, y, w, h = det["bbox"]
            cls_name = det["class"]
            color = self.class_colors.get(cls_name, self.default_color)
            pygame.draw.rect(self.screen, color, (x, y, w, h), 2)

            # 绘制标签（无乱码）
            label = f"{cls_name} {det['confidence']:.2f}"
            label_surf = self.font.render(label, True, color)
            self.screen.blit(label_surf, (x, max(0, y - 20)))


    def run(self):
        try:
            self.spawn_actors()
            self.detect_thread.start()
            clock = pygame.time.Clock()

            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

                self._draw_elements()
                pygame.display.flip()
                clock.tick(30)

        finally:
            # 清理资源（0.9.11 兼容）
            if self.camera:
                self.camera.stop()
                self.camera.destroy()
            if self.vehicle:
                self.vehicle.destroy()
            pygame.quit()
            logging.info("程序退出，资源已清理")


if __name__ == "__main__":
    detector = CarlaDetector()
    detector.run()