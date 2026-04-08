# -*- coding: utf-8 -*-
import carla
import pygame
import logging
import numpy as np
import cv2
import time
import os
import shutil
import threading
from logging import StreamHandler
import sys
from datetime import datetime
from ultralytics import YOLO
import multiprocessing


# -------------------------- 配置参数 --------------------------
# -------------------------- 配置参数 --------------------------
class Config:
    # 检测参数（修正类别ID，适配训练后模型）
    DETECT_CONF = 0.5
    DETECT_CLASSES = [0, 1, 2, 3, 4]  # 自己训练模型的5个类别ID（0-4）
    IMG_WIDTH, IMG_HEIGHT = 640, 480

    # 其他参数不变...
    DATASET_DIR = "carla_yolo_dataset"
    IMG_DIR = os.path.join(DATASET_DIR, "images")
    LABEL_DIR = os.path.join(DATASET_DIR, "labels")
    AUTO_COLLECT = False
    COLLECT_INTERVAL = 0.5
    MIN_DETECTIONS = 1
    MODEL_PATH = "carla_yolo_results/train2/weights/best.pt"  # 替换为你的实际权重路径
    TRAIN_EPOCHS = 50
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    RESUME_TRAIN = False
    LAST_WEIGHTS = ""
    WORKERS = 4


# -------------------------- 日志初始化 --------------------------
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


class CarlaYoloTrainer:
    def __init__(self, carla_host="localhost", carla_port=2000):
        self.current_save_dir = ""  # 新增：记录当前训练的保存目录
        if multiprocessing.current_process().name == "MainProcess":
            pygame.init()
            auto_collect_status = "关闭" if not Config.AUTO_COLLECT else "开启"
            pygame.display.set_caption(f"目标检测与训练系统（自动采集{auto_collect_status}）")
            self.screen = pygame.display.set_mode((Config.IMG_WIDTH, Config.IMG_HEIGHT))
            self.font = pygame.font.SysFont(["SimHei", "WenQuanYi Micro Hei"], 22)
        else:
            self.screen = None
            self.font = None

        self.client = carla.Client(carla_host, carla_port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_lib = self.world.get_blueprint_library()
        self.camera = None
        self.camera_img = None
        self.vehicle = None

        self.model = YOLO(Config.MODEL_PATH)
        self.detections = []

        self.is_training = False
        self.train_thread = None
        self.pause_training = False
        self.best_map = 0.0

        self.class_colors = {
            "person": (255, 0, 0), "car": (0, 255, 0), "motorcycle": (0, 0, 255),
            "bus": (255, 255, 0), "truck": (255, 0, 255)
        }
        self.default_color = (128, 128, 128)

        self.sample_count = len(os.listdir(Config.IMG_DIR)) if os.path.exists(Config.IMG_DIR) else 0
        self.auto_collect = Config.AUTO_COLLECT
        self.last_collect_time = time.time()

        os.makedirs(Config.IMG_DIR, exist_ok=True)
        os.makedirs(Config.LABEL_DIR, exist_ok=True)

        self.running = True

    # -------------------------- 摄像头与车辆初始化 --------------------------
    def setup_camera(self):
        vehicle_bp = self.blueprint_lib.filter("model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            logging.error("CARLA无可用生成点，无法生成车辆")
            return

        self.vehicle = None
        for spawn_point in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                if self.vehicle:
                    logging.info(f"在生成点 {spawn_points.index(spawn_point)} 生成车辆")
                    break
            except RuntimeError:
                continue

        if not self.vehicle:
            logging.error("所有生成点被占用，请关闭交通实体或重启CARLA")
            self.running = False
            return

        self.vehicle.set_autopilot(True)
        logging.info("自动驾驶车辆已启动")

        camera_bp = self.blueprint_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(Config.IMG_WIDTH))
        camera_bp.set_attribute("image_size_y", str(Config.IMG_HEIGHT))
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera.listen(lambda img: self._process_image(img))
        logging.info("摄像头已挂载，等待图像数据")

    def _process_image(self, img):
        img_array = np.frombuffer(img.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((img.height, img.width, 4))
        self.camera_img = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)

    # -------------------------- 数据采集 --------------------------
    def collect_data(self, manual_trigger=False):
        if not manual_trigger:
            current_time = time.time()
            if len(self.detections) < Config.MIN_DETECTIONS:
                return
            if current_time - self.last_collect_time < Config.COLLECT_INTERVAL:
                return

        if self.camera_img is None:
            logging.warning("无图像数据，无法采集")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"sample_{self.sample_count}_{timestamp}.jpg"
        img_path = os.path.join(Config.IMG_DIR, img_name)
        label_path = os.path.join(Config.LABEL_DIR, img_name.replace(".jpg", ".txt"))

        cv2.imwrite(img_path, cv2.cvtColor(self.camera_img, cv2.COLOR_RGB2BGR))
        with open(label_path, "w", encoding="utf-8") as f:
            for det in self.detections:
                x, y, w, h = det["bbox"]
                x_center = (x + w / 2) / Config.IMG_WIDTH
                y_center = (y + h / 2) / Config.IMG_HEIGHT
                width = w / Config.IMG_WIDTH
                height = h / Config.IMG_HEIGHT
                cls_id = Config.DETECT_CLASSES.index(det["class_id"])
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        self.sample_count += 1
        self.last_collect_time = time.time()
        trigger_type = "手动" if manual_trigger else "自动"
        logging.info(f"[{trigger_type}采集] 样本 {self.sample_count}：{img_name}")

    # -------------------------- 模型训练（修复路径问题） --------------------------
    def train_model(self):
        if not os.listdir(Config.IMG_DIR):
            logging.error("数据集为空，无法训练")
            self.is_training = False
            return

        train_args = {
            "data": self._generate_data_yaml(),
            "batch": Config.BATCH_SIZE,
            "lr0": Config.LEARNING_RATE,
            "device": "0" if self._check_cuda() else "cpu",
            "imgsz": 640,
            "project": "carla_yolo_results",
            "name": "train",
            "save_period": 5,
            "resume": Config.RESUME_TRAIN and Config.LAST_WEIGHTS != "",
            "workers": Config.WORKERS
        }

        if Config.RESUME_TRAIN and Config.LAST_WEIGHTS:
            self.model = YOLO(Config.LAST_WEIGHTS)
            logging.info(f"从 {Config.LAST_WEIGHTS} 续训")

        logging.info("开始训练...")
        try:
            # 训练并记录实际保存目录
            train_results = self.model.train(**train_args, epochs=Config.TRAIN_EPOCHS)
            self.current_save_dir = train_results.save_dir  # 记录实际保存的目录（如train2）
            self.evaluate_model()
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error("训练失败：GPU显存不足！请将BATCH_SIZE改为2")
            else:
                logging.error(f"训练出错：{str(e)}")
        except Exception as e:
            logging.error(f"训练出错：{str(e)}")
        finally:
            self.is_training = False
            # 从实际保存目录获取权重
            if self.current_save_dir:
                final_weight = os.path.join(self.current_save_dir, "weights", "last.pt")
                Config.LAST_WEIGHTS = final_weight if os.path.exists(final_weight) else ""
                logging.info(f"训练结束，最终权重：{final_weight if os.path.exists(final_weight) else '未生成'}")
            else:
                logging.info("训练结束，未获取到保存目录")

    def _generate_data_yaml(self):
        yaml_path = os.path.join(Config.DATASET_DIR, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"""
train: {os.path.abspath(Config.IMG_DIR)}
val: {os.path.abspath(Config.IMG_DIR)}
nc: {len(Config.DETECT_CLASSES)}
names: {[self.model.names[c] for c in Config.DETECT_CLASSES]}
            """.strip())
        return yaml_path

    # -------------------------- 模型评测（修复路径问题） --------------------------
    def evaluate_model(self):
        if not os.listdir(Config.LABEL_DIR):
            logging.warning("无标签数据，无法评测")
            return

        # 手动指定权重路径（替换成你的实际路径）
        manual_weight_path = "carla_yolo_results/train2/weights/best.pt"
        if not os.path.exists(manual_weight_path):
            logging.warning(f"手动指定的权重文件不存在：{manual_weight_path}")
            return

        self.model = YOLO(manual_weight_path)
        results = self.model.val(
            data=self._generate_data_yaml(),
            imgsz=640,
            device="0" if self._check_cuda() else "cpu",
            workers=Config.WORKERS
        )

        # 获取当前评测的mAP值
        current_map = results.box.map50
        # 更新最佳mAP（如果当前值更高）
        if current_map > self.best_map:
            self.best_map = current_map
            logging.info(f"最佳mAP更新为：{self.best_map:.4f}")

        logging.info("=" * 50)
        logging.info(f"评测结果（mAP@0.5）：{current_map:.4f}")
        logging.info(f"Precision：{results.box.p.mean():.4f} | Recall：{results.box.r.mean():.4f}")
        logging.info("=" * 50)

    # -------------------------- 辅助功能 --------------------------
    def _check_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _draw_detections(self):
        if self.screen is None:
            return

        if self.camera_img is not None:
            img_surf = pygame.surfarray.make_surface(np.swapaxes(self.camera_img, 0, 1))
            self.screen.blit(img_surf, (0, 0))

        for det in self.detections:
            x, y, w, h = det["bbox"]
            cls_name = det["class"]
            color = self.class_colors.get(cls_name, self.default_color)
            pygame.draw.rect(self.screen, color, (x, y, w, h), 2)
            label = f"{cls_name} {det['confidence']:.2f}"
            self.screen.blit(self.font.render(label, True, color), (x, max(0, y - 25)))

        status = [
            f"样本数: {self.sample_count}",
            f"训练状态: {'训练中' if self.is_training else '未训练'}",
            f"自动采集: {'开启' if self.auto_collect else '关闭'}",
            f"最佳mAP: {self.best_map:.4f}"
        ]
        for i, text in enumerate(status):
            self.screen.blit(self.font.render(text, True, (255, 255, 255)), (10, 10 + i * 30))

    # -------------------------- 主循环 --------------------------
    def run(self):
        self.setup_camera()
        if not self.running:
            if self.screen is not None:
                pygame.quit()
            logging.info("程序退出")
            return

        clock = pygame.time.Clock() if self.screen is not None else None

        while self.running:
            if self.screen is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_c:
                            self.collect_data(manual_trigger=True)
                        elif event.key == pygame.K_a:
                            self.auto_collect = not self.auto_collect
                            status = "开启" if self.auto_collect else "关闭"
                            pygame.display.set_caption(f"目标检测与训练系统（自动采集{status}）")
                            logging.info(f"自动采集已{status}")
                        elif event.key == pygame.K_t and not self.is_training:
                            self.is_training = True
                            self.train_thread = threading.Thread(target=self.train_model, daemon=True)
                            self.train_thread.start()
                        elif event.key == pygame.K_p and self.is_training:
                            self.pause_training = not self.pause_training
                            logging.info(f"训练{'暂停' if self.pause_training else '继续'}")
                        elif event.key == pygame.K_v:
                            self.evaluate_model()
                        elif event.key == pygame.K_ESCAPE:
                            self.running = False
                            self.is_training = False

            if self.camera_img is not None and not self.pause_training and not self.is_training:
                results = self.model(
                    self.camera_img,
                    conf=Config.DETECT_CONF,
                    classes=Config.DETECT_CLASSES,
                    device="0" if self._check_cuda() else "cpu",
                    verbose=False
                )
                self.detections = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        self.detections.append({
                            "bbox": (x1, y1, x2 - x1, y2 - y1),
                            "class": result.names[cls_id],
                            "class_id": cls_id,
                            "confidence": float(box.conf[0])
                        })

                if self.auto_collect:
                    self.collect_data(manual_trigger=False)

            if self.screen is not None:
                self._draw_detections()
                pygame.display.flip()
                if clock is not None:
                    clock.tick(30)

        if self.camera and not self.camera.is_alive():
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        if self.screen is not None:
            pygame.quit()
        logging.info("程序退出")


if __name__ == "__main__":
    trainer = CarlaYoloTrainer()
    trainer.run()