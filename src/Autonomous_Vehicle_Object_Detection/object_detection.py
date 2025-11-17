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
from sklearn.metrics import precision_score, recall_score, f1_score

# -------------------------- 配置参数（平衡效率与精度） --------------------------
class Config:
    # 检测参数
    DETECT_CONF = 0.5  # 检测置信度阈值（平衡召回率与误检）
    DETECT_CLASSES = [0, 2, 3, 5, 7]  # 行人、汽车、摩托、公交、卡车
    IMG_WIDTH, IMG_HEIGHT = 1024, 768  # 兼顾精度与速度的分辨率

    # 数据采集参数
    DATASET_DIR = "carla_yolo_dataset"  # 数据集根目录
    IMG_DIR = os.path.join(DATASET_DIR, "images")
    LABEL_DIR = os.path.join(DATASET_DIR, "labels")

    # 训练参数（效率与精度平衡）
    MODEL_PATH = "yolov8m.pt"  # 中等规模模型（比n大，比x小）
    TRAIN_EPOCHS = 50  # 基础训练轮次
    BATCH_SIZE = 8  # 根据GPU显存调整（8G显存推荐8）
    LEARNING_RATE = 0.001  # 适中学习率
    RESUME_TRAIN = False  # 是否续训
    LAST_WEIGHTS = ""  # 续训权重路径
    MIXED_PRECISION = True  # 混合精度训练（加速且不损失精度）

    # 评测参数
    EVAL_INTERVAL = 10  # 每10轮评测一次


# -------------------------- 日志与初始化 --------------------------
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
        # CARLA客户端（仅获取摄像头数据，不生成实体）
        self.client = carla.Client(carla_host, carla_port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_lib = self.world.get_blueprint_library()
        self.camera = None
        self.camera_img = None  # 摄像头图像（RGB）

        self.vehicle = None  # 新增：自动驾驶车辆

        # YOLO模型（检测+训练）
        self.model = YOLO(Config.MODEL_PATH)
        self.detections = []  # 检测结果缓存
        self.gt_labels = []  # 真实标签（用于评测）
        self.pred_labels = []  # 预测标签（用于评测）

        # 训练状态
        self.is_training = False
        self.train_thread = None
        self.pause_training = False
        self.best_map = 0.0  # 最佳mAP

        # 可视化
        pygame.init()
        self.screen = pygame.display.set_mode((Config.IMG_WIDTH, Config.IMG_HEIGHT))
        pygame.display.set_caption("目标检测与训练系统")
        self.font = pygame.font.SysFont(["SimHei", "WenQuanYi Micro Hei"], 22)

        # 类别颜色
        self.class_colors = {
            "person": (255, 0, 0), "car": (0, 255, 0), "motorcycle": (0, 0, 255),
            "bus": (255, 255, 0), "truck": (255, 0, 255)
        }
        self.default_color = (128, 128, 128)

        # 初始化数据集目录
        os.makedirs(Config.IMG_DIR, exist_ok=True)
        os.makedirs(Config.LABEL_DIR, exist_ok=True)
        self.sample_count = len(os.listdir(Config.IMG_DIR))  # 已有样本数

        # 运行控制
        self.running = True


    # -------------------------- 摄像头初始化（仅获取数据） --------------------------
    def setup_camera(self):
        """生成自动驾驶车辆 + 将摄像头挂载到车辆上（实现视角跟随）"""
        # 1. 生成自动驾驶车辆
        vehicle_bp = self.blueprint_lib.filter("model3")[0]  # 选择特斯拉Model3蓝图
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            logging.error("CARLA中无可用生成点，无法生成车辆")
            return
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        self.vehicle.set_autopilot(True)  # 开启自动驾驶
        logging.info("生成自动驾驶车辆并开启自动行驶")

        # 2. 将摄像头挂载到车辆车顶（跟随车辆移动）
        camera_bp = self.blueprint_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(Config.IMG_WIDTH))
        camera_bp.set_attribute("image_size_y", str(Config.IMG_HEIGHT))
        camera_bp.set_attribute("fov", "90")
        # 摄像头安装位置：车辆前方1.5米、上方2.4米（模拟真实自动驾驶视角）
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera.listen(lambda img: self._process_image(img))
        logging.info("摄像头已挂载到自动驾驶车辆，视角跟随车辆移动")


    def _process_image(self, img):
        """转换CARLA图像为RGB格式"""
        img_array = np.frombuffer(img.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((img.height, img.width, 4))  # BGRA
        self.camera_img = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)  # 转为RGB


    # -------------------------- 自动数据采集与标签生成 --------------------------
    def collect_data(self):
        """采集当前帧图像与检测标签（YOLO格式）"""
        if self.camera_img is None:
            logging.warning("无图像数据，无法采集")
            return

        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"sample_{self.sample_count}_{timestamp}.jpg"
        img_path = os.path.join(Config.IMG_DIR, img_name)
        label_path = os.path.join(Config.LABEL_DIR, img_name.replace(".jpg", ".txt"))

        # 保存图像
        cv2.imwrite(img_path, cv2.cvtColor(self.camera_img, cv2.COLOR_RGB2BGR))

        # 生成标签（YOLO格式：class x_center y_center width height（归一化））
        with open(label_path, "w", encoding="utf-8") as f:
            for det in self.detections:
                x, y, w, h = det["bbox"]
                # 归一化坐标
                x_center = (x + w/2) / Config.IMG_WIDTH
                y_center = (y + h/2) / Config.IMG_HEIGHT
                width = w / Config.IMG_WIDTH
                height = h / Config.IMG_HEIGHT
                cls_id = Config.DETECT_CLASSES.index(det["class_id"])  # 映射到0-4
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        self.sample_count += 1
        logging.info(f"已采集样本 {self.sample_count}：{img_name}")


    # -------------------------- 模型训练（支持断点续训） --------------------------
    def train_model(self):
        """训练线程（不阻塞检测可视化）"""
        if not os.listdir(Config.IMG_DIR):
            logging.error("数据集为空，无法训练，请先采集数据")
            self.is_training = False
            return

        # 准备训练配置
        train_args = {
            "data": self._generate_data_yaml(),  # 自动生成数据集配置
            "epochs": Config.TRAIN_EPOCHS,
            "batch": Config.BATCH_SIZE,
            "lr0": Config.LEARNING_RATE,
            "device": "0" if self._check_cuda() else "cpu",
            "imgsz": (Config.IMG_HEIGHT, Config.IMG_WIDTH),
            "mixed_precision": "fp16" if Config.MIXED_PRECISION else "None",
            "project": "carla_yolo_results",
            "name": "train",
            "save_period": 5,  # 每5轮保存一次权重
            "resume": Config.RESUME_TRAIN and Config.LAST_WEIGHTS != ""
        }

        # 加载续训权重
        if Config.RESUME_TRAIN and Config.LAST_WEIGHTS:
            self.model = YOLO(Config.LAST_WEIGHTS)
            logging.info(f"从 {Config.LAST_WEIGHTS} 继续训练")

        # 开始训练
        logging.info("开始训练...")
        try:
            for epoch in range(Config.TRAIN_EPOCHS):
                if not self.is_training:  # 外部终止
                    break
                if self.pause_training:  # 暂停
                    while self.pause_training and self.is_training:
                        time.sleep(1)
                # 单轮训练
                self.model.train(**train_args, epochs=epoch+1, resume=True)
                # 定期评测
                if (epoch + 1) % Config.EVAL_INTERVAL == 0:
                    self.evaluate_model()
        except Exception as e:
            logging.error(f"训练出错：{str(e)}")
        finally:
            self.is_training = False
            logging.info("训练结束，保存最终权重")
            final_weight = os.path.join("carla_yolo_results", "train", "weights", "last.pt")
            Config.LAST_WEIGHTS = final_weight
            logging.info(f"最终权重保存至：{final_weight}")


    def _generate_data_yaml(self):
        """自动生成YOLO训练所需的data.yaml"""
        yaml_path = os.path.join(Config.DATASET_DIR, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"""
train: {os.path.abspath(Config.IMG_DIR)}
val: {os.path.abspath(Config.IMG_DIR)}  # 简化：用训练集当验证集，实际应拆分
nc: {len(Config.DETECT_CLASSES)}
names: {[self.model.names[c] for c in Config.DETECT_CLASSES]}
            """.strip())
        return yaml_path


    # -------------------------- 检测精度评测 --------------------------
    def evaluate_model(self):
        """评测mAP、Precision、Recall等指标"""
        if not os.listdir(Config.LABEL_DIR):
            logging.warning("无标签数据，无法评测")
            return

        # 加载最新权重
        latest_weight = os.path.join("carla_yolo_results", "train", "weights", "last.pt")
        if os.path.exists(latest_weight):
            self.model = YOLO(latest_weight)

        # 运行验证
        results = self.model.val(
            data=self._generate_data_yaml(),
            imgsz=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
            device="0" if self._check_cuda() else "cpu"
        )

        # 输出关键指标
        logging.info("="*50)
        logging.info(f"评测结果（mAP@0.5）：{results.box.map50:.4f}")
        logging.info(f"Precision：{results.box.p:.4f} | Recall：{results.box.r:.4f}")
        logging.info("="*50)

        # 更新最佳模型
        if results.box.map50 > self.best_map:
            self.best_map = results.box.map50
            shutil.copy2(
                latest_weight,
                os.path.join("carla_yolo_results", "train", "weights", "best.pt")
            )
            logging.info(f"更新最佳模型（mAP@0.5：{self.best_map:.4f}）")


    # -------------------------- 辅助功能 --------------------------
    def _check_cuda(self):
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False


    def _draw_detections(self):
        """绘制检测结果与状态信息"""
        if self.camera_img is not None:
            # 绘制图像
            img_surf = pygame.surfarray.make_surface(np.swapaxes(self.camera_img, 0, 1))
            self.screen.blit(img_surf, (0, 0))

            # 绘制检测框
            for det in self.detections:
                x, y, w, h = det["bbox"]
                cls_name = det["class"]
                color = self.class_colors.get(cls_name, self.default_color)
                pygame.draw.rect(self.screen, color, (x, y, w, h), 2)
                label = f"{cls_name} {det['confidence']:.2f}"
                self.screen.blit(self.font.render(label, True, color), (x, max(0, y-25)))

        # 绘制状态文本
        status = [
            f"样本数: {self.sample_count}",
            f"训练状态: {'训练中' if self.is_training else '未训练'}",
            f"最佳mAP: {self.best_map:.4f}"
        ]
        for i, text in enumerate(status):
            self.screen.blit(self.font.render(text, True, (255, 255, 255)), (10, 10 + i*30))


    # -------------------------- 主循环 --------------------------
    def run(self):
        self.setup_camera()
        clock = pygame.time.Clock()

        while self.running:
            # 处理按键事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:  # 采集数据
                        self.collect_data()
                    elif event.key == pygame.K_t and not self.is_training:  # 开始训练
                        self.is_training = True
                        self.pause_training = False
                        self.train_thread = threading.Thread(target=self.train_model, daemon=True)
                        self.train_thread.start()
                    elif event.key == pygame.K_p and self.is_training:  # 暂停/继续训练
                        self.pause_training = not self.pause_training
                        state = "暂停" if self.pause_training else "继续"
                        logging.info(f"训练{state}")
                    elif event.key == pygame.K_v:  # 手动评测
                        self.evaluate_model()
                    elif event.key == pygame.K_ESCAPE:  # 退出
                        self.running = False
                        self.is_training = False  # 终止训练

            # 实时检测（训练时也保持检测）
            if self.camera_img is not None and not self.pause_training:
                results = self.model(
                    self.camera_img,
                    conf=Config.DETECT_CONF,
                    classes=Config.DETECT_CLASSES,
                    device="0" if self._check_cuda() else "cpu"
                )
                # 解析检测结果
                self.detections = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        self.detections.append({
                            "bbox": (x1, y1, x2-x1, y2-y1),
                            "class": result.names[cls_id],
                            "class_id": cls_id,
                            "confidence": float(box.conf[0])
                        })

            # 刷新画面
            self._draw_detections()
            pygame.display.flip()
            clock.tick(30)

        # 资源清理
        if self.camera and not self.camera.is_alive():
            self.camera.destroy()
        pygame.quit()
        logging.info("程序退出")


if __name__ == "__main__":
    trainer = CarlaYoloTrainer()
    trainer.run()