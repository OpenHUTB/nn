import carla
import pygame
import time
import numpy as np
import cv2
import os
import logging
import random
import shutil
from threading import Lock
from datetime import datetime

# -------------------------- 配置参数 --------------------------
# 数据集保存路径
DATASET_ROOT = "carla_dataset"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
# 类别映射（CARLA目标→YOLO类别ID）
CLASS_MAP = {"car": 0, "truck": 1, "bus": 2, "person": 3}
# 训练参数
TRAIN_EPOCHS = 50  # 训练轮次
TRAIN_BATCH_SIZE = 16  # 批次大小
# --------------------------------------------------------------

# 初始化数据集目录
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
logging.basicConfig(
    filename='dataset_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


class YOLODetector:
    def __init__(self, cfg_path=None, weights_path=None, classes_path="coco.names"):
        self.classes = self._load_classes(classes_path)
        # 支持加载自定义训练的模型
        if cfg_path and weights_path:
            self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            self.conf_threshold = 0.3
            self.nms_threshold = 0.3
        else:
            self.net = None  # 训练阶段可能不需要检测

    def _load_classes(self, classes_path):
        with open(classes_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def detect(self, image):
        if not self.net:
            return []
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (256, 256), swapRB=True, crop=False)
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
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        if len(indices) == 0:
            return []
        if isinstance(indices, tuple):
            indices = indices[0] if len(indices) > 0 else []
        indices = indices.flatten().tolist() if hasattr(indices, 'flatten') else indices

        results = []
        for i in indices:
            if 0 <= i < len(boxes):
                x, y, w, h = boxes[i]
                class_name = self.classes[class_ids[i]] if 0 <= class_ids[i] < len(self.classes) else "unknown"
                if class_name in CLASS_MAP:
                    results.append({
                        "box": (x, y, x + w, y + h),
                        "class": class_name,
                        "confidence": round(confidences[i], 2)
                    })
        return results


class DataCollector:
    """图片标签采集器：保存图像和YOLO格式标签"""
    def __init__(self):
        self.sample_count = 0  # 样本计数器

    def save_sample(self, image, ground_truth):
        """保存单帧数据（图像+标签）"""
        if not ground_truth:
            return  # 跳过无真实目标的帧

        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}_{self.sample_count}"
        img_path = os.path.join(IMAGES_DIR, f"{filename}.jpg")
        label_path = os.path.join(LABELS_DIR, f"{filename}.txt")

        # 保存图像
        cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # 生成YOLO格式标签（归一化坐标）
        height, width = image.shape[:2]
        with open(label_path, 'w') as f:
            for gt in ground_truth:
                cls = gt["class"]
                if cls not in CLASS_MAP:
                    continue
                cls_id = CLASS_MAP[cls]
                x1, y1, x2, y2 = gt["box"]
                # 计算中心点和宽高（归一化到0-1）
                cx = (x1 + x2) / 2 / width
                cy = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        self.sample_count += 1
        logging.info(f"保存样本 {self.sample_count}：{img_path} 和 {label_path}")
        return self.sample_count


class ModelTrainer:
    """模型训练器：基于YOLOv5微调模型"""
    def __init__(self, dataset_root=DATASET_ROOT):
        self.dataset_root = dataset_root
        self.yolov5_repo = "https://github.com/ultralytics/yolov5.git"
        self.requirements = "requirements.txt"

    def _prepare_env(self):
        """准备训练环境（安装YOLOv5和依赖）"""
        if not os.path.exists("yolov5"):
            os.system(f"git clone {self.yolov5_repo}")
        os.system(f"pip install -r yolov5/{self.requirements}")
        os.system(f"pip install ultralytics")

    def _split_dataset(self, train_ratio=0.8):
        """划分训练集和验证集"""
        images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
        random.shuffle(images)
        train_size = int(len(images) * train_ratio)

        # 创建划分目录
        for split in ["train", "val"]:
            os.makedirs(os.path.join(self.dataset_root, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_root, split, "labels"), exist_ok=True)

        # 复制文件
        for i, img in enumerate(images):
            prefix = img.split(".")[0]
            src_img = os.path.join(IMAGES_DIR, img)
            src_label = os.path.join(LABELS_DIR, f"{prefix}.txt")

            if i < train_size:
                split = "train"
            else:
                split = "val"

            dst_img = os.path.join(self.dataset_root, split, "images", img)
            dst_label = os.path.join(self.dataset_root, split, "labels", f"{prefix}.txt")
            shutil.copy(src_img, dst_img)
            shutil.copy(src_label, dst_label)

        logging.info(f"数据集划分完成：训练集 {train_size} 张，验证集 {len(images)-train_size} 张")

    def _generate_yaml(self):
        """生成YOLOv5所需的数据集配置文件"""
        yaml_content = f"""
train: {os.path.abspath(os.path.join(self.dataset_root, "train", "images"))}
val: {os.path.abspath(os.path.join(self.dataset_root, "val", "images"))}

nc: {len(CLASS_MAP)}
names: {list(CLASS_MAP.keys())}
        """
        with open("carla_dataset.yaml", "w") as f:
            f.write(yaml_content.strip())
        return "carla_dataset.yaml"

    def train(self):
        """启动模型训练"""
        if len(os.listdir(IMAGES_DIR)) < 10:
            raise ValueError("数据集样本不足（至少需要10张图像），请先采集更多数据")

        print("开始准备训练环境...")
        self._prepare_env()
        print("划分训练集和验证集...")
        self._split_dataset()
        yaml_path = self._generate_yaml()
        print(f"开始训练（{TRAIN_EPOCHS}轮）...")

        # 调用YOLOv5训练命令
        os.system(
            f"python yolov5/train.py "
            f"--img 640 "
            f"--batch {TRAIN_BATCH_SIZE} "
            f"--epochs {TRAIN_EPOCHS} "
            f"--data {yaml_path} "
            f"--weights yolov5s.pt "  # 基于小模型微调，速度快
            f"--project carla_train_results "
            f"--name exp"
        )
        print("训练完成！模型保存路径：carla_train_results/exp/weights/best.pt")
        logging.info("模型训练完成，最佳权重保存至 carla_train_results/exp/weights/best.pt")


class CarlaDetectionSystem:
    def __init__(self, detector):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(15.0)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.detector = detector
        self.data_collector = DataCollector()  # 数据采集器
        self.vehicle = None
        self.camera = None
        self.camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        self.display = None
        self.running = True
        self.image = None
        self.lock = Lock()
        self.frame_count = 0
        self.collect_data = False  # 是否开启数据采集（默认关闭）

    def _spawn_actors(self):
        try:
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                print("错误：未找到生成点")
                return False

            main_vehicle_bp = self.blueprint_library.filter("model3")[0]
            self.vehicle = self.world.spawn_actor(main_vehicle_bp, spawn_points[0])
            self.vehicle.set_autopilot(True)
            print(f"主车辆生成（ID: {self.vehicle.id}）")

            self.camera_bp.set_attribute("image_size_x", "640")
            self.camera_bp.set_attribute("image_size_y", "480")
            self.camera_bp.set_attribute("fov", "90")
            self.camera_bp.set_attribute("sensor_tick", "0.2")
            camera_transform = carla.Transform(
                carla.Location(x=1.5, y=0, z=2.0),
                carla.Rotation(pitch=-2)
            )
            self.camera = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.vehicle)
            self.camera.listen(self._on_image)
            print("相机生成完成")
            return True
        except Exception as e:
            print(f"生成失败: {e}")
            return False

    def _get_ground_truth(self):
        gt = []
        if not self.camera:
            return gt

        cam_transform = self.camera.get_transform()
        cam_loc = cam_transform.location
        cam_rot = cam_transform.rotation
        width, height = 640, 480
        fov = float(self.camera_bp.get_attribute("fov"))
        fx = width / (2 * np.tan(np.radians(fov) / 2))
        fy = fx
        cx, cy = width / 2, height / 2

        for actor in self.world.get_actors():
            is_vehicle = actor.type_id.startswith("vehicle.")
            is_walker = actor.type_id.startswith("walker.pedestrian.")
            if not (is_vehicle or is_walker):
                continue
            if self.vehicle and actor.id == self.vehicle.id:
                continue

            try:
                actor_loc = actor.get_transform().location
                actor_bbox = actor.bounding_box
            except:
                continue

            if actor_loc.distance(cam_loc) > 50:
                continue

            bbox_2d = self._project_3d_to_2d(
                actor_loc, actor_bbox, cam_loc, cam_rot, fx, fy, cx, cy
            )
            if bbox_2d:
                x1, y1, x2, y2 = bbox_2d
                cls = "car" if is_vehicle else "person"
                gt.append({"box": (x1, y1, x2, y2), "class": cls})
        return gt

    def _project_3d_to_2d(self, loc, bbox, cam_loc, cam_rot, fx, fy, cx, cy):
        dx = loc.x - cam_loc.x
        dy = loc.y - cam_loc.y
        dz = loc.z - cam_loc.z

        pitch = np.radians(cam_rot.pitch)
        yaw = np.radians(cam_rot.yaw)
        roll = np.radians(cam_rot.roll)

        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        cos_r, sin_r = np.cos(roll), np.sin(roll)

        R = np.array([
            [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
            [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
            [-sin_p, cos_p * sin_r, cos_p * cos_r]
        ])

        x_cam, y_cam, z_cam = R @ np.array([dx, dy, dz])
        if z_cam < 0.5:
            return None

        px = (fx * x_cam) / z_cam + cx
        py = (fy * y_cam) / z_cam + cy

        bbox_width = bbox.extent.x * 2
        bbox_height = bbox.extent.z * 2
        w = int((fx * bbox_width) / z_cam)
        h = int((fy * bbox_height) / z_cam)

        x1 = max(0, int(px - w//2))
        y1 = max(0, int(py - h//2))
        x2 = min(639, int(px + w//2))
        y2 = min(479, int(py + h//2))
        return (x1, y1, x2, y2)

    def _calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        inter_x1 = max(x1, x1g)
        inter_y1 = max(y1, y1g)
        inter_x2 = min(x2, x2g)
        inter_y2 = min(y2, y2g)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _evaluate_precision(self, detections, ground_truth):
        gt_count = len(ground_truth)
        det_count = len(detections)
        correct_count = 0
        matched_gt = [False] * gt_count

        for det in detections:
            det_box = det["box"]
            det_cls = det["class"]
            for i, gt in enumerate(ground_truth):
                if matched_gt[i]:
                    continue
                gt_box = gt["box"]
                gt_cls = gt["class"]
                if det_cls == gt_cls and self._calculate_iou(det_box, gt_box) > 0.3:
                    correct_count += 1
                    matched_gt[i] = True
                    break

        recall = correct_count / gt_count if gt_count > 0 else 0.0
        precision = correct_count / det_count if det_count > 0 else 0.0
        return {
            "recall": round(recall, 2),
            "precision": round(precision, 2),
            "gt_count": gt_count,
            "det_count": det_count,
            "correct_count": correct_count
        }

    def _on_image(self, image):
        with self.lock:
            if self.running:
                self.image = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                    (image.height, image.width, 4)
                )[:, :, [2, 1, 0]]

    def run(self):
        pygame.init()
        self.display = pygame.display.set_mode(
            (640, 480), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.HWACCEL
        )
        pygame.display.set_caption("CARLA 检测系统（带数据采集和训练）")

        if not self._spawn_actors():
            self.cleanup()
            return

        print("程序运行中！按键说明：")
        print(" - ESC: 退出程序")
        print(" - C: 开启/关闭数据采集（默认关闭）")
        print(" - T: 开始模型训练（需先采集足够数据）")

        trainer = ModelTrainer()  # 初始化训练器
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_c:
                            self.collect_data = not self.collect_data
                            status = "开启" if self.collect_data else "关闭"
                            print(f"数据采集已{status}")
                        elif event.key == pygame.K_t:
                            print("开始模型训练...")
                            try:
                                trainer.train()
                                print("训练完成！可加载新模型继续检测")
                            except Exception as e:
                                print(f"训练失败: {e}")

                self.world.tick()
                self.frame_count += 1

                with self.lock:
                    if self.image is not None:
                        img = self.image.copy()
                        detections = self.detector.detect(img)
                        ground_truth = self._get_ground_truth()

                        # 数据采集（按C开启后）
                        if self.collect_data and ground_truth:
                            self.data_collector.save_sample(img, ground_truth)

                        # 绘制真实目标（红框）
                        for gt in ground_truth:
                            x1, y1, x2, y2 = gt["box"]
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(
                                img, f"GT:{gt['class']}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                            )

                        # 绘制检测结果（绿框）
                        for det in detections:
                            x1, y1, x2, y2 = det["box"]
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                img, f"Det:{det['class']}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                            )

                        # 显示数据采集状态
                        cv2.putText(
                            img, f"采集: {'开' if self.collect_data else '关'}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                        )

                        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                        self.display.blit(surf, (0, 0))
                        pygame.display.flip()

                time.sleep(0.01)
        except Exception as e:
            print(f"运行错误: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.running = False
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        pygame.quit()
        print("程序退出")


if __name__ == "__main__":
    # 初始使用YOLOv3-tiny，训练后可替换为自定义模型
    YOLO_CFG = "yolov3-tiny.cfg"
    YOLO_WEIGHTS = "yolov3-tiny.weights"
    YOLO_CLASSES = "coco.names"

    try:
        yolo = YOLODetector(YOLO_CFG, YOLO_WEIGHTS, YOLO_CLASSES)
        system = CarlaDetectionSystem(yolo)
        system.run()
    except Exception as e:
        print(f"初始化失败: {e}")