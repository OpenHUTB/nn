import carla
import cv2
import numpy as np
import pygame
import time
import os
import logging
from typing import List, Tuple, Dict

# 配置日志（记录每帧精度数据）
logging.basicConfig(
    filename='detection_accuracy.log',
    level=logging.INFO,
    format='%(asctime)s - 帧号: %(frame)d - 召回率: %(recall).2f - 精确率: %(precision).2f - 真实目标数: %(gt)d - 检测目标数: %(det)d - 正确检测数: %(correct)d'
)


class YOLOv3TinyDetector:
    def __init__(self, cfg_path: str, weights_path: str, classes_path: str):
        self._check_file_exists(cfg_path, weights_path, classes_path)
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with open(classes_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.output_layers = ["yolo_16", "yolo_23"]  # YOLOv3-Tiny输出层
        self.conf_threshold = 0.2  # 降低阈值以捕捉远处目标
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

        # 修复索引迭代问题
        indices_arr = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        indices = []
        if len(indices_arr) > 0:
            indices = indices_arr.flatten().tolist()

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
        # 相机参数（用于3D到2D投影）
        self.camera_intrinsic = None  # 内参矩阵
        self.camera_transform = None  # 外参（世界坐标→相机坐标）

    def _spawn_actors(self):
        # 生成车辆
        vehicle_bp = self.blueprint_library.filter("model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        self.vehicle.set_autopilot(True)

        # 生成相机并获取内参
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "480")
        camera_bp.set_attribute("fov", "100")  # 广角捕捉远处目标
        self.camera_transform = carla.Transform(
            carla.Location(x=0.8, y=0, z=1.2),
            carla.Rotation(pitch=-1)
        )
        self.camera = self.world.spawn_actor(
            camera_bp, self.camera_transform, attach_to=self.vehicle
        )
        # 计算相机内参（fx, fy, cx, cy）
        fov = float(camera_bp.get_attribute("fov"))
        width = int(camera_bp.get_attribute("image_size_x"))
        height = int(camera_bp.get_attribute("image_size_y"))
        fx = width / (2 * np.tan(np.radians(fov) / 2))
        fy = fx  # 假设x/y方向焦距相同
        cx, cy = width / 2, height / 2
        self.camera_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        self.camera.listen(self._process_image)

    def _get_ground_truth(self) -> List[Tuple[int, int, int, int, str]]:
        """获取场景中真实目标（车辆、行人）的2D边界框（作为Ground Truth）"""
        ground_truth = []
        # 遍历CARLA中的所有Actor，筛选车辆和行人
        for actor in self.world.get_actors():
            if actor.type_id.startswith("vehicle.") or actor.type_id.startswith("walker."):
                # 跳过自身车辆
                if actor.id == self.vehicle.id:
                    continue
                # 目标类别（车辆/行人）
                cls = "car" if actor.type_id.startswith("vehicle.") else "person"
                # 获取目标的3D位置和边界框
                actor_transform = actor.get_transform()
                actor_location = actor_transform.location
                actor_bbox = actor.bounding_box  # 3D边界框
                # 将3D目标投影到2D图像
                bbox_2d = self._project_3d_to_2d(actor_location, actor_bbox)
                if bbox_2d:  # 仅保留相机视野内的目标
                    x1, y1, x2, y2 = bbox_2d
                    ground_truth.append((x1, y1, x2, y2, cls))
        return ground_truth

    def _project_3d_to_2d(self, location: carla.Location, bbox: carla.BoundingBox) -> Tuple[int, int, int, int] or None:
        """将3D目标位置投影到2D图像平面，返回边界框"""
        # 转换世界坐标到相机坐标（相机为原点）
        camera_loc = self.camera_transform.location
        dx = location.x - camera_loc.x
        dy = location.y - camera_loc.y
        dz = location.z - camera_loc.z

        # 相机旋转（yaw/pitch/roll）转换为旋转矩阵
        yaw = np.radians(self.camera_transform.rotation.yaw)
        pitch = np.radians(self.camera_transform.rotation.pitch)
        roll = np.radians(self.camera_transform.rotation.roll)

        # 旋转计算（简化版，仅保留关键转换）
        x = dx * np.cos(yaw) + dy * np.sin(yaw)
        y = -dx * np.sin(yaw) + dy * np.cos(yaw)
        z = dz * np.cos(pitch) - y * np.sin(pitch)
        y = dz * np.sin(pitch) + y * np.cos(pitch)

        if z < 0.1:  # 目标在相机后方，忽略
            return None

        # 投影到图像平面（像素坐标）
        px = self.camera_intrinsic[0, 0] * x / z + self.camera_intrinsic[0, 2]
        py = self.camera_intrinsic[1, 1] * y / z + self.camera_intrinsic[1, 2]

        # 估算2D边界框大小（基于目标距离和3D尺寸）
        bbox_extent = bbox.extent  # 3D边界框半长
        w = int((self.camera_intrinsic[0, 0] * bbox_extent.x * 2) / z)  # 宽度
        h = int((self.camera_intrinsic[1, 1] * bbox_extent.z * 2) / z)  # 高度（z方向为竖直）

        # 确保边界框在图像范围内
        x1 = max(0, int(px - w / 2))
        y1 = max(0, int(py - h / 2))
        x2 = min(639, int(px + w / 2))
        y2 = min(479, int(py + h / 2))
        return (x1, y1, x2, y2)

    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的交并比（IoU），判断匹配程度"""
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        # 计算交集区域
        inter_x1 = max(x1, x1g)
        inter_y1 = max(y1, y1g)
        inter_x2 = min(x2, x2g)
        inter_y2 = min(y2, y2g)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # 计算并集区域
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _process_image(self, image: carla.Image):
        if not self.running:
            return

        # 1. 转换CARLA图像为OpenCV格式
        img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )[:, :, :3]  # 去除Alpha通道
        img = np.ascontiguousarray(img)

        # 2. 获取检测结果和真实目标
        try:
            detections = self.detector.detect(img)  # 模型检测结果：(x1,y1,x2,y2,cls,conf)
            ground_truth = self._get_ground_truth()  # 真实目标：(x1,y1,x2,y2,cls)
        except Exception as e:
            print(f"处理出错：{e}")
            return

        # 3. 计算精度指标（召回率、精确率）
        gt_count = len(ground_truth)  # 真实目标总数
        det_count = len(detections)  # 检测到的目标总数
        correct_count = 0  # 正确检测的目标数
        matched_gt = [False] * gt_count  # 标记已匹配的真实目标

        # 遍历检测结果，与真实目标匹配（IoU>0.5且类别一致视为正确）
        for det in detections:
            det_box = (det[0], det[1], det[2], det[3])
            det_cls = det[4]
            for i, gt in enumerate(ground_truth):
                if matched_gt[i]:
                    continue  # 跳过已匹配的真实目标
                gt_box = (gt[0], gt[1], gt[2], gt[3])
                gt_cls = gt[4]
                if det_cls == gt_cls and self._calculate_iou(det_box, gt_box) > 0.5:
                    correct_count += 1
                    matched_gt[i] = True
                    break

        # 计算召回率（正确检测/真实总数）和精确率（正确检测/检测总数）
        recall = correct_count / gt_count if gt_count > 0 else 0.0
        precision = correct_count / det_count if det_count > 0 else 0.0

        # 4. 记录日志
        logging.info(
            "",
            extra={
                'frame': image.frame,
                'recall': recall,
                'precision': precision,
                'gt': gt_count,
                'det': det_count,
                'correct': correct_count
            }
        )

        # 5. 可视化：绘制真实目标框、检测框和精度指标
        # 绘制真实目标（红色框）
        for gt in ground_truth:
            x1, y1, x2, y2, cls = gt
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色真实框
            cv2.putText(
                img, f"GT:{cls}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )

        # 绘制检测结果（绿色框）
        for det in detections:
            x1, y1, x2, y2, cls, conf = det
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色检测框
            cv2.putText(
                img, f"Det:{cls} {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # 显示精度指标（左上角）
        cv2.putText(
            img, f"召回率: {recall:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
        )
        cv2.putText(
            img, f"精确率: {precision:.2f}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
        )

        # 6. 转换为Pygame格式并显示
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1)).convert()
        self.display.blit(surf, (0, 0))
        pygame.display.flip()

    def run(self):
        # 初始化Pygame显示
        pygame.init()
        self.display = pygame.display.set_mode(
            (640, 480), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA 目标检测（带精度分析）")

        # 生成车辆和相机
        self._spawn_actors()
        print("程序启动成功！按ESC键退出...")
        print("精度数据已记录到 detection_accuracy.log")

        # 主循环
        try:
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
            print("程序退出，精度日志已保存")


if __name__ == "__main__":
    # 模型文件路径（确保与脚本同目录）
    YOLO_CFG = "yolov3-tiny.cfg"
    YOLO_WEIGHTS = "yolov3-tiny.weights"
    YOLO_CLASSES = "coco.names"

    # 初始化检测器并启动
    yolo_detector = YOLOv3TinyDetector(YOLO_CFG, YOLO_WEIGHTS, YOLO_CLASSES)
    carla_detector = CarlaObjectDetector(yolo_detector)
    carla_detector.run()