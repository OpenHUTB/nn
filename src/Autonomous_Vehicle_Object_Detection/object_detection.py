import carla
import pygame
import time
import numpy as np
import cv2
import os
import logging
import random
from threading import Lock

# 精度日志配置（记录每帧精度指标）
logging.basicConfig(
    filename='detection_accuracy.log',
    level=logging.INFO,
    format='%(asctime)s - 帧号: %(frame)d - 召回率: %(recall).2f - 精确率: %(precision).2f - '
           '真实目标数: %(gt)d - 检测目标数: %(det)d - 正确检测数: %(correct)d'
)

class YOLODetector:
    def __init__(self, cfg_path, weights_path, classes_path):
        self._check_files(cfg_path, weights_path, classes_path)
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.conf_threshold = 0.3
        self.nms_threshold = 0.3

    def _check_files(self, *paths):
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"模型文件缺失：{path}（请放在代码目录）")

    def detect(self, image):
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
                if class_name in ["car", "truck", "bus", "person"]:
                    results.append({
                        "box": (x, y, x + w, y + h),
                        "class": class_name,
                        "confidence": round(confidences[i], 2)
                    })
        return results


class CarlaDetectionEvaluator:
    def __init__(self, yolo_detector):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(15.0)
        self.world = self.client.get_world()
        # 启用同步模式确保精度计算稳定
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20Hz更新
        self.world.apply_settings(settings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.detector = yolo_detector
        self.vehicle = None  # 仅用于挂载相机
        self.camera = None
        self.camera_bp = self.blueprint_library.find("sensor.camera.rgb")  # 相机配置
        self.display = None
        self.running = True
        self.image = None
        self.lock = Lock()
        self.frame_count = 0  # 帧计数器

    def _spawn_actors(self):
        """仅生成主车辆和相机（不生成目标，依赖外部生成）"""
        try:
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                print("错误：未找到车辆生成点，请检查CARLA地图是否加载")
                return False

            # 生成主车辆（用于挂载相机）
            main_vehicle_bp = self.blueprint_library.filter("model3")[0]
            self.vehicle = self.world.spawn_actor(main_vehicle_bp, spawn_points[0])
            self.vehicle.set_autopilot(True)
            print(f"主车辆生成（ID: {self.vehicle.id}，用于挂载相机）")

            # 配置相机参数
            self.camera_bp.set_attribute("image_size_x", "640")
            self.camera_bp.set_attribute("image_size_y", "480")
            self.camera_bp.set_attribute("fov", "90")
            self.camera_bp.set_attribute("sensor_tick", "0.2")  # 5Hz采样（流畅且不卡顿）
            camera_transform = carla.Transform(
                carla.Location(x=1.5, y=0, z=2.0),  # 车辆前方1.5米，高2米
                carla.Rotation(pitch=-2)  # 略微下倾
            )
            # 生成相机并挂载到主车辆
            self.camera = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.vehicle)
            self.camera.listen(self._on_image)  # 相机数据回调
            print("相机生成完成，开始接收图像...")

            # 提示用户依赖外部生成的目标
            print("提示：已禁用自动生成目标，请确保已通过外部代码生成车辆和行人")
            return True
        except Exception as e:
            print(f"生成主车辆/相机失败: {e}")
            return False

    def _get_ground_truth(self):
        """获取外部生成的真实目标（车辆/行人）的2D边界框"""
        gt = []
        if not self.camera:
            return gt

        # 相机参数（用于3D到2D投影）
        cam_transform = self.camera.get_transform()
        cam_loc = cam_transform.location
        cam_rot = cam_transform.rotation
        width, height = 640, 480
        fov = float(self.camera_bp.get_attribute("fov"))
        fx = width / (2 * np.tan(np.radians(fov) / 2))  # 焦距x
        fy = fx  # 焦距y（假设x=y）
        cx, cy = width / 2, height / 2  # 图像中心坐标

        # 遍历所有演员，筛选出车辆和行人（外部生成的目标）
        for actor in self.world.get_actors():
            is_vehicle = actor.type_id.startswith("vehicle.")
            is_walker = actor.type_id.startswith("walker.pedestrian.")
            if not (is_vehicle or is_walker):
                continue  # 只关注车辆和行人
            if self.vehicle and actor.id == self.vehicle.id:
                continue  # 排除主车辆

            # 获取目标的位置和边界框
            try:
                actor_loc = actor.get_transform().location
                actor_bbox = actor.bounding_box  # 3D边界框
            except:
                continue  # 跳过获取信息失败的目标

            # 只处理相机30米范围内的目标（太远的目标无需检测）
            if actor_loc.distance(cam_loc) > 30:
                continue

            # 将3D目标投影到2D图像，得到真实边界框
            bbox_2d = self._project_3d_to_2d(
                actor_loc, actor_bbox, cam_loc, cam_rot, fx, fy, cx, cy
            )
            if bbox_2d:
                x1, y1, x2, y2 = bbox_2d
                # 确保边界框在图像范围内
                if 0 <= x1 < x2 < width and 0 <= y1 < y2 < height:
                    cls = "car" if is_vehicle else "person"  # 区分车辆和行人
                    gt.append({"box": (x1, y1, x2, y2), "class": cls})

        return gt

    def _project_3d_to_2d(self, loc, bbox, cam_loc, cam_rot, fx, fy, cx, cy):
        """将3D世界坐标投影到2D图像坐标，计算真实边界框"""
        # 目标相对相机的位置（世界坐标系）
        dx = loc.x - cam_loc.x
        dy = loc.y - cam_loc.y
        dz = loc.z - cam_loc.z

        # 相机旋转角（转换为弧度）
        pitch = np.radians(cam_rot.pitch)
        yaw = np.radians(cam_rot.yaw)
        roll = np.radians(cam_rot.roll)

        # 旋转矩阵（CARLA世界坐标系 → 相机坐标系）
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        cos_r, sin_r = np.cos(roll), np.sin(roll)

        R = np.array([
            [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
            [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
            [-sin_p, cos_p * sin_r, cos_p * cos_r]
        ])

        # 转换到相机坐标系
        x_cam, y_cam, z_cam = R @ np.array([dx, dy, dz])

        # 过滤相机后方的目标（z_cam <= 0表示在相机后方，无需显示）
        if z_cam < 1.0:
            return None

        # 透视投影计算2D坐标
        px = (fx * x_cam) / z_cam + cx  # 图像x坐标
        py = (fy * y_cam) / z_cam + cy  # 图像y坐标

        # 根据3D尺寸计算2D边界框大小
        bbox_width = bbox.extent.x * 2  # 3D宽度（x方向）
        bbox_height = bbox.extent.z * 2  # 3D高度（z方向）
        # 投影到2D的宽度和高度
        w = int((fx * bbox_width) / z_cam)
        h = int((fy * bbox_height) / z_cam)

        # 计算边界框左上角和右下角坐标
        x1 = max(0, int(px - w//2))
        y1 = max(0, int(py - h//2))
        x2 = min(639, int(px + w//2))
        y2 = min(479, int(py + h//2))
        return (x1, y1, x2, y2)

    def _calculate_iou(self, box1, box2):
        """计算两个边界框的交并比（IoU），用于判断检测是否正确"""
        x1, y1, x2, y2 = box1  # 检测框
        x1g, y1g, x2g, y2g = box2  # 真实框

        # 计算交集区域
        inter_x1 = max(x1, x1g)
        inter_y1 = max(y1, y1g)
        inter_x2 = min(x2, x2g)
        inter_y2 = min(y2, y2g)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # 计算并集区域
        area1 = (x2 - x1) * (y2 - y1)  # 检测框面积
        area2 = (x2g - x1g) * (y2g - y1g)  # 真实框面积
        union_area = area1 + area2 - inter_area

        # 返回IoU（避免除以0）
        return inter_area / union_area if union_area > 0 else 0

    def _evaluate_precision(self, detections, ground_truth):
        """计算召回率（Recall）和精确率（Precision）"""
        gt_count = len(ground_truth)  # 真实目标总数
        det_count = len(detections)   # 检测到的目标总数
        correct_count = 0             # 正确检测的目标数
        matched_gt = [False] * gt_count  # 标记已匹配的真实目标（避免重复匹配）

        # 遍历所有检测结果，与真实目标匹配
        for det in detections:
            det_box = det["box"]
            det_cls = det["class"]
            for i, gt in enumerate(ground_truth):
                if matched_gt[i]:
                    continue  # 跳过已匹配的真实目标
                gt_box = gt["box"]
                gt_cls = gt["class"]
                # 类别相同且IoU>0.5 → 正确检测
                if det_cls == gt_cls and self._calculate_iou(det_box, gt_box) > 0.5:
                    correct_count += 1
                    matched_gt[i] = True
                    break  # 一个检测框只匹配一个真实目标

        # 计算指标（避免除以0）
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
        """相机回调：仅保存图像数据（避免线程冲突）"""
        with self.lock:
            if self.running:
                # 直接转换为RGB格式（BGRA→RGB）
                self.image = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                    (image.height, image.width, 4)
                )[:, :, [2, 1, 0]]  # 取R、G、B通道

    def run(self):
        """主循环：显示图像、执行检测、计算精度"""
        pygame.init()
        # 初始化Pygame窗口（启用硬件加速）
        self.display = pygame.display.set_mode(
            (640, 480), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.HWACCEL
        )
        pygame.display.set_caption("CARLA 目标检测（带精度验证）")

        # 初始化演员（主车辆+相机）
        if not self._spawn_actors():
            self.cleanup()
            return

        print("程序运行中！按ESC键退出...")
        print("画面说明：红色框=真实目标，绿色框=检测结果，左上角显示精度指标")

        try:
            while self.running:
                # 处理退出事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False

                # 同步CARLA世界更新（避免画面跳帧）
                self.world.tick()
                self.frame_count += 1

                # 处理图像和检测逻辑
                with self.lock:
                    if self.image is not None:
                        # 复制图像（避免线程冲突）
                        img = self.image.copy()
                        # 1. 执行目标检测
                        detections = self.detector.detect(img)
                        # 2. 获取真实目标（外部生成的车辆/行人）
                        ground_truth = self._get_ground_truth()
                        # 3. 计算精度指标
                        metrics = self._evaluate_precision(detections, ground_truth)
                        recall = metrics["recall"]
                        precision = metrics["precision"]
                        gt_count = metrics["gt_count"]
                        det_count = metrics["det_count"]
                        correct_count = metrics["correct_count"]

                        # 每10帧打印一次验证信息（终端输出，用于核对）
                        if self.frame_count % 10 == 0:
                            print(f"\n帧号: {self.frame_count}")
                            print(f"真实目标数: {gt_count} | 检测目标数: {det_count} | 正确检测数: {correct_count}")
                            print(f"召回率: {recall} | 精确率: {precision}")

                        # 记录日志（用于后续分析）
                        logging.info(
                            "",
                            extra={
                                'frame': self.frame_count,
                                'recall': recall,
                                'precision': precision,
                                'gt': gt_count,
                                'det': det_count,
                                'correct': correct_count
                            }
                        )

                        # 4. 绘制边界框和精度指标
                        # 绘制真实目标（红色框）
                        for gt in ground_truth:
                            x1, y1, x2, y2 = gt["box"]
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 红色
                            cv2.putText(
                                img, f"GT:{gt['class']}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
                            )

                        # 绘制检测结果（绿色框）
                        for det in detections:
                            x1, y1, x2, y2 = det["box"]
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 绿色
                            cv2.putText(
                                img, f"Det:{det['class']}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
                            )

                        # 显示精度指标（左上角）
                        cv2.putText(
                            img, f"Recall: {recall}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                        )
                        cv2.putText(
                            img, f"Precision: {precision}", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                        )

                        # 显示到Pygame窗口
                        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                        self.display.blit(surf, (0, 0))
                        pygame.display.flip()

                # 降低CPU占用
                time.sleep(0.01)
        except Exception as e:
            print(f"运行错误: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """安全清理资源（只销毁自己生成的主车辆和相机，保留用户生成的目标）"""
        self.running = False
        # 恢复CARLA异步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        # 销毁主车辆和相机
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
            print("相机已销毁")
        if self.vehicle:
            self.vehicle.destroy()
            print("主车辆已销毁")
        # 不销毁用户生成的目标（关键！）
        pygame.quit()
        print("程序退出，资源清理完成")
        print("精度日志已保存至 detection_accuracy.log")


if __name__ == "__main__":
    # YOLO模型文件路径（请确保这3个文件在代码目录）
    YOLO_CFG = "yolov3-tiny.cfg"
    YOLO_WEIGHTS = "yolov3-tiny.weights"
    YOLO_CLASSES = "coco.names"

    try:
        # 初始化YOLO检测器
        yolo = YOLODetector(YOLO_CFG, YOLO_WEIGHTS, YOLO_CLASSES)
        # 启动检测和精度评估
        evaluator = CarlaDetectionEvaluator(yolo)
        evaluator.run()
    except Exception as e:
        print(f"初始化失败: {e}")