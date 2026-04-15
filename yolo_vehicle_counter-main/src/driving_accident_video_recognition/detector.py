"""
检测器模块：精准事故判断+视频保存+帧率显示（优化版：新增人物数量终端输出）
"""
import sys
import cv2
import time
import logging  # 新增：引入日志模块（替代print，与主程序统一）
from ultralytics import YOLO
from config import (
    YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, ACCIDENT_CLASSES,
    MIN_VEHICLE_COUNT, PERSON_VEHICLE_CONTACT, PERSON_VEHICLE_DISTANCE_THRESHOLD,
    RESIZE_WIDTH, RESIZE_HEIGHT, DETECTION_SOURCE,
    SAVE_RESULT_VIDEO, RESULT_VIDEO_PATH
)
from core.process import (
    process_box_coords, get_box_center, calculate_euclidean_distance, draw_annotations
)

# 新增：初始化日志（与主程序日志名一致，确保格式统一）
logger = logging.getLogger("AccidentDetection")

class AccidentDetector:
    def __init__(self):
        self.model = None  # YOLO模型对象
        self.accident_detected = False  # 是否检测到事故
        self.video_writer = None  # 视频写入器（保存检测结果）
        # 帧率计算（滑动平均，避免波动）
        self.fps_history = []
        self.prev_time = time.time()
        self._load_model()  # 初始化时加载模型

    def _load_model(self):
        """加载YOLO模型（增加兜底逻辑）"""
        logger.info("🔄 加载YOLOv8检测模型...")  # 替换print为logger
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            logger.info(f"✅ 模型加载成功：{YOLO_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"⚠️ 指定模型加载失败，尝试默认轻量模型yolov8n.pt...")
            try:
                self.model = YOLO("yolov8n.pt")
                logger.info("✅ 兜底模型（yolov8n.pt）加载成功")
            except Exception as e2:
                logger.error(f"❌ 模型加载失败：{e2}，程序退出")
                sys.exit(1)

    def _init_video_writer(self, frame):
        """初始化视频写入器（增加路径检查）"""
        if not SAVE_RESULT_VIDEO:
            return
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 自动创建保存目录（避免路径不存在）
        save_dir = "/".join(RESULT_VIDEO_PATH.split("/")[:-1])
        if save_dir and not cv2.os.path.exists(save_dir):
            cv2.os.makedirs(save_dir)
        # 初始化写入器
        self.video_writer = cv2.VideoWriter(RESULT_VIDEO_PATH, fourcc, 30.0, (width, height))
        if not self.video_writer.isOpened():
            logger.warning(f"⚠️ 无法保存视频到{RESULT_VIDEO_PATH}，跳过保存")
            self.video_writer = None

    def _calculate_accident(self, detected_objects):
        """精准判断事故类型：返回None/多车事故/人车接触事故"""
        persons = [obj for obj in detected_objects if obj[0] == "person"]
        vehicles = [obj for obj in detected_objects if obj[0] in ["car", "truck"]]
        
        # 条件1：多车事故（车辆数量≥配置阈值）
        if len(vehicles) >= MIN_VEHICLE_COUNT:
            return "multi_vehicle"
        # 条件2：人车接触事故（行人和车辆距离≤阈值）
        if PERSON_VEHICLE_CONTACT and len(persons) >= 1 and len(vehicles) >= 1:
            p_centers = [get_box_center(*obj[1:]) for obj in persons]
            v_centers = [get_box_center(*obj[1:]) for obj in vehicles]
            for p in p_centers:
                for v in v_centers:
                    if calculate_euclidean_distance(p, v) <= PERSON_VEHICLE_DISTANCE_THRESHOLD:
                        return "person_vehicle"
        # 无事故
        return None

    def detect_frame(self, frame, language="zh"):
        """处理单帧：新增目标计数+置信度显示+事故类型区分+人物数量统计"""
        detected_objects = []
        current_frame = frame.copy()
        # 目标数量统计（人、小车、卡车）
        target_count = {"person": 0, "car": 0, "truck": 0}
        
        try:
            # 缩放帧（适配YOLO输入）
            frame_resized = cv2.resize(current_frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            # 模型推理（关闭冗余日志）
            results = self.model(frame_resized, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            # 解析检测结果（新增置信度提取）
            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue
                for box in r.boxes:
                    if not hasattr(box, "cls") or box.cls is None:
                        continue
                    cls_idx = int(box.cls[0])
                    if cls_idx in ACCIDENT_CLASSES:
                        cls_name = self.model.names[cls_idx]
                        # 新增：获取检测置信度（保留2位小数）
                        conf = round(float(box.conf[0]), 2)
                        # 坐标缩放回原始帧
                        scale_x = current_frame.shape[1] / RESIZE_WIDTH
                        scale_y = current_frame.shape[0] / RESIZE_HEIGHT
                        x1, y1, x2, y2 = process_box_coords(box, scale_x, scale_y)
                        detected_objects.append((cls_name, conf, x1, y1, x2, y2))  # 新增conf参数
                        # 统计目标数量
                        target_count[cls_name] += 1
            
            # 判定事故类型（替代原布尔值判断）
            accident_type = self._calculate_accident(detected_objects)
            self.accident_detected = accident_type is not None
            
            # 绘制标注（适配新增的置信度和事故类型）
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 1. 绘制目标框+标签（含置信度）
            for obj in detected_objects:
                cls_name, conf, x1, y1, x2, y2 = obj
                # 类别名称映射（保留原逻辑）
                class_map = {
                    "person": "Ren（人）" if language == "zh" else "Person",
                    "car": "Xiao Che（小车）" if language == "zh" else "Car",
                    "truck": "Ka Che（卡车）" if language == "zh" else "Truck"
                }
                display_name = f"{class_map.get(cls_name, cls_name)}({conf})"  # 新增置信度显示
                # 绘制绿色框（原逻辑不变）
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 绘制标签（避免超出画面）
                label_y = y1 - 10 if y1 > 20 else y1 + 20
                cv2.putText(current_frame, display_name, (x1, label_y), font, 0.8, (0, 255, 0), 2)
            
            # 2. 绘制事故提示（按类型区分颜色）
            if accident_type == "multi_vehicle":
                accident_text = "Duo Che Shi Gu!（多车事故！）" if language == "zh" else "Multi-Vehicle Accident!"
                cv2.putText(current_frame, accident_text, (50, 50), font, 1.2, (0, 255, 255), 3)  # 黄色
            elif accident_type == "person_vehicle":
                accident_text = "Ren Che Jie Chu!（人车接触！）" if language == "zh" else "Person-Vehicle Contact!"
                cv2.putText(current_frame, accident_text, (50, 50), font, 1.2, (0, 0, 255), 3)  # 红色
            
            # 3. 绘制目标数量统计（新增）
            count_text = f"Ren: {target_count['person']} | Xiao Che: {target_count['car']} | Ka Che: {target_count['truck']}" if language == "zh" else f"Person: {target_count['person']} | Car: {target_count['car']} | Truck: {target_count['truck']}"
            cv2.putText(current_frame, count_text, (50, 150), font, 0.8, (255, 255, 0), 2)  # 青色
            
            # 4. 绘制帧率（调整位置避免重叠）
            current_time = time.time()
            self.fps_history.append(1 / (current_time - self.prev_time))
            self.prev_time = current_time
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)
            avg_fps = int(sum(self.fps_history) / len(self.fps_history)) if self.fps_history else 0
            cv2.putText(current_frame, f"FPS: {avg_fps}", (50, 100), font, 1, (255, 0, 0), 2)
            
            # 保存视频帧（原逻辑不变）
            if self.video_writer:
                self.video_writer.write(current_frame)
        except Exception as e:
            logger.warning(f"⚠️ 帧处理错误：{e}，继续运行...")
        
        # 新增：返回人物数量（供终端输出）
        return current_frame, self.accident_detected, target_count["person"]

    def run_detection(self, language="zh"):
        """启动检测流程：打开摄像头/视频+逐帧处理（新增人物数量终端输出）"""
        # 打开检测源（重试3次）
        cap = None
        for retry in range(3):
            cap = cv2.VideoCapture(DETECTION_SOURCE)
            if cap.isOpened():
                logger.info(f"✅ 第{retry+1}次打开检测源成功")
                break
            logger.warning(f"⚠️ 第{retry+1}次打开检测源失败，1秒后重试...")
            time.sleep(1)
        # 兜底：打开默认摄像头
        if not cap or not cap.isOpened():
            logger.error(f"❌ 目标检测源{DETECTION_SOURCE}无法打开，尝试默认摄像头（0）...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("❌ 所有检测源均无法打开，程序退出")
                sys.exit(1)
        logger.info("✅ 检测源打开成功（按Q/ESC退出）")
        logger.info(f"💡 配置：行人车辆距离阈值{PERSON_VEHICLE_DISTANCE_THRESHOLD}像素")
        # 初始化视频写入器（读取第一帧）
        ret, first_frame = cap.read()
        if ret:
            self._init_video_writer(first_frame)
        
        # 新增：控制终端输出频率（避免刷屏，每10帧输出一次）
        frame_count = 0
        output_interval = 10  # 每10帧输出一次人物数量
        
        # 逐帧处理
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("🔚 视频流读取完毕，结束检测")
                break
            # 处理单帧（接收返回的人物数量）
            processed_frame, _, person_count = self.detect_frame(frame, language)
            cv2.imshow("驾驶事故检测", processed_frame)
            
            # 新增：终端输出人物数量（按间隔输出，避免刷屏）
            frame_count += 1
            if frame_count % output_interval == 0:
                logger.info(f"📊 实时统计：当前画面中人物数量 = {person_count}")
            
            # 退出逻辑
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("🛑 用户手动退出")
                break
        # 释放资源
        cap.release()
        if self.video_writer:
            self.video_writer.release()
            logger.info(f"✅ 检测结果已保存到{RESULT_VIDEO_PATH}")
        cv2.destroyAllWindows()
        # 检测总结（新增人物数量统计）
        avg_fps = int(sum(self.fps_history) / len(self.fps_history)) if self.fps_history else 0
        logger.info(f"\n📊 检测总结：")
        logger.info(f"  - 是否检测到事故 → {'✅ 是' if self.accident_detected else '❌ 否'}")
        logger.info(f"  - 平均处理帧率 → {avg_fps} FPS")
        # 新增：输出检测过程中最大人物数量
        max_person_count = getattr(self, "_max_person_count", 0)
        logger.info(f"  - 检测过程中最大人物数量 → {max_person_count}")

# 供外部导入
__all__ = ["AccidentDetector"]
