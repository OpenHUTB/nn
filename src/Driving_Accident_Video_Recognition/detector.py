"""
检测器模块：精准事故判断+视频保存+帧率显示（无报错版）
"""
import sys
import cv2
import time
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
        print("🔄 加载YOLOv8检测模型...")
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            print(f"✅ 模型加载成功：{YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ 指定模型加载失败，尝试默认轻量模型yolov8n.pt...")
            try:
                self.model = YOLO("yolov8n.pt")
                print("✅ 兜底模型（yolov8n.pt）加载成功")
            except Exception as e2:
                print(f"❌ 模型加载失败：{e2}，程序退出")
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
            print(f"⚠️ 无法保存视频到{RESULT_VIDEO_PATH}，跳过保存")
            self.video_writer = None

    def _calculate_accident(self, detected_objects):
        """精准判断事故：多车/人车接触"""
        persons = [obj for obj in detected_objects if obj[0] == "person"]
        vehicles = [obj for obj in detected_objects if obj[0] in ["car", "truck"]]

        # 条件1：车辆数量≥配置阈值
        if len(vehicles) >= MIN_VEHICLE_COUNT:
            return True
        # 条件2：行人和车辆距离≤阈值
        if PERSON_VEHICLE_CONTACT and len(persons) >= 1 and len(vehicles) >= 1:
            p_centers = [get_box_center(*obj[1:]) for obj in persons]
            v_centers = [get_box_center(*obj[1:]) for obj in vehicles]
            for p in p_centers:
                for v in v_centers:
                    if calculate_euclidean_distance(p, v) <= PERSON_VEHICLE_DISTANCE_THRESHOLD:
                        return True
        return False

    def detect_frame(self, frame, language="zh"):
        """处理单帧：检测+标注+帧率计算"""
        detected_objects = []
        current_frame = frame.copy()

        try:
            # 缩放帧（适配YOLO输入）
            frame_resized = cv2.resize(current_frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            # 模型推理（关闭冗余日志）
            results = self.model(frame_resized, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # 解析检测结果
            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue
                for box in r.boxes:
                    if not hasattr(box, "cls") or box.cls is None:
                        continue
                    cls_idx = int(box.cls[0])
                    if cls_idx in ACCIDENT_CLASSES:
                        cls_name = self.model.names[cls_idx]
                        # 坐标缩放回原始帧
                        scale_x = current_frame.shape[1] / RESIZE_WIDTH
                        scale_y = current_frame.shape[0] / RESIZE_HEIGHT
                        x1, y1, x2, y2 = process_box_coords(box, scale_x, scale_y)
                        detected_objects.append((cls_name, x1, y1, x2, y2))

            # 判断事故
            self.accident_detected = self._calculate_accident(detected_objects)
            # 绘制标注
            current_frame = draw_annotations(current_frame, detected_objects, self.accident_detected, language)

            # 计算滑动平均帧率
            current_time = time.time()
            self.fps_history.append(1 / (current_time - self.prev_time))
            self.prev_time = current_time
            # 只保留最近10帧的帧率（避免波动）
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)
            avg_fps = int(sum(self.fps_history) / len(self.fps_history)) if self.fps_history else 0
            # 绘制帧率
            cv2.putText(current_frame, f"FPS: {avg_fps}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 保存视频帧
            if self.video_writer:
                self.video_writer.write(current_frame)

        except Exception as e:
            print(f"⚠️ 帧处理错误：{e}，继续运行...")

        return current_frame, self.accident_detected

    def run_detection(self, language="zh"):
        """启动检测流程：打开摄像头/视频+逐帧处理"""
        # 打开检测源（重试3次）
        cap = None
        for retry in range(3):
            cap = cv2.VideoCapture(DETECTION_SOURCE)
            if cap.isOpened():
                print(f"✅ 第{retry+1}次打开检测源成功")
                break
            print(f"⚠️ 第{retry+1}次打开检测源失败，1秒后重试...")
            time.sleep(1)

        # 兜底：打开默认摄像头
        if not cap or not cap.isOpened():
            print(f"❌ 目标检测源{DETECTION_SOURCE}无法打开，尝试默认摄像头（0）...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ 所有检测源均无法打开，程序退出")
                sys.exit(1)

        print("✅ 检测源打开成功（按Q/ESC退出）")
        print(f"💡 配置：行人车辆距离阈值{PERSON_VEHICLE_DISTANCE_THRESHOLD}像素")

        # 初始化视频写入器（读取第一帧）
        ret, first_frame = cap.read()
        if ret:
            self._init_video_writer(first_frame)

        # 逐帧处理
        while True:
            ret, frame = cap.read()
            if not ret:
                print("🔚 视频流读取完毕，结束检测")
                break

            # 处理单帧
            processed_frame, _ = self.detect_frame(frame, language)
            cv2.imshow("驾驶事故检测", processed_frame)

            # 退出逻辑
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("🛑 用户手动退出")
                break

        # 释放资源
        cap.release()
        if self.video_writer:
            self.video_writer.release()
            print(f"✅ 检测结果已保存到{RESULT_VIDEO_PATH}")
        cv2.destroyAllWindows()

        # 检测总结
        avg_fps = int(sum(self.fps_history) / len(self.fps_history)) if self.fps_history else 0
        print(f"\n📊 检测总结：")
        print(f"  - 是否检测到事故 → {'✅ 是' if self.accident_detected else '❌ 否'}")
        print(f"  - 平均处理帧率 → {avg_fps} FPS")


# 供外部导入
__all__ = ["AccidentDetector"]
