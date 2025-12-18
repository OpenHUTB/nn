"""
事故检测器核心类：支持接收命令行参数，传递多语言配置
"""
import sys
import cv2
from ultralytics import YOLO
from config import (
    YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, ACCIDENT_CLASSES,
    MIN_VEHICLE_COUNT, PERSON_VEHICLE_CONTACT,
    RESIZE_WIDTH, RESIZE_HEIGHT, DEFAULT_DETECTION_SOURCE
)
from core.process import process_box_coords, draw_annotations

class AccidentDetector:
    def __init__(self):
        """初始化检测器，加载YOLOv8模型"""
        self.model = None
        self.accident_detected = False
        self._load_model()

    def _load_model(self):
        """私有方法：加载模型，包含重试逻辑"""
        try:
            print("🔄 正在加载YOLOv8模型（首次运行会自动下载）...")
            self.model = YOLO(YOLO_MODEL_PATH)
            print("✅ YOLOv8模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            # 重试加载模型
            try:
                print("🔄 尝试重新下载模型...")
                self.model = YOLO("yolov8n.pt")
                print("✅ 模型重新加载成功")
            except Exception as e2:
                print(f"❌ 模型重新加载失败：{e2}")
                sys.exit(1)

    def detect_frame(self, frame, language="zh"):  # 新增：接收语言参数
        """处理单帧，返回标注后的帧和是否检测到事故（支持多语言）"""
        detected_objects = []
        try:
            # 缩放帧提升速度
            frame_resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            # YOLOv8推理
            results = self.model(frame_resized, conf=CONFIDENCE_THRESHOLD)

            # 解析检测结果
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        if not hasattr(box, 'cls') or box.cls is None:
                            continue
                        cls_idx = int(box.cls[0])
                        if cls_idx in ACCIDENT_CLASSES:
                            cls_name = self.model.names[cls_idx]
                            # 处理坐标
                            scale_x = frame.shape[1] / RESIZE_WIDTH
                            scale_y = frame.shape[0] / RESIZE_HEIGHT
                            x1, y1, x2, y2 = process_box_coords(box, scale_x, scale_y)
                            detected_objects.append((cls_name, x1, y1, x2, y2))

            # 判断事故
            person_count = sum(1 for obj in detected_objects if obj[0] == "person")
            vehicle_count = sum(1 for obj in detected_objects if obj[0] in ["car", "truck"])
            is_accident = (vehicle_count >= MIN_VEHICLE_COUNT) or (person_count >= 1 and vehicle_count >= 1 and PERSON_VEHICLE_CONTACT)
            self.accident_detected = is_accident

            # 绘制标注（新增：传递语言参数）
            frame = draw_annotations(frame, detected_objects, is_accident, language)
        except Exception as e:
            print(f"⚠️ 帧处理出现小错误：{e}，继续运行...")
        return frame, self.accident_detected

    def run_detection(self, source=None, language="zh"):  # 新增：接收检测源、语言参数
        """启动检测流程（支持命令行参数，包含容错逻辑）"""
        # 确定检测源：命令行指定优先，否则用默认值
        detection_source = source if source is not None else DEFAULT_DETECTION_SOURCE
        # 处理检测源类型：数字→摄像头（整数），否则→视频路径（字符串）
        if isinstance(detection_source, str) and detection_source.isdigit():
            detection_source = int(detection_source)

        # 多次尝试打开检测源
        cap = None
        for i in range(3):
            cap = cv2.VideoCapture(detection_source)
            if cap.isOpened():
                break
            print(f"⚠️ 第{i+1}次打开检测源失败，重试中...")
            cv2.waitKey(1000)

        if not cap or not cap.isOpened():
            print(f"❌ 无法打开检测源：{detection_source}")
            # 强制切换为默认摄像头
            print("🔄 强制切换为电脑摄像头...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ 摄像头也无法打开，请检查设备")
                sys.exit(1)

        # 新增：根据语言显示提示信息
        start_msg = {
            "zh": "✅ 检测源打开成功，开始实时检测（按Q/ESC键退出）",
            "en": "✅ Detection source opened successfully, start real-time detection (press Q/ESC to exit)"
        }[language]
        tip_msg = {
            "zh": "💡 提示：检测到2辆车或行人和车辆同时出现时，显示红色警告",
            "en": "💡 Tip: Red warning appears when 2 vehicles or person-vehicle contact is detected"
        }[language]
        print(start_msg)
        print(tip_msg)

        # 逐帧处理
        while True:
            ret, frame = cap.read()
            if not ret:
                end_msg = {"zh": "🔚 视频/摄像头流结束", "en": "🔚 Video/camera stream ended"}[language]
                print(end_msg)
                break

            # 新增：传递语言参数给detect_frame
            frame, _ = self.detect_frame(frame, language)
            window_title = {"zh": "驾驶事故检测（按Q退出）", "en": "Driving Accident Detection (press Q to exit)"}[language]
            cv2.imshow(window_title, frame)

            # 退出逻辑
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                exit_msg = {"zh": "🛑 用户手动退出", "en": "🛑 User exited manually"}[language]
                print(exit_msg)
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        # 新增：多语言检测总结
        summary_title = {"zh": "\n📊 检测总结：是否检测到事故 → ", "en": "\n📊 Detection Summary: Accident Detected → "}[language]
        accident_status = {"zh": "✅ 是", "en": "✅ Yes"}[language] if self.accident_detected else {"zh": "❌ 否", "en": "❌ No"}[language]
        print(f"{summary_title}{accident_status}")

# 供外部导入的类
__all__ = ["AccidentDetector"]
