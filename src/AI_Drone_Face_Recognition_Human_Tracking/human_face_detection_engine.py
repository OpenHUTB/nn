from ultralytics import YOLO
import cv2
import numpy as np
import os
from typing import Optional, Tuple, List


class FaceDatabase:
    """人脸数据库：存储已知人脸特征并提供匹配功能"""

    def __init__(self, face_features_dir: str = "face_features", threshold: float = 0.6):
        self.threshold = threshold  # 相似度阈值
        self.face_features = {}  # 格式: {name: feature}
        self.face_features_dir = face_features_dir
        os.makedirs(face_features_dir, exist_ok=True)

    @staticmethod
    def preprocess_face(face_roi: np.ndarray) -> Optional[np.ndarray]:
        """人脸预处理：灰度化、resize、直方图均衡化"""
        if face_roi.size == 0:
            return None
        # 转为灰度图
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        # 统一尺寸（便于特征提取）
        gray = cv2.resize(gray, (128, 128))
        # 直方图均衡化增强对比度
        gray = cv2.equalizeHist(gray)
        # 归一化
        feature = gray.flatten() / 255.0
        return feature

    @staticmethod
    def calculate_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """计算余弦相似度（越接近1越相似）"""
        if len(feature1) != len(feature2):
            return 0.0
        dot_product = np.dot(feature1, feature2)
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def add_face(self, name: str, face_roi: np.ndarray) -> bool:
        """添加人脸到数据库"""
        feature = self.preprocess_face(face_roi)
        if feature is None:
            return False
        self.face_features[name] = feature
        # 保存特征到文件
        np.save(os.path.join(self.face_features_dir, f"{name}.npy"), feature)
        return True

    def load_face(self, name: str) -> bool:
        """从文件加载人脸特征"""
        feature_path = os.path.join(self.face_features_dir, f"{name}.npy")
        if os.path.exists(feature_path):
            self.face_features[name] = np.load(feature_path)
            return True
        return False

    def match_face(self, face_roi: np.ndarray) -> Optional[str]:
        """匹配人脸，返回匹配到的名字（无则返回None）"""
        query_feature = self.preprocess_face(face_roi)
        if query_feature is None or not self.face_features:
            return None

        max_sim = 0.0
        matched_name = None
        for name, feature in self.face_features.items():
            sim = self.calculate_similarity(query_feature, feature)
            if sim > max_sim and sim > self.threshold:
                max_sim = sim
                matched_name = name
        return matched_name


class DetectionEngine:
    def __init__(self,
                 model_path: str = "yolov8n.pt",
                 conf_thres: float = 0.5,
                 track_thres: float = 0.4,
                 is_face_model: bool = False):
        """
        初始化检测引擎
        :param model_path: YOLO模型路径
        :param conf_thres: 置信度阈值
        :param track_thres: IOU阈值（跟踪/非极大值抑制）
        :param is_face_model: 是否使用人脸检测模型（yolov8n-face）
        """
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.track_thres = track_thres
        self.class_names = self.model.names

        # 类别ID配置
        self.human_class_id = 0  # COCO数据集的"person"类别ID
        self.face_class_id = 0 if is_face_model else None  # 人脸模型默认类别ID为0

    def detect(self, frame: np.ndarray) -> List:
        """
        检测帧中的目标
        :param frame: BGR格式的图像帧
        :return: YOLO检测结果列表
        """
        if frame is None or frame.size == 0:
            return []
        # 执行检测（关闭自动显示）
        results = self.model(
            frame,
            conf=self.conf_thres,
            iou=self.track_thres,
            show=False,
            verbose=False  # 关闭控制台日志
        )
        return results

    def get_largest_human(self, results: List) -> Optional[Tuple[int, int, int, int]]:
        """
        获取检测结果中面积最大的人体边界框
        :param results: YOLO检测结果
        :return: 最大人体的bbox (x1, y1, x2, y2)，无则返回None
        """
        largest_bbox = None
        max_area = 0

        for r in results:
            if not hasattr(r, 'boxes') or r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == self.human_class_id:
                    # 提取并校验坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area and area > 100:  # 过滤极小框（噪声）
                        max_area = area
                        largest_bbox = (x1, y1, x2, y2)
        return largest_bbox

    def match_faces(self, frame: np.ndarray, results: List, face_db: FaceDatabase) -> np.ndarray:
        """
        匹配检测到的人脸并绘制边界框
        :param frame: 原始图像帧
        :param results: YOLO检测结果
        :param face_db: 人脸数据库实例
        :return: 绘制了检测框的图像帧
        """
        frame_copy = frame.copy()  # 避免修改原始帧
        h, w = frame_copy.shape[:2]

        for r in results:
            if not hasattr(r, 'boxes') or r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # 提取并校验边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # 人脸检测分支
                if self.face_class_id is not None and cls == self.face_class_id:
                    face_roi = frame_copy[y1:y2, x1:x2]
                    name = face_db.match_face(face_roi) or "未知人脸"
                    frame_copy = self.draw_detection_box(frame_copy, (x1, y1, x2, y2), name, conf)
                # 人体检测分支
                elif cls == self.human_class_id:
                    frame_copy = self.draw_detection_box(frame_copy, (x1, y1, x2, y2), "人体", conf)
        return frame_copy

    @staticmethod
    def draw_detection_box(frame: np.ndarray,
                           bbox: Tuple[int, int, int, int],
                           label: str,
                           confidence: float) -> np.ndarray:
        """
        绘制检测框、标签和置信度
        :param frame: 图像帧
        :param bbox: 边界框 (x1, y1, x2, y2)
        :param label: 类别标签
        :param confidence: 置信度
        :return: 绘制后的图像帧
        """
        x1, y1, x2, y2 = bbox
        # 绘制边界框（绿色，线宽2）
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制标签背景（半透明）
        label_text = f"{label} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(label_text, font, 0.6, 1)
        text_w, text_h = text_size
        # 标签背景位置（避免越界）
        bg_x1, bg_y1 = x1, max(y1 - text_h - 10, 0)
        bg_x2, bg_y2 = x1 + text_w, y1
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), -1)

        # 绘制标签文字（白色）
        cv2.putText(frame, label_text, (x1, bg_y1 + text_h + 2),
                    font, 0.6, (255, 255, 255), 1)
        return frame


# ------------------------------
# 测试示例
# ------------------------------
if __name__ == "__main__":
    # 1. 初始化组件
    # 可选：使用人脸模型 "yolov8n-face.pt"（需先下载）
    # engine = DetectionEngine(model_path="yolov8n-face.pt", is_face_model=True)
    engine = DetectionEngine(model_path="yolov8n.pt", is_face_model=False)

    # 初始化人脸库并添加测试人脸
    face_db = FaceDatabase(threshold=0.6)

    # 2. 读取测试图像/视频
    cap = cv2.VideoCapture(0)  # 0为摄像头，也可替换为视频路径

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. 执行检测
        results = engine.detect(frame)

        # 4. 获取最大人体
        largest_human_bbox = engine.get_largest_human(results)
        if largest_human_bbox:
            print(f"检测到最大人体: {largest_human_bbox}")

        # 5. 人脸匹配与可视化
        frame_with_boxes = engine.match_faces(frame, results, face_db)

        # 6. 显示结果
        cv2.imshow("Detection Result", frame_with_boxes)

        # 退出按键：q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()