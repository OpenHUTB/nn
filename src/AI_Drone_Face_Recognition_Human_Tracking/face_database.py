from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from typing import Optional, Tuple, List, Dict


class FaceDatabase:
    # 【复用之前的FaceDatabase类代码，此处省略（保持不变）】
    def __init__(self, data_dir: str = "face_database", threshold: float = 0.6):
        self.data_dir = data_dir
        self.feat_dir = os.path.join(data_dir, "features")
        self.meta_dir = os.path.join(data_dir, "metadata")
        self.threshold = threshold
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.feat_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        self.face_features: Dict[str, np.ndarray] = {}
        self.face_metadata: Dict[str, dict] = {}

    @staticmethod
    def preprocess_face(face_roi: np.ndarray) -> Optional[np.ndarray]:
        if face_roi.size == 0:
            return None
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        gray = cv2.equalizeHist(gray)
        feature = gray.flatten() / 255.0
        return feature

    def save_face(self, name: str, face_roi: np.ndarray, remark: str = "", overwrite: bool = True) -> bool:
        feature = self.preprocess_face(face_roi)
        if feature is None:
            print(f"❌ 人脸预处理失败，无法保存{name}")
            return False
        feat_path = os.path.join(self.feat_dir, f"{name}.npy")
        meta_path = os.path.join(self.meta_dir, f"{name}.json")
        if os.path.exists(feat_path) and not overwrite:
            print(f"⚠️ {name}已存在，跳过保存（如需覆盖请设置overwrite=True）")
            return False
        np.save(feat_path, feature)
        metadata = {
            "name": name,
            "remark": remark,
            "feature_shape": feature.shape,
            "add_time": str(np.datetime64('now')),
            "version": "1.0"
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        self.face_features[name] = feature
        self.face_metadata[name] = metadata
        print(f"✅ {name}人脸数据保存成功")
        return True

    def load_face(self, name: str) -> bool:
        feat_path = os.path.join(self.feat_dir, f"{name}.npy")
        meta_path = os.path.join(self.meta_dir, f"{name}.json")
        if not os.path.exists(feat_path) or not os.path.exists(meta_path):
            print(f"❌ {name}人脸数据不存在")
            return False
        feature = np.load(feat_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.face_features[name] = feature
        self.face_metadata[name] = metadata
        print(f"✅ 加载{name}人脸数据成功")
        return True

    def load_all_faces(self) -> int:
        loaded_count = 0
        for feat_file in os.listdir(self.feat_dir):
            if not feat_file.endswith(".npy"):
                continue
            name = os.path.splitext(feat_file)[0]
            if self.load_face(name):
                loaded_count += 1
        print(f"📊 批量加载完成，共加载{loaded_count}个人脸数据")
        return loaded_count

    def delete_face(self, name: str) -> bool:
        feat_path = os.path.join(self.feat_dir, f"{name}.npy")
        meta_path = os.path.join(self.meta_dir, f"{name}.json")
        for path in [feat_path, meta_path]:
            if os.path.exists(path):
                os.remove(path)
        if name in self.face_features:
            del self.face_features[name]
        if name in self.face_metadata:
            del self.face_metadata[name]
        print(f"🗑️ {name}人脸数据已删除")
        return True

    def list_faces(self) -> List[str]:
        face_list = [os.path.splitext(f)[0] for f in os.listdir(self.feat_dir) if f.endswith(".npy")]
        print(f"📋 已保存的人脸列表：{face_list}")
        return face_list

    @staticmethod
    def calculate_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
        if len(feature1) != len(feature2):
            return 0.0
        dot = np.dot(feature1, feature2)
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        return dot / (norm1 * norm2) if (norm1 * norm2) != 0 else 0.0

    def match_face(self, face_roi: np.ndarray) -> Optional[str]:
        query_feat = self.preprocess_face(face_roi)
        if query_feat is None or not self.face_features:
            return None
        max_sim = 0.0
        matched_name = None
        for name, feat in self.face_features.items():
            sim = self.calculate_similarity(query_feat, feat)
            if sim > max_sim and sim > self.threshold:
                max_sim = sim
                matched_name = name
        return matched_name


class DetectionEngine:
    # 【复用之前的DetectionEngine类代码，此处省略（保持不变）】
    def __init__(self,
                 model_path: str = "yolov8n.pt",
                 conf_thres: float = 0.5,
                 track_thres: float = 0.4,
                 is_face_model: bool = False):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.track_thres = track_thres
        self.class_names = self.model.names
        self.human_class_id = 0
        self.face_class_id = 0 if is_face_model else None

    def detect(self, frame: np.ndarray) -> List:
        if frame is None or frame.size == 0:
            return []
        results = self.model(
            frame,
            conf=self.conf_thres,
            iou=self.track_thres,
            show=False,
            verbose=False
        )
        return results

    def get_largest_human(self, results: List) -> Optional[Tuple[int, int, int, int]]:
        largest_bbox = None
        max_area = 0
        for r in results:
            if not hasattr(r, 'boxes') or r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == self.human_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area and area > 100:
                        max_area = area
                        largest_bbox = (x1, y1, x2, y2)
        return largest_bbox

    def match_faces(self, frame: np.ndarray, results: List, face_db: FaceDatabase) -> np.ndarray:
        frame_copy = frame.copy()
        h, w = frame_copy.shape[:2]
        for r in results:
            if not hasattr(r, 'boxes') or r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if self.face_class_id is not None and cls == self.face_class_id:
                    face_roi = frame_copy[y1:y2, x1:x2]
                    name = face_db.match_face(face_roi) or "未知人脸"
                    frame_copy = self.draw_detection_box(frame_copy, (x1, y1, x2, y2), name, conf)
                elif cls == self.human_class_id:
                    frame_copy = self.draw_detection_box(frame_copy, (x1, y1, x2, y2), "人体", conf)
        return frame_copy

    @staticmethod
    def draw_detection_box(frame: np.ndarray,
                           bbox: Tuple[int, int, int, int],
                           label: str,
                           confidence: float) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{label} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(label_text, font, 0.6, 1)
        text_w, text_h = text_size
        bg_x1, bg_y1 = x1, max(y1 - text_h - 10, 0)
        bg_x2, bg_y2 = x1 + text_w, y1
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), -1)
        cv2.putText(frame, label_text, (x1, bg_y1 + text_h + 2),
                    font, 0.6, (255, 255, 255), 1)
        return frame


# ------------------------------
# 主程序（集成按键操作）
# ------------------------------
if __name__ == "__main__":
    # 1. 初始化组件
    engine = DetectionEngine(model_path="yolov8n.pt", is_face_model=False)  # 如需人脸检测，切换为yolov8n-face.pt
    face_db = FaceDatabase(data_dir="my_face_db", threshold=0.6)
    face_db.load_all_faces()  # 启动时加载所有已保存的人脸

    # 2. 初始化状态变量
    cap = cv2.VideoCapture(0)
    is_paused = False  # 暂停状态
    show_visual = True  # 可视化检测框
    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)

    # 3. 打印操作提示
    print("=" * 50)
    print("📱 操作按键说明：")
    print("   q - 退出程序")
    print("   s - 保存当前最大人脸到库（需先检测到人脸）")
    print("   d - 删除指定姓名的人脸数据")
    print("   l - 重新加载所有人脸数据")
    print("   v - 切换检测框可视化（显示/隐藏）")
    print("   p - 暂停/继续实时检测")
    print("   f - 保存当前帧截图到screenshots目录")
    print("   t - 调整人脸匹配阈值（0~1）")
    print("=" * 50)

    while cap.isOpened():
        # 暂停状态下只处理按键，不读取帧
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面，程序退出")
                break

            # 执行检测
            results = engine.detect(frame)
            largest_human_bbox = engine.get_largest_human(results)
            if largest_human_bbox:
                print(f"🔍 检测到最大人体：{largest_human_bbox}", end="\r")  # 实时打印（覆盖行）

            # 绘制检测框（根据可视化状态）
            if show_visual:
                frame_display = engine.match_faces(frame, results, face_db)
            else:
                frame_display = frame.copy()
        else:
            frame_display = frame.copy()  # 暂停时保持最后一帧

        # 显示画面
        cv2.imshow("🤖 人体/人脸检测系统", frame_display)

        # 按键响应（非阻塞，等待1ms）
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 退出程序
            print("\n👋 程序正常退出")
            break
        elif key == ord('p'):
            # 暂停/继续
            is_paused = not is_paused
            status = "暂停" if is_paused else "继续"
            print(f"\n⏸️  检测已{status}")
        elif key == ord('v'):
            # 切换可视化
            show_visual = not show_visual
            status = "显示" if show_visual else "隐藏"
            print(f"\n🎨 检测框已{status}")
        elif key == ord('f'):
            # 保存截图
            screenshot_path = os.path.join(screenshot_dir, f"screenshot_{np.datetime64('now').astype(str)}.png")
            cv2.imwrite(screenshot_path, frame_display)
            print(f"\n📸 截图已保存：{screenshot_path}")
        elif key == ord('l'):
            # 重新加载人脸
            face_db.load_all_faces()
        elif key == ord('d'):
            # 删除人脸（控制台输入姓名）
            del_name = input("\n🗑️  请输入要删除的人脸姓名：")
            face_db.delete_face(del_name)
        elif key == ord('t'):
            # 调整阈值
            try:
                new_thresh = float(input("\n🎛️  请输入新的人脸匹配阈值（0~1）："))
                if 0 <= new_thresh <= 1:
                    face_db.threshold = new_thresh
                    print(f"✅ 阈值已更新为：{new_thresh}")
                else:
                    print("❌ 阈值需在0~1之间")
            except ValueError:
                print("❌ 输入无效，请输入数字")
        elif key == ord('s'):
            # 保存当前最大人脸（优先人脸检测，无则取人体框内的人脸区域）
            save_name = input("\n💾 请输入要保存的人脸姓名：")
            save_remark = input("📝 请输入备注（可选）：")

            # 尝试获取人脸ROI（优先检测到的人脸，无则取最大人体的中间区域）
            face_roi = None
            results = engine.detect(frame)
            # 方式1：如果用了人脸模型，提取第一个检测到的人脸
            if engine.face_class_id is not None:
                for r in results:
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        for box in r.boxes:
                            if int(box.cls[0]) == engine.face_class_id:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                face_roi = frame[y1:y2, x1:x2]
                                break
            # 方式2：无检测人脸时，取最大人体的上半部分（人脸区域）
            if face_roi is None:
                largest_human = engine.get_largest_human(results)
                if largest_human:
                    x1, y1, x2, y2 = largest_human
                    # 截取人体上半部分作为人脸ROI（需手动调整比例）
                    face_h = int((y2 - y1) * 0.3)
                    face_roi = frame[y1:y1 + face_h, x1:x2]

            if face_roi is not None and face_roi.size > 0:
                face_db.save_face(save_name, face_roi, save_remark, overwrite=True)
            else:
                print("❌ 未检测到有效人脸区域，保存失败")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()