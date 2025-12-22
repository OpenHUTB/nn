import cv2
import os
import time
import numpy as np
import threading


class FaceDetector:
    def __init__(self):
        # 加载多个级联分类器（提高检测率）
        self.cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        ]

        self.cascades = []
        for path in self.cascade_paths:
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                self.cascades.append(cascade)
                print(f"✅ 加载分类器: {os.path.basename(path)}")

        if not self.cascades:
            raise Exception("❌ 所有人脸检测模型加载失败")

        # 检测参数（优化性能）
        self.scale_factor = 1.1
        self.min_neighbors = 3  # 降低以提高召回率
        self.min_size = (30, 30)
        self.max_size = (300, 300)

        # 检测缓存
        self.last_detection = []
        self.last_frame_hash = None
        self.cache_lock = threading.Lock()

        # 性能统计
        self.detection_count = 0
        self.detection_times = []

    def detect_faces(self, frame):
        """检测人脸（带缓存优化）"""
        start_time = time.time()

        # 生成帧哈希（用于缓存）
        frame_small = cv2.resize(frame, (160, 120))
        frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        frame_hash = hash(frame_gray.tobytes())

        # 检查缓存
        with self.cache_lock:
            if self.last_frame_hash == frame_hash and self.last_detection:
                detection_time = time.time() - start_time
                self.detection_times.append(detection_time)
                self.detection_count += 1
                return self.last_detection.copy()

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 均衡化（提高检测率）
        gray = cv2.equalizeHist(gray)

        all_faces = []

        # 使用多个分类器检测
        for cascade in self.cascades:
            try:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors,
                    minSize=self.min_size,
                    maxSize=self.max_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                # 合并检测结果
                for (x, y, w, h) in faces:
                    # 非极大值抑制（避免重复框）
                    overlap = False
                    for (fx, fy, fw, fh) in all_faces:
                        # 计算IoU
                        ix1 = max(x, fx)
                        iy1 = max(y, fy)
                        ix2 = min(x + w, fx + fw)
                        iy2 = min(y + h, fy + fh)

                        if ix2 > ix1 and iy2 > iy1:
                            area_i = (ix2 - ix1) * (iy2 - iy1)
                            area_a = w * h
                            area_b = fw * fh
                            iou = area_i / (area_a + area_b - area_i)

                            if iou > 0.5:  # 重叠度超过50%
                                overlap = True
                                break

                    if not overlap:
                        all_faces.append((x, y, w, h))

            except Exception as e:
                print(f"分类器检测错误: {e}")
                continue

        # 更新缓存
        with self.cache_lock:
            self.last_detection = all_faces.copy()
            self.last_frame_hash = frame_hash

        # 记录性能
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        self.detection_count += 1

        # 保持最近100次记录
        if len(self.detection_times) > 100:
            self.detection_times = self.detection_times[-100:]

        return all_faces

    def draw_faces(self, frame, faces):
        """绘制人脸框"""
        for (x, y, w, h) in faces:
            # 绘制框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 绘制标签
            cv2.putText(frame, "Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示检测统计
        if len(self.detection_times) > 0:
            avg_time = np.mean(self.detection_times[-10:]) * 1000
            stats_text = f"FaceDet: {avg_time:.1f}ms"
            cv2.putText(frame, stats_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def get_detection_stats(self):
        """获取检测统计"""
        if len(self.detection_times) == 0:
            return {"avg_time": 0, "fps": 0}

        avg_time = np.mean(self.detection_times) * 1000
        fps = 1.0 / np.mean(self.detection_times) if np.mean(self.detection_times) > 0 else 0

        return {
            "avg_time_ms": avg_time,
            "fps": fps,
            "total_detections": self.detection_count,
            "cascade_count": len(self.cascades)
        }

    def detect_from_camera(self, save_path=None):
        """摄像头实时检测（测试用）"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("❌ 摄像头打开失败")

        print("📷 人脸检测中（按q退出）")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.detect_faces(frame)
            frame = self.draw_faces(frame, faces)

            # 显示检测数量
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def test_face_detector():
    """测试人脸检测"""
    detector = FaceDetector()
    detector.detect_from_camera()


if __name__ == "__main__":
    test_face_detector()