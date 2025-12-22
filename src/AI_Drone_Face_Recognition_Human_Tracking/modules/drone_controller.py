import cv2
import numpy as np
import time


class DroneController:
    def __init__(self):
        self.cap = None
        self.mode = "camera"
        self.init_camera()

    def init_camera(self):
        """初始化摄像头"""
        self.mode = "camera"

        # 尝试打开摄像头
        for cam_index in [0, 1]:
            try:
                self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print(f"✅ 摄像头 {cam_index} 初始化成功")
                    return True
            except:
                continue

        print("⚠️  摄像头打开失败，使用模拟模式")
        return False

    def get_frame(self):
        """获取画面"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame

        # 生成模拟画面
        return self._generate_test_frame()

    def _generate_test_frame(self):
        """生成测试画面"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 网格
        for i in range(0, 640, 40):
            cv2.line(frame, (i, 0), (i, 480), (50, 50, 50), 1)
        for i in range(0, 480, 40):
            cv2.line(frame, (0, i), (640, i), (50, 50, 50), 1)

        # 中心
        center_x, center_y = 320, 240
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (80, 80, 120), 2)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (80, 80, 120), 2)

        # 文字
        cv2.putText(frame, "模拟画面", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "按T起飞，按Y追踪", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 100), 1)

        return frame

    def disconnect(self):
        """断开连接"""
        if self.cap:
            self.cap.release()
        print("✅ 摄像头已释放")