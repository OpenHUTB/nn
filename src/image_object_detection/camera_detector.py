# camera_detector.py
# 功能：封装基于摄像头的实时目标检测流程，支持帧保存与 FPS 显示

import cv2
import time
import traceback


class CameraDetector:
    """
    摄像头实时检测器类。
    使用传入的 detection_engine（如 YOLO 模型）对摄像头视频流进行逐帧推理，
    并提供可视化、FPS 统计、帧保存等功能。
    """

    def __init__(self, detection_engine, output_interval=1.0):
        """
        初始化检测器。

        参数:
            detection_engine: 实现 detect(frame) 方法的对象，用于执行目标检测
            output_interval (float): FPS 输出的时间间隔（秒），默认每 1 秒打印一次
        """
        self.engine = detection_engine           # 外部传入的检测引擎（如 DetectionEngine 实例）
        self.output_interval = output_interval   # FPS 打印间隔（秒）
        self.last_output_time = 0                # 上次打印 FPS 的时间戳
        self.frame_count = 0                     # 自上次打印以来处理的帧数
        self.window_name = "YOLO_Live_Detection" # OpenCV 窗口名称（英文，避免编码问题）

    def start_detection(self, camera_index=0):
        """
        启动摄像头并开始实时检测。

        参数:
            camera_index (int): 摄像头设备索引，默认为 0（主摄像头）
        """
        # 尝试打开指定索引的摄像头
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_index}")
            return

        print("Starting live detection. Press 'q' to quit, 's' to save frame.")
        # 创建可调整大小的 OpenCV 窗口（只需创建一次）
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                # 从摄像头读取一帧
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Failed to read frame from camera")
                    break

                current_time = time.time()

                # 使用检测引擎对当前帧进行推理，返回带标注的图像和原始结果
                annotated_frame, results = self.engine.detect(frame)

                # 安全检查：确保返回的图像有效
                if annotated_frame is None or annotated_frame.size == 0:
                    print("Warning: Invalid detection result")
                    continue

                # 显示带检测框的图像
                cv2.imshow(self.window_name, annotated_frame)

                # 检查键盘输入（等待 1 毫秒）
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):      # 按 'q' 退出
                    break
                elif key == ord('s'):    # 按 's' 保存当前帧
                    self.save_frame(annotated_frame)

                # 更新 FPS 统计并按需打印
                self._print_fps_if_needed(current_time)
                self.frame_count += 1

        except KeyboardInterrupt:
            print("\nDetection interrupted by user.")
        except Exception as e:
            print(f"Unexpected error during detection: {e}")
            traceback.print_exc()  # 打印完整错误栈，便于调试
        finally:
            # 确保资源被正确释放
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed.")

    def _print_fps_if_needed(self, current_time):
        """
        根据设定的时间间隔计算并打印当前 FPS（帧率）。

        参数:
            current_time (float): 当前时间戳（秒）
        """
        # 判断是否到达输出间隔
        if current_time - self.last_output_time >= self.output_interval:
            # 避免除零：首次运行时 last_output_time 为 0，不计算 FPS
            fps = self.frame_count / (current_time - self.last_output_time) if self.last_output_time > 0 else 0
            print(f"FPS: {fps:.2f}")

            # 重置计时器和帧计数器
            self.last_output_time = current_time
            self.frame_count = 0

    def save_frame(self, frame):
        """
        将当前帧保存为 JPEG 图像文件，文件名包含时间戳。

        参数:
            frame (np.ndarray): 要保存的图像数组（BGR 格式）
        """
        import os  # 延迟导入（仅在此方法中使用）
        timestamp = int(time.time())  # 使用 Unix 时间戳确保文件名唯一
        filename = f"saved_frame_{timestamp}.jpg"
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"Frame saved as {filename}")
        else:
            print("Failed to save frame")
