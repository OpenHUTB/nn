# camera_detector.py
import cv2
import time
import os
import traceback

class CameraDetector:
    def __init__(self, detection_engine, camera_index=0):
        self.engine = detection_engine
        self.camera_index = camera_index
        self.window_name = "YOLO 实时检测 - 摄像头"
    
    def detect_camera(self):
        """对外统一接口：启动摄像头实时检测"""
        self.start_detection()

    def start_detection(self):
        """实际执行摄像头检测的逻辑"""
        print("正在初始化摄像头...")
        cap = self._open_camera()
        if cap is None:
            return
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        
        print("摄像头已启动。按 'q' 或 ESC 退出，按 's' 保存当前帧。")
        
        prev_time = time.time()
        saved_frame_index = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("警告: 无法读取摄像头帧（可能设备被拔出或占用）")
                    break
                
                annotated_frame, results = self._perform_detection(frame)
                if annotated_frame is None:
                    annotated_frame = frame
                
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                num_objects = len(results[0].boxes) if results else 0
                info_text = f"FPS: {fps:.1f} | Objects: {num_objects}"
                cv2.putText(
                    annotated_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow(self.window_name, annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("用户请求退出。")
                    break
                elif key == ord('s'):
                    saved_frame_index = self._save_frame(annotated_frame, saved_frame_index)
        
        except Exception as e:
            print(f"检测过程中发生未预期错误: {e}")
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("摄像头已关闭，窗口已清理.")

    def _open_camera(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头（索引 {self.camera_index}）")
            print("请检查摄像头是否连接、是否被其他程序占用。")
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def _perform_detection(self, frame):
        try:
            # 抑制 YOLO 内部输出
            import sys, os
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            annotated_frame, results = self.engine.detect(frame)
            sys.stdout = old_stdout
            return annotated_frame, results
        except Exception as e:
            print(f"单帧检测失败: {e}")
            return None, []

    def _save_frame(self, frame, index):
        save_dir = "saved_frames"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"frame_{index:04d}.jpg")
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"✅ 帧已保存: {filename}")
            return index + 1
        else:
            print("❌ 保存帧失败！")
            return index
