# camera_detector.py
import cv2
import time

class CameraDetector:
    def __init__(self, detection_engine):
        self.engine = detection_engine
    
    def detect_camera(self):
        """检测摄像头实时画面"""
        try:
            # 打开摄像头
            print("正在打开摄像头...")
            cap = self._open_camera()
            
            if cap is None:
                return
            
            print("摄像头已打开，开始实时检测。")
            print("在显示窗口中按 'q' 键退出，或在终端中按 Ctrl+C 中断程序。\n")
            
            self._run_camera_loop(cap)
            
        except KeyboardInterrupt:
            print("\n\n用户按下了 Ctrl+C，强制退出摄像头检测...")
            print("程序已安全退出。")
        except Exception as e:
            print(f"摄像头检测过程中发生错误: {e}")
        finally:
            self._cleanup_resources(cap)
    
    def _open_camera(self):
        """打开摄像头"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            return None
        
        return cap
    
    def _run_camera_loop(self, cap):
        """摄像头检测主循环"""
        # 记录上次输出时间，控制输出频率
        last_output_time = time.time()
        output_interval = 1.0  # 每秒最多输出一次检测信息
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面，退出...")
                break
            
            results = self._perform_detection(frame)
            annotated_frame = results[0].plot()
            
            # 控制输出频率，避免终端被刷屏
            self._handle_output_frequency(results, last_output_time, output_interval)
            last_output_time = time.time()
            
            cv2.imshow('YOLO 检测结果 - 摄像头', annotated_frame)
            
            # 按 'q' 键退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户按下 'q' 键，退出摄像头检测...")
                break
    
    def _perform_detection(self, frame):
        """执行检测"""
        annotated_frame, results = self.engine.detect(frame)
        return results
    
    def _handle_output_frequency(self, results, last_output_time, output_interval):
        """处理输出频率"""
        current_time = time.time()
        if current_time - last_output_time >= output_interval:
            detected_count = len(results[0].boxes)
            if detected_count > 0:
                # 构建检测对象字符串
                detected_objects = []
                for box in results[0].boxes:
                    cls_index = int(box.cls)
                    cls_name = self.engine.model.names[cls_index]
                    confidence = box.conf.item()
                    detected_objects.append(f"{cls_name}({confidence:.2f})")
                
                print(f"检测到 {detected_count} 个对象: {', '.join(detected_objects)}")
    
    def _cleanup_resources(self, cap):
        """清理资源"""
        print("释放摄像头资源...")
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        print("摄像头检测已停止。")
