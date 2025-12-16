# camera_detector.py
import cv2
import time
import traceback

class CameraDetector:
    def __init__(self, detection_engine, output_interval=1.0):
        self.engine = detection_engine
        self.output_interval = output_interval
        self.last_output_time = 0
        self.frame_count = 0
        self.window_name = "YOLO_Live_Detection"  # 👈 英文窗口名
    
    def start_detection(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_index}")
            return
        
        print("Starting live detection. Press 'q' to quit, 's' to save frame.")
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # 创建一次即可
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Failed to read frame from camera")
                    break
                
                current_time = time.time()
                annotated_frame, results = self.engine.detect(frame)
                
                if annotated_frame is None or annotated_frame.size == 0:
                    print("Warning: Invalid detection result")
                    continue
                
                cv2.imshow(self.window_name, annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_frame(annotated_frame)
                
                self._print_fps_if_needed(current_time)
                self.frame_count += 1
            
        except KeyboardInterrupt:
            print("\nDetection interrupted by user.")
        except Exception as e:
            print(f"Unexpected error during detection: {e}")
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed.")
    
    def _print_fps_if_needed(self, current_time):
        if current_time - self.last_output_time >= self.output_interval:
            fps = self.frame_count / (current_time - self.last_output_time) if self.last_output_time > 0 else 0
            print(f"FPS: {fps:.2f}")
            self.last_output_time = current_time
            self.frame_count = 0
    
    def save_frame(self, frame):
        import os
        timestamp = int(time.time())
        filename = f"saved_frame_{timestamp}.jpg"
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"Frame saved as {filename}")
        else:
            print("Failed to save frame")
