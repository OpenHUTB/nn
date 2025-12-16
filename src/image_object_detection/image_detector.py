# image_detector.py
import cv2
import numpy as np
import traceback
import os

class ImageDetector:
    def __init__(self, detection_engine):
        self.engine = detection_engine
    
    def detect_static_image(self, image_path):
        print(f"Loading image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found - {image_path}")
            return
        
        frame = cv2.imread(image_path)
        if frame is None or frame.size == 0:
            print("Error: Failed to load image")
            return
        
        print("Running detection...")
        annotated_frame, results = self.engine.detect(frame)
        
        if annotated_frame is None or annotated_frame.size == 0:
            print("Error: Invalid annotated frame")
            return
        
        self._display_results(results)
        self._show_image(annotated_frame, "YOLO_Static_Detection")
    
    def _display_results(self, results):
        if not results:
            print("No objects detected.")
            return
        
        result = results[0]
        boxes = result.boxes
        if len(boxes) == 0:
            print("No objects detected.")
            return
        
        print(f"Detected {len(boxes)} object(s):")
        names_list = self.engine.model.names
        
        for i, box in enumerate(boxes):
            try:
                cls_index = int(box.cls.item())
                confidence = box.conf.item()
                cls_name = names_list[cls_index] if 0 <= cls_index < len(names_list) else f"unknown_{cls_index}"
                print(f"  {i+1}. {cls_name} (confidence: {confidence:.2f})")
            except Exception as e:
                print(f"  Warning: Failed to parse box {i+1}: {e}")
    
    def _show_image(self, annotated_frame, window_name="YOLO_Static_Detection"):
        try:
            cv2.destroyAllWindows()
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            cv2.imshow(window_name, annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Failed to display image: {e}")
            traceback.print_exc()
