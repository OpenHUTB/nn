# ui_handler.py
import os
import cv2
import traceback
from detection_engine import DetectionEngine
from image_detector import ImageDetector
from camera_detector import CameraDetector

class UIHandler:
    def __init__(self, config):
        self.config = config
        self.engine = DetectionEngine(
            model_path=config.model_path,
            conf_threshold=config.confidence_threshold
        )
        self.stop_flag = False
    
    def run(self):
        print("=== YOLO Detection System ===")
        print("1. Static Image Detection")
        print("2. Live Camera Detection")
        print("3. Exit")
        
        choice = input("Please select an option (1-3): ").strip()
        
        if choice == "1":
            self._run_static_detection()
        elif choice == "2":
            self._run_camera_detection()
        elif choice == "3":
            print("Exiting program.")
        else:
            print("Invalid option. Please enter 1, 2, or 3.")
    
    def _run_static_detection(self):
        image_path = self.config.test_image_path
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
        
        detector = ImageDetector(self.engine)
        detector.detect_static_image(image_path)
    
    def _run_camera_detection(self):
        detector = CameraDetector(
            detection_engine=self.engine,
            output_interval=self.config.output_interval
        )
        detector.start_detection(camera_index=self.config.camera_index)
