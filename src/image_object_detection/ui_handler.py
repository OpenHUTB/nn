# ui_handler.py
import os
import cv2
import argparse
from detection_engine import DetectionEngine
from image_detector import ImageDetector
from camera_detector import CameraDetector

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Detection System")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--camera", action="store_true", help="Start live camera detection")
    return parser.parse_args()

class UIHandler:
    def __init__(self, config):
        self.config = config
        self.engine = DetectionEngine(
            model_path=config.model_path,
            conf_threshold=config.confidence_threshold
        )

    def run(self):
        args = parse_args()

        if args.image is not None:
            print(f"[CLI Mode] Detecting static image: {args.image}")
            self._run_static_detection(image_path=args.image)
        elif args.camera:
            print("[CLI Mode] Starting live camera detection...")
            self._run_camera_detection()
        else:
            self._interactive_menu()

    def _interactive_menu(self):
        print("=== YOLO Detection System ===")
        print("1. Static Image Detection")
        print("2. Live Camera Detection")
        print("3. Exit")

        choice = input("Please select an option (1-3): ").strip()

        if choice == "1":
            self._choose_image_source()
        elif choice == "2":
            self._run_camera_detection()
        elif choice == "3":
            print("Exiting program.")
        else:
            print("Invalid option. Please enter 1, 2, or 3.")

    def _choose_image_source(self):
        """让用户选择使用默认图还是自定义路径"""
        default_image_path = r"C:\Users\apple\OneDrive\桌面\test.jpg"
        
        print("\n--- Static Image Detection ---")
        print(f"a) Use default test image at: {default_image_path}")
        print("b) Enter custom image path")
        sub_choice = input("Choose (a/b): ").strip().lower()

        if sub_choice == "a":
            if not os.path.exists(default_image_path):
                print(f"\n⚠️  Default image not found at:\n    {default_image_path}")
                print("💡 Please place a 'test.jpg' file in the specified location, or choose option (b).")
                return
            print(f"Using default image: {default_image_path}")
            self._run_static_detection(image_path=default_image_path)

        elif sub_choice == "b":
            custom_path = input("Enter full or relative image path: ").strip()
            # 支持 ~ 和相对路径
            custom_path = os.path.expanduser(custom_path)
            if not os.path.exists(custom_path):
                print(f"❌ Error: File not found at: {custom_path}")
                return
            self._run_static_detection(image_path=custom_path)
        else:
            print("Invalid choice. Returning to main menu.")

    def _run_static_detection(self, image_path):
        """执行静态图像检测"""
        print(f"🔍 Detecting objects in: {image_path}")
        try:
            detector = ImageDetector(self.engine)
            detector.detect_static_image(image_path)
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            import traceback
            traceback.print_exc()

    def _run_camera_detection(self):
        try:
            detector = CameraDetector(
                detection_engine=self.engine,
                output_interval=self.config.output_interval
            )
            detector.start_detection(camera_index=self.config.camera_index)
        except Exception as e:
            print(f"❌ Camera detection failed: {e}")
            import traceback
            traceback.print_exc()
