# ui_handler.py
# 功能：统一处理用户交互逻辑，支持命令行参数和交互式菜单两种模式

import os
import cv2
import argparse
from detection_engine import DetectionEngine


def parse_args():
    """
    解析命令行参数。
    
    支持两种运行模式：
        --image <path>   : 指定静态图像路径进行检测
        --camera         : 启动实时摄像头检测
    
    返回:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="YOLOv8 Detection System")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--camera", action="store_true", help="Start live camera detection")
    return parser.parse_args()


class UIHandler:
    """
    用户界面处理器类。
    负责协调命令行参数、交互式菜单、图像/摄像头检测流程，
    是整个应用的调度中心。
    """

    def __init__(self, config):
        """
        初始化 UI 处理器。

        参数:
            config (Config): 配置对象，包含模型路径、阈值、摄像头索引等参数
        """
        self.config = config
        # 初始化检测引擎（加载 YOLO 模型）
        self.engine = DetectionEngine(
            model_path=config.model_path,
            conf_threshold=config.confidence_threshold
        )

    def run(self):
        """
        主运行入口。
        优先检查命令行参数；若无，则进入交互式菜单。
        """
        args = parse_args()

        if args.image is not None:
            print(f"[CLI Mode] Detecting static image: {args.image}")
            self._run_static_detection(image_path=args.image)
        elif args.camera:
            print("[CLI Mode] Starting live camera detection...")
            self._run_camera_detection()
        else:
            # 无命令行参数时，启动交互式菜单
            self._interactive_menu()

    def _interactive_menu(self):
        """
        显示交互式主菜单，供用户选择操作模式。
        支持选项：
            1. 静态图像检测（可选默认图或自定义路径）
            2. 实时摄像头检测
            3. 退出程序
        """
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
        """
        子菜单：让用户选择使用默认测试图像还是输入自定义路径。
        默认路径硬编码为桌面的 test.jpg（适用于快速测试）。
        """
        default_image_path = self.config.default_image_path  # 使用配置中的默认图片路径
        print("\n--- Static Image Detection ---")
        print(f"a) Use default test image at: {default_image_path}")
        print("b) Enter custom image path")
        sub_choice = input("Choose (a/b): ").strip().lower()

        if sub_choice == "a":
            # 检查默认图像是否存在
            if not os.path.exists(default_image_path):
                print(f"\n⚠️ Default image not found at:\n {default_image_path}")
                print("💡 Please place a 'test.jpg' file in the specified location, or choose option (b).")
                return
            print(f"Using default image: {default_image_path}")
            self._run_static_detection(image_path=default_image_path)

        elif sub_choice == "b":
            # 获取用户输入的路径，并展开 ~ 符号（如 ~/Pictures/img.jpg）
            custom_path = input("Enter full or relative image path: ").strip()
            custom_path = os.path.expanduser(custom_path)
            # 移除可能的不可见 Unicode 控制字符（特别是从 Windows 复制的路径）
            custom_path = ''.join(ch for ch in custom_path if ord(ch) != 0x202A)
            if not os.path.exists(custom_path):
                print(f"❌ Error: File not found at: {custom_path}")
                return
            self._run_static_detection(image_path=custom_path)

        else:
            print("Invalid choice. Returning to main menu.")

    def _run_static_detection(self, image_path):
        """ 
        执行静态图像检测流程。 
        
        参数:
            image_path (str): 待检测图像的完整路径
        """
        print(f"🔍 Detecting objects in: {image_path}")
        try:
            # 直接读取图像
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"❌ Failed to load image from: {image_path}")
                return

            # 使用已有的 self.engine（DetectionEngine）进行检测
            annotated_frame, results = self.engine.detect(frame)

            # 显示结果
            cv2.imshow("YOLO Detection Result", annotated_frame)
            print("Detection completed. Press any key to close the window.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 可选：保存结果
            save_path = image_path.replace(".jpg", "_detected.jpg").replace(".png", "_detected.png")
            cv2.imwrite(save_path, annotated_frame)
            print(f"Result saved to: {save_path}")

        except Exception as e:
            print(f"❌ Detection failed: {e}")
            import traceback
            traceback.print_exc()  # 打印完整错误栈，便于调试

    def _run_camera_detection(self):
        """
        执行实时摄像头检测流程。
        使用配置中的摄像头索引和输出间隔参数。
        """
        try:
            from camera_detector import CameraDetector
            detector = CameraDetector(
                detection_engine=self.engine,
                output_interval=self.config.output_interval
            )
            detector.start_detection(camera_index=self.config.camera_index)
        except Exception as e:
            print(f"❌ Camera detection failed: {e}")
            import traceback
            traceback.print_exc()
