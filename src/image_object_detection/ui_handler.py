# ui_handler.py
# 功能：用户交互调度中心（User Interface Handler）
# 职责：
#   - 提供命令行接口（CLI）和交互式菜单两种启动方式
#   - 解析用户输入（图像路径 / 摄像头指令）
#   - 验证文件路径是否存在、可读、格式有效
#   - 调度静态图像检测 或 实时摄像头检测
#   - 处理用户中断（Ctrl+C）并优雅退出
#   - 保存检测结果图像并反馈保存状态
#
# 设计原则：
#   - 用户友好：错误提示具体到“文件不存在”、“无权限”、“格式不支持”
#   - 安全兜底：即使用户输错路径，也不崩溃，而是返回主菜单
#   - 松耦合：依赖 DetectionEngine 和 CameraDetector，但不硬编码其内部逻辑
#   - 可扩展：支持未来新增模式（如视频文件检测）

import os
import cv2
import argparse
import traceback

from detection_engine import DetectionEngine, ModelLoadError
from camera_detector import CameraOpenError


def parse_args():
    """
    解析命令行参数，支持 --image <path> 或 --camera 两种模式。
    返回 argparse.Namespace 对象。
    """
    parser = argparse.ArgumentParser(description="YOLOv8 Detection System")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--camera", action="store_true", help="Start live camera detection")
    return parser.parse_args()


class UIHandler:
    """
    用户界面控制器。
    初始化时加载模型，失败则立即退出。
    支持 CLI 模式和交互式菜单。
    """

    def __init__(self, config):
        """
        初始化 UIHandler。
        若 DetectionEngine 初始化失败（如模型加载错误），打印错误并退出。
        """
        self.config = config
        try:
            self.engine = DetectionEngine(
                model_path=config.model_path,
                conf_threshold=config.confidence_threshold
            )
        except ModelLoadError as e:
            print(f"❌ Fatal: Failed to initialize detection engine: {e}")
            raise SystemExit(1)

    def run(self):
        """
        主流程入口：
          - 若有 --image 参数 → 静态检测
          - 若有 --camera 参数 → 摄像头检测
          - 否则 → 交互式菜单
        """
        args = parse_args()
        if args.image is not None:
            print(f"[CLI Mode] Detecting static image: {args.image}")
            self._run_static_detection(args.image)
        elif args.camera:
            print("[CLI Mode] Starting live camera detection...")
            self._run_camera_detection()
        else:
            self._interactive_menu()

    def _interactive_menu(self):
        """
        显示交互式文本菜单，处理用户选择。
        支持 Ctrl+C 中断，无效输入递归重试。
        """
        try:
            print("\n" + "=" * 40)
            print("🚀 YOLOv8 Detection System")
            print("=" * 40)
            print("1. Static Image Detection")
            print("2. Live Camera Detection")
            print("3. Exit")
            choice = input("Please select an option (1-3): ").strip()
        except KeyboardInterrupt:
            print("\nUser cancelled. Exiting...")
            return

        if choice == "1":
            self._choose_image_source()
        elif choice == "2":
            self._run_camera_detection()
        elif choice == "3":
            print("Goodbye!")
        else:
            print("Invalid option. Please enter 1, 2, or 3.")
            self._interactive_menu()

    def _choose_image_source(self):
        """
        子菜单：让用户选择默认测试图或自定义路径。
        对自定义路径进行 ~ 展开和不可见字符清理。
        分级验证路径有效性（存在性、可读性）。
        """
        default_path = self.config.default_image_path
        print("\n--- Static Image Detection ---")
        print(f"a) Use default test image at: {default_path}")
        print("b) Enter custom image path")
        try:
            sub_choice = input("Choose (a/b): ").strip().lower()
        except KeyboardInterrupt:
            return

        if sub_choice == "a":
            if not os.path.exists(default_path):
                print(f"⚠️ Default image not found: {default_path}")
                print("💡 Place 'test.jpg' in the 'data/' folder or choose (b).")
                return
            self._run_static_detection(default_path)
        elif sub_choice == "b":
            try:
                custom_path = input("Enter image path: ").strip()
                custom_path = os.path.expanduser(custom_path)
                # 清理从某些系统复制时可能带入的不可见 Unicode 控制字符（如 U+202A）
                custom_path = ''.join(ch for ch in custom_path if ord(ch) != 0x202A)
            except KeyboardInterrupt:
                return

            if not os.path.exists(custom_path):
                print(f"❌ File not found: {custom_path}")
                return
            if not os.access(custom_path, os.R_OK):
                print(f"❌ Permission denied: {custom_path}")
                return

            self._run_static_detection(custom_path)
        else:
            print("Invalid choice. Returning to main menu.")

    def _run_static_detection(self, image_path):
        """
        执行单张图像检测：
          - 使用 cv2.imread 读取
          - 若失败，分级诊断原因（路径？权限？格式？）
          - 显示结果窗口，等待按键关闭
          - 自动保存结果图（原文件名 + "_detected" + 原扩展名）
        """
        print(f"🔍 Detecting objects in: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            # 分级诊断 imread 失败原因
            if not os.path.exists(image_path):
                print(f"❌ Path does not exist: {image_path}")
            elif not os.access(image_path, os.R_OK):
                print(f"❌ No read permission: {image_path}")
            else:
                print(f"❌ Unsupported or corrupted image format: {image_path}")
            return

        annotated_frame, _ = self.engine.detect(frame)

        window_name = "YOLO Detection Result"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, annotated_frame)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 智能保留原扩展名（JPG/PNG）
        ext = ".jpg" if image_path.lower().endswith(".jpg") else ".png"
        save_path = image_path.replace(ext, f"_detected{ext}")
        try:
            success = cv2.imwrite(save_path, annotated_frame)
            if success:
                print(f"✅ Result saved to: {save_path}")
            else:
                print("❌ Failed to save result (OpenCV write error)")
        except Exception as e:
            print(f"⚠️ Failed to save result: {e}")

    def _run_camera_detection(self):
        """
        启动实时摄像头检测。
        动态创建 CameraDetector 实例并运行。
        捕获摄像头专属异常和其他未预期错误。
        """
        try:
            from camera_detector import CameraDetector
            detector = CameraDetector(
                detection_engine=self.engine,
                output_interval=self.config.output_interval
            )
            detector.start_detection(camera_index=self.config.camera_index)
        except CameraOpenError as e:
            print(f"❌ Camera error: {e}")
        except Exception as e:
            print(f"💥 Camera detection failed: {e}")
            traceback.print_exc()
