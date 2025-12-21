# ui_handler.py
# 功能：用户交互调度中心（User Interface Handler）
# 职责：
#   - 提供命令行接口（CLI）和交互式菜单两种启动方式
#   - 解析用户输入（图像路径 / 摄像头指令 / 批量目录）
#   - 验证文件路径是否存在、可读、格式有效
#   - 调度静态图像检测、实时摄像头检测 或 批量图像检测
#   - 处理用户中断（Ctrl+C）并优雅退出
#   - 保存检测结果图像并反馈保存状态
#   - 支持运行时切换检测模型（热切换）
#
# 设计原则：
#   - 用户友好：错误提示具体到“文件不存在”、“无权限”、“格式不支持”
#   - 安全兜底：即使用户输错路径或模型，也不崩溃，而是返回主菜单
#   - 松耦合：依赖 DetectionEngine、CameraDetector 和 BatchDetector，但不硬编码其内部逻辑
#   - 可扩展：支持未来新增模式（如视频文件检测）

import os
import cv2
import argparse
import traceback

from detection_engine import DetectionEngine, ModelLoadError
from camera_detector import CameraOpenError
from model_manager import ModelManager  # ← 新增导入


def parse_args():
    """
    解析命令行参数，支持 --image <path>、--camera 或 --batch <dir> 三种模式。
    返回 argparse.Namespace 对象。
    """
    parser = argparse.ArgumentParser(description="YOLOv8 Detection System")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--camera", action="store_true", help="Start live camera detection")
    parser.add_argument("--batch", type=str, help="Path to input directory for batch detection")
    return parser.parse_args()


class UIHandler:
    """
    用户界面控制器。
    初始化时加载初始模型，失败则立即退出。
    支持 CLI 模式、交互式菜单及模型热切换。
    """

    def __init__(self, config):
        """
        初始化 UIHandler。
        若初始模型加载失败，打印错误并退出。
        """
        self.config = config
        try:
            # 使用 ModelManager 管理检测引擎，支持后续热切换
            self.model_manager = ModelManager(
                initial_model_path=config.model_path,
                conf_threshold=config.confidence_threshold
            )
        except Exception as e:
            # ModelManager 内部已处理加载异常，但若完全无法初始化，应退出
            print(f"❌ Fatal: Cannot initialize detection engine with initial model: {e}")
            raise SystemExit(1)

    def run(self):
        """
        主流程入口：
          - 若有 --image 参数 → 静态检测
          - 若有 --camera 参数 → 摄像头检测
          - 若有 --batch 参数 → 批量检测
          - 否则 → 交互式菜单
        """
        args = parse_args()
        if args.image is not None:
            print(f"[CLI Mode] Detecting static image: {args.image}")
            self._run_static_detection(args.image)
        elif args.camera:
            print("[CLI Mode] Starting live camera detection...")
            self._run_camera_detection()
        elif args.batch is not None:
            print(f"[CLI Mode] Running batch detection on directory: {args.batch}")
            self._run_batch_detection(args.batch)
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
            print("3. Batch Image Detection")
            print("4. Switch Detection Model")  # ← 新增选项
            print("5. Exit")
            choice = input("Please select an option (1-5): ").strip()
        except KeyboardInterrupt:
            print("\nUser cancelled. Exiting...")
            return

        if choice == "1":
            self._choose_image_source()
        elif choice == "2":
            self._run_camera_detection()
        elif choice == "3":
            self._run_batch_detection_interactive()
        elif choice == "4":
            self._switch_model_interactive()  # ← 新增方法
        elif choice == "5":
            print("Goodbye!")
        else:
            print("Invalid option. Please enter 1, 2, 3, 4, or 5.")
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

        # 使用当前模型进行检测
        annotated_frame, _ = self.model_manager.get_current_engine().detect(frame)

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
                detection_engine=self.model_manager.get_current_engine(),  # ← 使用当前模型
                output_interval=self.config.output_interval
            )
            detector.start_detection(camera_index=self.config.camera_index)
        except CameraOpenError as e:
            print(f"❌ Camera error: {e}")
        except Exception as e:
            print(f"💥 Camera detection failed: {e}")
            traceback.print_exc()

    def _run_batch_detection_interactive(self):
        """
        交互式批量检测：用户输入输入目录，自动将结果保存到同级 test_picture/ 目录。
        """
        try:
            input_dir = input("Enter input directory path (e.g., ../data): ").strip()
            input_dir = os.path.expanduser(input_dir)
            # 清理不可见字符（如从 Windows 资源管理器复制的路径）
            input_dir = ''.join(ch for ch in input_dir if ord(ch) != 0x202A)
        except KeyboardInterrupt:
            return

        if not os.path.isdir(input_dir):
            print(f"❌ Directory not found: {input_dir}")
            return

        # 默认输出目录：与输入目录同级的 test_picture/
        output_dir = os.path.join(input_dir, "test_picture")
        self._run_batch_detection(input_dir, output_dir)

    def _run_batch_detection(self, input_dir, output_dir=None):
        """
        执行批量图像检测。
        参数:
            input_dir (str): 输入图像目录
            output_dir (str, optional): 输出目录，默认为 input_dir/test_picture
        """
        if output_dir is None:
            output_dir = os.path.join(input_dir, "test_picture")

        try:
            from batch_detector import BatchDetector
            detector = BatchDetector(
                detection_engine=self.model_manager.get_current_engine(),  # ← 使用当前模型
                input_dir=input_dir,
                output_dir=output_dir
            )
            detector.run()
        except ValueError as e:
            print(f"❌ Batch detection setup error: {e}")
        except Exception as e:
            print(f"💥 Batch detection failed: {e}")
            traceback.print_exc()

    def _switch_model_interactive(self):
        """
        交互式切换检测模型。
        允许用户输入新模型路径（本地文件或官方名称），尝试热加载。
        成功后，所有后续检测将使用新模型。
        """
        print("\n--- Switch Detection Model ---")
        print("Examples:")
        print("  • yolov8n.pt   (smallest, fastest)")
        print("  • yolov8s.pt   (balanced)")
        print("  • yolov8m.pt   (more accurate)")
        print("  • ./models/custom.pt  (your own model)")
        try:
            new_model = input("Enter new model path or name: ").strip()
        except KeyboardInterrupt:
            print("\nModel switch cancelled.")
            return

        if not new_model:
            print("Empty input. Model switch cancelled.")
            return

        success = self.model_manager.switch_model(new_model)
        if success:
            print("✅ Model switch completed successfully.")
        else:
            print("⚠️ Model switch failed. Current model remains active.")
