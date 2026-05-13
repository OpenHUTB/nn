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
#   - 用户友好：错误提示具体到"文件不存在"、"无权限"、"格式不支持"
#   - 安全兜底：即使用户输错路径或模型，也不崩溃，而是返回主菜单
#   - 松耦合：依赖 DetectionEngine、CameraDetector 和 BatchDetector，但不硬编码其内部逻辑
#   - 可扩展：支持未来新增模式（如视频文件检测）

import os
import cv2
import argparse
import traceback

from detection_engine import DetectionEngine, ModelLoadError
from camera_detector import CameraOpenError
from model_manager import ModelManager
from video_detector import VideoDetector


def parse_args():
    """
    Parse command line arguments.
    Returns argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="YOLOv8 Image Object Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--image", type=str, help="Path to input image file")
    mode_group.add_argument("--camera", action="store_true", help="Start live camera detection")
    mode_group.add_argument("--batch", type=str, help="Path to input directory for batch detection")
    mode_group.add_argument("--video", type=str, help="Path to input video file")
    mode_group.add_argument("--compare", action="store_true", help="Enable multi-model comparison")

    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--models", type=str, nargs='+', help="Multiple models for comparison")

    parser.add_argument("--output", type=str, default=None, help="Output directory/path")
    parser.add_argument("--save", action="store_true", default=True, help="Save results")
    parser.add_argument("--no-save", action="store_true", help="Do not save results")

    parser.add_argument("--cam-index", type=int, default=0, help="Camera device index")
    parser.add_argument("--output-interval", type=float, default=1.0, help="Output interval (seconds)")

    parser.add_argument("--stats", action="store_true", help="Generate statistics report")
    parser.add_argument("--export-json", type=str, default=None, help="Export results to JSON")

    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    return parser.parse_args()


class UIHandler:
    """User interface handler for YOLO detection system."""

    def __init__(self, config):
        """Initialize UIHandler with configuration."""
        self.config = config
        self.video_detector = VideoDetector()
        try:
            self.model_manager = ModelManager(
                initial_model_path=config.model_path,
                conf_threshold=config.confidence_threshold
            )
        except Exception as e:
            print(f"❌ Fatal: Cannot initialize detection engine: {e}")
            raise SystemExit(1)

    def run(self):
        """Main entry point. Handles CLI arguments or interactive menu."""
        args = parse_args()

        if args.conf != self.config.confidence_threshold:
            self.config.confidence_threshold = args.conf
            print(f"🔧 Confidence threshold set to: {args.conf}")

        if args.model != self.config.model_path:
            self.config.model_path = args.model
            print(f"🔧 Model set to: {args.model}")

        if args.image is not None:
            print(f"[CLI Mode] Detecting static image: {args.image}")
            self._run_static_detection(args.image, output_path=args.output)
            if args.stats:
                self._generate_statistics(args.export_json)
        elif args.camera:
            print("[CLI Mode] Starting live camera detection...")
            self._run_camera_detection(camera_index=args.cam_index,
                                       output_interval=args.output_interval)
        elif args.batch is not None:
            print(f"[CLI Mode] Running batch detection on directory: {args.batch}")
            output_dir = args.output if args.output else os.path.join(args.batch, "test_picture")
            self._run_batch_detection(args.batch, output_dir)
            if args.stats:
                self._generate_statistics(args.export_json)
        elif args.video is not None:
            print(f"[CLI Mode] Processing video file: {args.video}")
            self._run_video_detection(args.video, output_path=args.output)
            if args.stats:
                self._generate_statistics(args.export_json)
        elif args.compare:
            print("[CLI Mode] Starting multi-model comparison...")
            self._run_model_comparison(args.models)
        else:
            self._interactive_menu()

    def _interactive_menu(self):
        """Display interactive text menu."""
        try:
            print("\n" + "=" * 40)
            print("🚀 YOLOv8 Detection System")
            print("=" * 40)
            print("1. Static Image Detection")
            print("2. Live Camera Detection")
            print("3. Batch Image Detection")
            print("4. Video File Detection")
            print("5. Switch Detection Model")
            print("6. Exit")
            choice = input("Please select an option (1-6): ").strip()
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
            self.video_file_detection()
        elif choice == "5":
            self._switch_model_interactive()
        elif choice == "6":
            print("Goodbye!")
        else:
            print("Invalid option. Please enter 1-6.")
            self._interactive_menu()

    def _choose_image_source(self):
        """Let user choose between default or custom image path."""
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
                return
            self._run_static_detection(default_path)
        elif sub_choice == "b":
            try:
                custom_path = input("Enter image path: ").strip()
                custom_path = os.path.expanduser(custom_path)
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

    def video_file_detection(self):
        """Interactive video file detection."""
        print("\n=== Video File Detection ===")
        video_path = input("Enter video file path: ").strip()

        if not os.path.exists(video_path):
            print(f"Error: File not found - {video_path}")
            return

        save_choice = input("Save output video? (y/n): ").lower()
        output_path = None
        if save_choice == 'y':
            output_path = input("Enter output video path: ").strip()

        print("\nProcessing video, press 'q' to stop...\n")
        self.video_detector.process_video_file(video_path, output_path)

        if output_path:
            print(f"\nDetection complete! Result saved to: {output_path}")
        else:
            print("\nDetection complete!")

    def _run_static_detection(self, image_path, output_path=None):
        """Run single image detection."""
        print(f"🔍 Detecting objects in: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            if not os.path.exists(image_path):
                print(f"❌ Path does not exist: {image_path}")
            elif not os.access(image_path, os.R_OK):
                print(f"❌ No read permission: {image_path}")
            else:
                print(f"❌ Unsupported or corrupted image format: {image_path}")
            return

        annotated_frame, _ = self.model_manager.get_current_engine().detect(frame)

        window_name = "YOLO Detection Result"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, annotated_frame)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if output_path:
            save_path = output_path
        else:
            ext = ".jpg" if image_path.lower().endswith(".jpg") else ".png"
            save_path = image_path.replace(ext, f"_detected{ext}")

        try:
            success = cv2.imwrite(save_path, annotated_frame)
            if success:
                print(f"✅ Result saved to: {save_path}")
            else:
                print("❌ Failed to save result")
        except Exception as e:
            print(f"⚠️ Failed to save result: {e}")

    def _run_camera_detection(self, camera_index=None, output_interval=None):
        """Run camera detection."""
        try:
            from camera_detector import CameraDetector
            idx = camera_index if camera_index is not None else self.config.camera_index
            interval = output_interval if output_interval is not None else self.config.output_interval

            detector = CameraDetector(
                detection_engine=self.model_manager.get_current_engine(),
                output_interval=interval
            )
            detector.start_detection(camera_index=idx)
        except CameraOpenError as e:
            print(f"❌ Camera error: {e}")
        except Exception as e:
            print(f"💥 Camera detection failed: {e}")
            traceback.print_exc()

    def _run_video_detection(self, video_path, output_path=None):
        """Run video file detection."""
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return

        print(f"\nProcessing video: {video_path}")
        print("Press 'q' to quit...")
        self.video_detector.process_video_file(video_path, output_path)

        if output_path:
            print(f"\n✅ Video processing completed. Result saved to: {output_path}")
        else:
            print("\n✅ Video processing completed.")

    def _run_model_comparison(self, models=None):
        """Run multi-model comparison."""
        try:
            from model_comparison import ModelComparison

            if not models:
                models = ['yolov8n.pt', 'yolov8s.pt']
                print(f"⚠️ No models specified, using defaults: {models}")

            print(f"🔄 Loading models for comparison: {models}")
            comparison = ModelComparison(models, conf_threshold=self.config.confidence_threshold)

            available_models = comparison.get_available_models()
            if not available_models:
                print("❌ No models loaded successfully")
                return

            test_image_path = self.config.default_image_path
            if os.path.exists(test_image_path):
                print(f"\n📊 Running comparison on: {test_image_path}")
                comparison.compare_on_image(test_image_path)
            else:
                print(f"⚠️ Test image not found, skipping image comparison")

            print("\n" + "=" * 70)
            print("📊 Multi-Model Comparison Report")
            print("=" * 70)
            print(comparison.get_comparison_summary())

            comparison.export_to_json("model_comparison_report.json")
            print("\n✅ Comparison report saved to: model_comparison_report.json")

        except Exception as e:
            print(f"❌ Model comparison failed: {e}")
            traceback.print_exc()

    def _generate_statistics(self, export_path=None):
        """Generate statistics report."""
        try:
            from stats_analyzer import DetectionStatsAnalyzer

            analyzer = DetectionStatsAnalyzer()
            print("\n📊 Generating statistics report...")
            print(analyzer.generate_report())

            if export_path:
                analyzer.export_to_json(export_path)
                print(f"✅ Statistics exported to: {export_path}")
            else:
                analyzer.export_to_json("detection_stats.json")
                print("✅ Statistics exported to: detection_stats.json")

        except Exception as e:
            print(f"❌ Statistics generation failed: {e}")
            traceback.print_exc()

    def _run_batch_detection_interactive(self):
        """Interactive batch detection."""
        try:
            input_dir = input("Enter input directory path: ").strip()
            input_dir = os.path.expanduser(input_dir)
        except KeyboardInterrupt:
            return

        if not os.path.isdir(input_dir):
            print(f"❌ Directory not found: {input_dir}")
            return

        output_dir = os.path.join(input_dir, "test_picture")
        self._run_batch_detection(input_dir, output_dir)

    def _run_batch_detection(self, input_dir, output_dir=None):
        """Run batch detection."""
        if output_dir is None:
            output_dir = os.path.join(input_dir, "test_picture")

        try:
            from batch_detector import BatchDetector
            detector = BatchDetector(
                detection_engine=self.model_manager.get_current_engine(),
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
        """Interactive model switching."""
        print("\n--- Switch Detection Model ---")
        print("Examples:")
        print("  • yolov8n.pt   (smallest, fastest)")
        print("  • yolov8s.pt   (balanced)")
        print("  • yolov8m.pt   (more accurate)")
        print("  • ./models/custom.pt  (custom model)")
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
