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

# ui_handler.py
# 功能：用户交互调度中心（User Interface Handler）
# 优化版本：增强用户体验，提供更友好的交互界面

import os
import sys
import cv2
import argparse
import traceback
from datetime import datetime

from detection_engine import DetectionEngine, ModelLoadError
from camera_detector import CameraOpenError
from model_manager import ModelManager
from video_detector import VideoDetector


class Colors:
    """终端颜色代码"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def colored(text, color):
    """为文本添加颜色"""
    return f"{color}{text}{Colors.ENDC}"


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🚀 YOLOv8 Image Object Detection System v2.0            ║
║                                                              ║
║     图像目标检测系统 - 交互式菜单                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(colored(banner, Colors.CYAN))


def print_menu():
    """打印主菜单"""
    print(colored("\n📋 主菜单", Colors.BOLD))
    print("─" * 50)
    print(colored("  1.", Colors.GREEN) + " 📷 单图像检测")
    print(colored("  2.", Colors.GREEN) + " 📹 摄像头实时检测")
    print(colored("  3.", Colors.GREEN) + " 📁 批量图像检测")
    print(colored("  4.", Colors.GREEN) + " 🎬 视频文件检测")
    print(colored("  5.", Colors.GREEN) + " ⚖️ 多模型对比")
    print(colored("  6.", Colors.YELLOW) + " ⚙️  设置")
    print(colored("  7.", Colors.YELLOW) + " 📊 统计分析")
    print(colored("  8.", Colors.BLUE) + " ℹ️  系统信息")
    print(colored("  0.", Colors.RED) + " ❌ 退出")
    print("─" * 50)


def print_settings_menu():
    """打印设置子菜单"""
    print(colored("\n⚙️  设置菜单", Colors.BOLD))
    print("─" * 50)
    print(colored("  1.", Colors.GREEN) + " 🔄 切换检测模型")
    print(colored("  2.", Colors.GREEN) + " 🎚️ 设置置信度阈值")
    print(colored("  3.", Colors.GREEN) + " 📁 设置默认路径")
    print(colored("  4.", Colors.YELLOW) + " 📋 查看当前配置")
    print(colored("  0.", Colors.RED) + " ↩️  返回主菜单")
    print("─" * 50)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLOv8 Image Object Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--image", type=str, help="单图像检测")
    mode_group.add_argument("--camera", action="store_true", help="摄像头检测")
    mode_group.add_argument("--batch", type=str, help="批量检测")
    mode_group.add_argument("--video", type=str, help="视频检测")
    mode_group.add_argument("--compare", action="store_true", help="多模型对比")

    parser.add_argument("--model", type=str, default="yolov8n.pt", help="模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--models", type=str, nargs='+', help="多模型列表")
    parser.add_argument("--output", type=str, help="输出路径")
    parser.add_argument("--stats", action="store_true", help="生成统计")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    parser.add_argument("--debug", action="store_true", help="调试模式")

    return parser.parse_args()


class UIHandler:
    """用户界面控制器 - 优化版本"""

    def __init__(self, config):
        self.config = config
        self.video_detector = VideoDetector()
        self.operation_history = []
        
        try:
            self.model_manager = ModelManager(
                initial_model_path=config.model_path,
                conf_threshold=config.confidence_threshold
            )
            self._add_history("系统初始化成功")
        except Exception as e:
            print(colored(f"❌ 系统初始化失败: {e}", Colors.RED))
            raise SystemExit(1)

    def _add_history(self, operation):
        """添加操作历史"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.operation_history.append(f"[{timestamp}] {operation}")
        if len(self.operation_history) > 10:
            self.operation_history.pop(0)

    def run(self):
        """主入口"""
        args = parse_args()

        if not args.quiet:
            print_banner()

        if args.conf != self.config.confidence_threshold:
            self.config.confidence_threshold = args.conf
            self._add_history(f"置信度阈值: {args.conf}")

        if args.model != self.config.model_path:
            self.config.model_path = args.model
            self._add_history(f"模型切换: {args.model}")

        if args.image is not None:
            self._run_static_detection(args.image, output_path=args.output)
            if args.stats:
                self._generate_statistics(args.export_json)
        elif args.camera:
            self._run_camera_detection()
        elif args.batch is not None:
            self._run_batch_detection(args.batch, args.output)
            if args.stats:
                self._generate_statistics(args.export_json)
        elif args.video is not None:
            self._run_video_detection(args.video, args.output)
            if args.stats:
                self._generate_statistics(args.export_json)
        elif args.compare:
            self._run_model_comparison(args.models)
        else:
            self._interactive_menu()

    def _interactive_menu(self):
        """交互式菜单"""
        while True:
            try:
                print_menu()
                choice = input(colored("请输入选项: ", Colors.CYAN)).strip()

                if choice == "1":
                    self._choose_image_source()
                elif choice == "2":
                    self._run_camera_detection()
                elif choice == "3":
                    self._run_batch_detection_interactive()
                elif choice == "4":
                    self._video_file_detection_interactive()
                elif choice == "5":
                    self._model_comparison_interactive()
                elif choice == "6":
                    self._settings_menu()
                elif choice == "7":
                    self._show_statistics()
                elif choice == "8":
                    self._show_system_info()
                elif choice == "0":
                    self._exit_system()
                    break
                else:
                    print(colored("⚠️  无效选项，请输入 0-8", Colors.YELLOW))

            except KeyboardInterrupt:
                print(colored("\n\n⚠️  检测到中断信号，输入 0 退出或继续操作...", Colors.YELLOW))

    def _settings_menu(self):
        """设置子菜单"""
        while True:
            try:
                print_settings_menu()
                choice = input(colored("请输入选项: ", Colors.CYAN)).strip()

                if choice == "1":
                    self._switch_model_interactive()
                elif choice == "2":
                    self._set_confidence_interactive()
                elif choice == "3":
                    self._set_default_path_interactive()
                elif choice == "4":
                    self._show_current_config()
                elif choice == "0":
                    break
                else:
                    print(colored("⚠️  无效选项", Colors.YELLOW))

            except KeyboardInterrupt:
                break

    def _choose_image_source(self):
        """选择图像来源"""
        print(colored("\n📷 单图像检测", Colors.BOLD))
        print("─" * 30)
        print(f"  a) 使用默认图像: {self.config.default_image_path}")
        print(f"  b) 输入自定义路径")
        print(f"  c) 浏览当前目录")
        print(f"  0) 返回主菜单")

        try:
            choice = input(colored("\n请选择: ", Colors.CYAN)).strip().lower()

            if choice == "0":
                return
            elif choice == "a":
                if os.path.exists(self.config.default_image_path):
                    self._run_static_detection(self.config.default_image_path)
                else:
                    print(colored(f"⚠️  默认图像不存在: {self.config.default_image_path}", Colors.YELLOW))
            elif choice == "b":
                path = input(colored("请输入图像路径: ", Colors.CYAN)).strip()
                path = os.path.expanduser(path)
                if os.path.exists(path):
                    self._run_static_detection(path)
                else:
                    print(colored(f"❌ 文件不存在: {path}", Colors.RED))
            elif choice == "c":
                self._browse_directory()
            else:
                print(colored("⚠️  无效选项", Colors.YELLOW))

        except KeyboardInterrupt:
            pass

    def _browse_directory(self):
        """浏览目录"""
        current_dir = self.config.input_dir if os.path.isdir(self.config.input_dir) else "."
        
        if not os.path.isdir(current_dir):
            current_dir = "."
        
        print(colored(f"\n📁 浏览目录: {current_dir}", Colors.BOLD))
        
        try:
            files = []
            for f in os.listdir(current_dir):
                full_path = os.path.join(current_dir, f)
                if os.path.isfile(full_path):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                        files.append(f)
            
            if files:
                print(colored("\n可用图像文件:", Colors.GREEN))
                for i, f in enumerate(files[:10], 1):
                    print(f"  {i}. {f}")
                if len(files) > 10:
                    print(f"  ... 还有 {len(files) - 10} 个文件")
                
                choice = input(colored("\n选择文件编号 (0返回): ", Colors.CYAN)).strip()
                if choice.isdigit() and 1 <= int(choice) <= len(files):
                    selected = files[int(choice) - 1]
                    self._run_static_detection(os.path.join(current_dir, selected))
            else:
                print(colored("⚠️  该目录下没有图像文件", Colors.YELLOW))

        except Exception as e:
            print(colored(f"❌ 浏览失败: {e}", Colors.RED))

    def _video_file_detection_interactive(self):
        """视频检测交互"""
        print(colored("\n🎬 视频文件检测", Colors.BOLD))
        print("─" * 30)

        video_path = input(colored("请输入视频路径: ", Colors.CYAN)).strip()
        video_path = os.path.expanduser(video_path)

        if not os.path.exists(video_path):
            print(colored(f"❌ 文件不存在: {video_path}", Colors.RED))
            return

        save_choice = input(colored("是否保存输出视频? (y/n): ", Colors.CYAN)).strip().lower()
        output_path = None
        if save_choice == 'y':
            output_path = input(colored("请输入输出路径: ", Colors.CYAN)).strip()

        print(colored("\n⏳ 正在处理视频，按 'q' 键退出...", Colors.YELLOW))
        self.video_detector.process_video_file(video_path, output_path)

        if output_path:
            print(colored(f"✅ 视频处理完成，已保存至: {output_path}", Colors.GREEN))
        else:
            print(colored("✅ 视频处理完成", Colors.GREEN))

    def _model_comparison_interactive(self):
        """多模型对比交互"""
        print(colored("\n⚖️  多模型对比测试", Colors.BOLD))
        print("─" * 30)
        print("可用模型:")
        print("  1. yolov8n.pt (最小、最快)")
        print("  2. yolov8s.pt (较小、较快)")
        print("  3. yolov8m.pt (中等、平衡)")
        print("  4. 自定义模型")

        try:
            choice = input(colored("\n选择模型组合 (如 1 2 或 1 3): ", Colors.CYAN)).strip()
            
            model_map = {
                "1": "yolov8n.pt",
                "2": "yolov8s.pt",
                "3": "yolov8m.pt",
            }
            
            models = []
            for c in choice.split():
                if c in model_map:
                    models.append(model_map[c])
                elif c == "4":
                    custom = input(colored("请输入自定义模型路径: ", Colors.CYAN)).strip()
                    if custom:
                        models.append(custom)
            
            if not models:
                models = ["yolov8n.pt", "yolov8s.pt"]
                print(colored("⚠️  未选择有效模型，使用默认组合", Colors.YELLOW))
            
            self._run_model_comparison(models)

        except KeyboardInterrupt:
            pass

    def _set_confidence_interactive(self):
        """设置置信度"""
        print(colored(f"\n🎚️  设置置信度阈值 (当前: {self.config.confidence_threshold})", Colors.BOLD))
        print("  提示: 较低的值会检测更多目标，但可能包含更多误检")
        
        try:
            value = input(colored("请输入 0.0-1.0 之间的值 (直接回车取消): ", Colors.CYAN)).strip()
            if value:
                new_conf = float(value)
                if 0.0 <= new_conf <= 1.0:
                    self.config.confidence_threshold = new_conf
                    self._add_history(f"置信度: {new_conf}")
                    print(colored(f"✅ 置信度已设置为: {new_conf}", Colors.GREEN))
                else:
                    print(colored("❌ 值必须在 0.0-1.0 范围内", Colors.RED))
        except ValueError:
            print(colored("❌ 无效的数值", Colors.RED))
        except KeyboardInterrupt:
            pass

    def _set_default_path_interactive(self):
        """设置默认路径"""
        print(colored("\n📁 设置默认路径", Colors.BOLD))
        print(f"  当前: {self.config.default_image_path}")
        
        try:
            new_path = input(colored("请输入新路径 (直接回车取消): ", Colors.CYAN)).strip()
            if new_path:
                new_path = os.path.expanduser(new_path)
                if os.path.exists(new_path):
                    self.config.default_image_path = new_path
                    self._add_history(f"默认路径: {new_path}")
                    print(colored(f"✅ 默认路径已更新", Colors.GREEN))
                else:
                    print(colored("❌ 路径不存在", Colors.RED))
        except KeyboardInterrupt:
            pass

    def _show_current_config(self):
        """显示当前配置"""
        print(colored("\n📋 当前配置", Colors.BOLD))
        print("─" * 40)
        self.config.print_summary()
        input(colored("\n按回车键继续...", Colors.CYAN))

    def _show_statistics(self):
        """显示统计信息"""
        print(colored("\n📊 统计分析", Colors.BOLD))
        print("─" * 40)
        
        if self.operation_history:
            print(colored("最近操作:", Colors.GREEN))
            for op in self.operation_history[-5:]:
                print(f"  {op}")
        else:
            print("暂无操作记录")
        
        input(colored("\n按回车键继续...", Colors.CYAN))

    def _show_system_info(self):
        """显示系统信息"""
        print(colored("\nℹ️  系统信息", Colors.BOLD))
        print("─" * 40)
        print(f"  当前模型: {self.config.model_path}")
        print(f"  置信度阈值: {self.config.confidence_threshold}")
        print(f"  默认图像: {self.config.default_image_path}")
        print(f"  支持格式: {', '.join(self.config.supported_formats)}")
        print(f"  PyTorch: {'✓' if self._check_pytorch() else '✗'}")
        print(f"  OpenCV: {'✓' if self._check_opencv() else '✗'}")
        print(f"  Ultralytics: {'✓' if self._check_ultralytics() else '✗'}")
        print("─" * 40)
        input(colored("\n按回车键继续...", Colors.CYAN))

    def _check_pytorch(self):
        try:
            import torch
            return torch.__version__
        except:
            return False

    def _check_opencv(self):
        try:
            return cv2.__version__
        except:
            return False

    def _check_ultralytics(self):
        try:
            import ultralytics
            return ultralytics.__version__
        except:
            return False

    def _exit_system(self):
        """退出系统"""
        print(colored("\n" + "=" * 40, Colors.CYAN))
        print(colored("感谢使用 YOLOv8 检测系统!", Colors.GREEN))
        print(colored("再见! 👋", Colors.CYAN))
        print(colored("=" * 40, Colors.CYAN))

    def _run_static_detection(self, image_path, output_path=None):
        """执行单图像检测"""
        self._add_history(f"检测图像: {os.path.basename(image_path)}")
        print(colored(f"\n🔍 正在检测: {image_path}", Colors.YELLOW))
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(colored(f"❌ 无法读取图像: {image_path}", Colors.RED))
            return

        annotated_frame, results = self.model_manager.get_current_engine().detect(frame)

        detection_count = len(results[0].boxes) if results and results[0].boxes is not None else 0
        print(colored(f"✅ 检测完成! 发现 {detection_count} 个目标", Colors.GREEN))

        window_name = "YOLO Detection Result"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, annotated_frame)
        print(colored("按任意键关闭窗口...", Colors.CYAN))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if output_path:
            save_path = output_path
        else:
            ext = os.path.splitext(image_path)[1] or ".jpg"
            save_path = image_path.replace(ext, f"_detected{ext}")

        try:
            if cv2.imwrite(save_path, annotated_frame):
                print(colored(f"💾 结果已保存: {save_path}", Colors.GREEN))
            else:
                print(colored("❌ 保存失败", Colors.RED))
        except Exception as e:
            print(colored(f"❌ 保存失败: {e}", Colors.RED))

    def _run_camera_detection(self):
        """执行摄像头检测"""
        self._add_history("摄像头检测")
        print(colored("\n📹 启动摄像头检测...", Colors.YELLOW))
        print(colored("按 'q' 键退出", Colors.CYAN))

        try:
            from camera_detector import CameraDetector
            detector = CameraDetector(
                detection_engine=self.model_manager.get_current_engine(),
                output_interval=self.config.output_interval
            )
            detector.start_detection(camera_index=self.config.camera_index)
        except CameraOpenError as e:
            print(colored(f"❌ 摄像头错误: {e}", Colors.RED))
        except Exception as e:
            print(colored(f"❌ 检测失败: {e}", Colors.RED))

    def _run_batch_detection(self, input_dir, output_dir=None):
        """执行批量检测"""
        if not os.path.isdir(input_dir):
            print(colored(f"❌ 目录不存在: {input_dir}", Colors.RED))
            return

        self._add_history(f"批量检测: {input_dir}")
        
        if output_dir is None:
            output_dir = os.path.join(input_dir, "output")
        
        print(colored(f"\n📁 开始批量检测...", Colors.YELLOW))
        print(colored(f"   输入: {input_dir}", Colors.CYAN))
        print(colored(f"   输出: {output_dir}", Colors.CYAN))

        try:
            from batch_detector import BatchDetector
            detector = BatchDetector(
                detection_engine=self.model_manager.get_current_engine(),
                input_dir=input_dir,
                output_dir=output_dir
            )
            detector.run()
            print(colored("✅ 批量检测完成!", Colors.GREEN))
        except Exception as e:
            print(colored(f"❌ 批量检测失败: {e}", Colors.RED))

    def _run_batch_detection_interactive(self):
        """批量检测交互"""
        print(colored("\n📁 批量图像检测", Colors.BOLD))
        print("─" * 30)
        
        input_dir = input(colored("请输入图像目录路径: ", Colors.CYAN)).strip()
        input_dir = os.path.expanduser(input_dir)

        if not os.path.isdir(input_dir):
            print(colored(f"❌ 目录不存在: {input_dir}", Colors.RED))
            return

        output_dir = os.path.join(input_dir, "output")
        self._run_batch_detection(input_dir, output_dir)

    def _run_video_detection(self, video_path, output_path=None):
        """执行视频检测"""
        self._add_history(f"视频检测: {os.path.basename(video_path)}")
        print(colored(f"\n🎬 正在处理视频: {video_path}", Colors.YELLOW))

        self.video_detector.process_video_file(video_path, output_path)

        if output_path:
            print(colored(f"✅ 视频处理完成: {output_path}", Colors.GREEN))

    def _run_model_comparison(self, models=None):
        """执行多模型对比"""
        if not models:
            models = ['yolov8n.pt', 'yolov8s.pt']

        self._add_history(f"多模型对比: {', '.join(models)}")
        print(colored(f"\n⚖️  正在加载模型: {models}", Colors.YELLOW))

        try:
            from model_comparison import ModelComparison
            comparison = ModelComparison(models, conf_threshold=self.config.confidence_threshold)

            available = comparison.get_available_models()
            if not available:
                print(colored("❌ 没有模型加载成功", Colors.RED))
                return

            print(colored(f"✅ 成功加载 {len(available)} 个模型", Colors.GREEN))

            test_img = self.config.default_image_path
            if os.path.exists(test_img):
                comparison.compare_on_image(test_img)

            print("\n" + "=" * 60)
            print(colored("📊 多模型对比报告", Colors.BOLD))
            print("=" * 60)
            print(comparison.get_comparison_summary())

            comparison.export_to_json("model_comparison_report.json")
            print(colored("\n✅ 报告已保存: model_comparison_report.json", Colors.GREEN))

        except Exception as e:
            print(colored(f"❌ 对比失败: {e}", Colors.RED))

    def _generate_statistics(self, export_path=None):
        """生成统计报告"""
        try:
            from stats_analyzer import DetectionStatsAnalyzer
            analyzer = DetectionStatsAnalyzer()
            print(colored("\n📊 正在生成统计报告...", Colors.YELLOW))
            print(analyzer.generate_report())

            path = export_path or "detection_stats.json"
            analyzer.export_to_json(path)
            print(colored(f"✅ 报告已导出: {path}", Colors.GREEN))
        except Exception as e:
            print(colored(f"❌ 统计生成失败: {e}", Colors.RED))
