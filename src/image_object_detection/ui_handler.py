# ui_handler.py
import cv2
from detection_engine import DetectionEngine
from image_detector import ImageDetector
from camera_detector import CameraDetector
import sys
import os

class UIHandler:
    def __init__(self, config):
        self.config = config
        self.detection_engine = DetectionEngine()
        self.image_detector = ImageDetector(self.detection_engine)
        self.camera_detector = CameraDetector(self.detection_engine)
    
    def print_debug(self, message):
        """调试输出函数"""
        print(f"[DEBUG] {message}")
        sys.stdout.flush()  # 强制刷新输出缓冲区
    
    def display_menu(self):
        """显示主菜单"""
        print("=" * 60)
        print("           欢迎使用 YOLO 对象检测系统")
        print("=" * 60)
        
        print(f"\n测试图像路径: {self.config.test_image_path}")
        print(f"图像文件存在: {os.path.exists(self.config.test_image_path)}")
        
        print("\n请选择检测模式:")
        print("1. 检测静态图像")
        print("2. 检测摄像头实时画面")
        print("3. 退出程序")
    
    def handle_choice(self, choice):
        """处理用户选择"""
        if choice == '1':
            print("\n您选择了: 检测静态图像")
            self.image_detector.detect_static_image(self.config.test_image_path)
            return True
        elif choice == '2':
            print("\n您选择了: 检测摄像头实时画面")
            print("注意：要退出摄像头模式，请按 Ctrl+C ！")
            self.camera_detector.detect_camera()
            return True
        elif choice == '3':
            print("\n程序已退出。")
            return False
        else:
            print("\n无效选择，请输入 1、2 或 3")
            return True
    
    def run(self):
        """运行主程序"""
        self.print_debug("程序开始执行")
        self.display_menu()
        
        running = True
        while running:
            try:
                choice = input("\n请输入您的选择 (1/2/3): ").strip()
                self.print_debug(f"用户输入: {choice}")
                running = self.handle_choice(choice)
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断。")
                break
            except EOFError:
                print("\n输入结束，程序退出。")
                break
            except Exception as e:
                print(f"\n发生错误: {e}")
                break
        
        self.print_debug("程序结束")