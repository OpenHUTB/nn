#!/usr/bin/env python3
"""
简化启动脚本 - 测试UI是否正常
"""
import pygame
import sys
import os

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.join(current_dir, 'modules')

if modules_dir not in sys.path:
    sys.path.insert(0, modules_dir)

print("=" * 60)
print("🚀 测试UI控制器")
print("=" * 60)


def test_ui():
    """测试UI控制器"""
    try:
        from ui_controller import UIController
        print("导入UI控制器成功")

        # 创建UI实例
        ui = UIController()
        print("UI实例创建成功")

        # 运行简单的测试循环
        print("开始UI测试循环...")
        print("按Q或ESC退出")

        clock = pygame.time.Clock()
        running = True

        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

            # 创建一个测试图像
            import numpy as np
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            test_image[200:280, 280:360] = [255, 0, 0]  # 红色方块
            test_image[100:180, 100:180] = [0, 255, 0]  # 绿色方块

            # 更新UI
            ui.update_lightweight(test_image)

            # 更新无人机状态（模拟）
            ui.update_drone_state({
                'drone_status': '已连接',
                'is_flying': True,
                'drone_position': (1.5, 2.3, 1.8),
                'drone_yaw': 45.0,
                'tracking_mode': '手动',
                'camera_status': '640x480 @ 30fps',
                'detected_faces': 2,
                'detected_persons': 3,
                'recognized_person': '张三',
                'fps': 30,
            })

            clock.tick(30)

        ui.quit()
        print("✅ UI测试完成")

    except Exception as e:
        print(f"❌ UI测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_main():
    """测试主程序"""
    try:
        print("\n测试主程序导入...")
        # 尝试导入各个模块
        from drone_controller import DroneController
        print("  ✅ drone_controller")

        from face_detector import FaceDetector
        print("  ✅ face_detector")

        from person_detector import PersonDetector
        print("  ✅ person_detector")

        from face_recognizer import FaceRecognizer
        print("  ✅ face_recognizer")

        print("✅ 所有模块导入成功")
        return True

    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False


if __name__ == "__main__":
    print("\n选择测试模式:")
    print("1. 测试UI控制器")
    print("2. 测试模块导入")
    print("3. 退出")

    choice = input("请选择 (1-3): ").strip()

    if choice == "1":
        test_ui()
    elif choice == "2":
        test_main()
    elif choice == "3":
        print("退出")
    else:
        print("无效选择")

    print("\n程序结束")