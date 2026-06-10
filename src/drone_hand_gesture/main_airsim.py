# -*- coding: utf-8 -*-
"""
手势控制无人机 - AirSim 真实模拟器版
基于 drone_hand_gesture 项目，添加 AirSim 集成
"""

import cv2
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from airsim_controller import AirSimController

# 优先使用增强版检测器
USE_ENHANCED_DETECTOR = False
HAS_ENHANCED = False

try:
    import mediapipe as mp
    from gesture_detector_enhanced import EnhancedGestureDetector
    USE_ENHANCED_DETECTOR = True
    HAS_ENHANCED = True
except ImportError:
    try:
        from gesture_detector_cv import CVGestureDetector
        from gesture_detector import GestureDetector
        print("[INFO] MediaPipe 不可用，使用 OpenCV 检测器")
    except ImportError:
        from gesture_detector import GestureDetector


def main(show_window=True):
    """主函数"""
    print("=" * 70)
    print("手势控制无人机 - AirSim 真实模拟器版")
    print("=" * 70)
    print()
    
    # 1. 连接 AirSim
    print("[1/4] 正在连接 AirSim 模拟器...")
    controller = AirSimController(ip_address="127.0.0.1", port=41451)
    
    if controller.connect():
        print("[OK] AirSim 连接成功")
    else:
        print("\n[ERROR] AirSim 连接失败")
        print("\n请检查:")
        print("  1. AirSim 模拟器是否运行")
        print("  2. 防火墙设置")
        print("\n按回车键退出...")
        input()
        return
    
    # 2. 初始化手势检测器
    print("\n[2/4] 正在初始化手势检测器...")
    
    # 查找机器学习模型
    model_candidates = [
        "dataset/models/gesture_svm.pkl",
        "dataset/models/gesture_random_forest.pkl",
        "dataset/models/gesture_mlp.pkl",
    ]
    
    selected_model = None
    for model_path in model_candidates:
        if os.path.exists(model_path):
            selected_model = model_path
            print(f"[INFO] 找到模型: {model_path}")
            break
    
    # 初始化检测器
    if USE_ENHANCED_DETECTOR and HAS_ENHANCED:
        if selected_model:
            detector = EnhancedGestureDetector(ml_model_path=selected_model, use_ml=True)
            print("[OK] 使用增强版检测器（机器学习模式）")
        else:
            detector = EnhancedGestureDetector(use_ml=False)
            print("[OK] 使用增强版检测器（规则模式）")
    else:
        try:
            from gesture_detector_cv import CVGestureDetector
            detector = CVGestureDetector()
            detector.use_ml = False
            print("[OK] 使用 OpenCV 检测器")
        except:
            from gesture_detector import GestureDetector
            detector = GestureDetector()
            print("[OK] 使用基础手势检测器")
    
    print("[OK] 手势检测器就绪")
    
    # 3. 初始化摄像头（修复卡死问题）
    print("\n[3/4] 正在初始化摄像头...")
    # 尝试多个摄像头ID
    cap = None
    for cam_id in [1, 0]:
        print(f"  尝试打开摄像头 {cam_id}...")
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print(f"[OK] 摄像头 {cam_id} 已就绪")
                break
            else:
                cap.release()
                cap = None
    
    if not cap or not cap.isOpened():
        print("[ERROR] 摄像头不可用")
        controller.disconnect()
        return
    
    # 使用正常分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 手势阈值配置
    gesture_thresholds = {
        'open_palm': 0.5,
        'closed_fist': 0.55,
        'victory': 0.55,
        'thumb_up': 0.55,
        'thumb_down': 0.55,
        'pointing_up': 0.5,
        'pointing_down': 0.5,
        'ok_sign': 0.6,
        'default': 0.5
    }
    
    # 4. 系统就绪
    print("\n[4/4] 系统就绪！")
    print("\n" + "=" * 70)
    print("手势控制:")
    print("  张开手掌   - 起飞")
    print("  握拳       - 降落")
    print("  食指上指   - 上升")
    print("  食指下指   - 下降")
    print("  胜利手势   - 前进")
    print("  大拇指     - 后退")
    print("  OK手势     - 悬停")
    print("  大拇指向下 - 停止")
    print("\n键盘控制:")
    print("  空格键 - 起飞/降落")
    print("  T     - 手动起飞")
    print("  L     - 手动降落")
    print("  H     - 悬停")
    print("  Q/ESC - 退出程序")
    print("=" * 70)
    print()
    
    # 主循环（恢复每帧检测）
    is_flying = False
    last_command_time = 0
    last_processed_gesture = ""
    last_processed_time = 0
    command_cooldown = 1.5
    frame_count = 0
    start_time = time.time()
    
    print("[INFO] 按 空格键 或 T 键 起飞")
    print("[INFO] 调试模式: 显示手势检测信息")
    
    try:
        while True:
            ret, gesture_frame = cap.read()
            if not ret:
                print("[WARNING] 无法读取手势摄像头画面")
                break
            
            # 镜像翻转画面
            gesture_frame = cv2.flip(gesture_frame, 1)
            frame_count += 1
            
            # 每帧检测手势（流畅响应）
            debug_frame, gesture, confidence, _ = detector.detect_gestures(gesture_frame, simulation_mode=False)
            
            # 处理手势
            current_time = time.time()
            in_cooldown = current_time - last_command_time <= command_cooldown
            same_gesture = (gesture == last_processed_gesture and
                           current_time - last_processed_time < 2.0)
            
            command = detector.get_command(gesture)
            threshold = gesture_thresholds.get(gesture, gesture_thresholds['default'])
            
            if (gesture not in ["no_hand", "hand_detected", "none"]
                    and confidence > threshold
                    and not in_cooldown
                    and not same_gesture
                    and command != "none"):
                
                print(f"[CMD] 手势: {gesture} (置信度: {confidence:.2f}) -> 执行: {command}")
                controller.send_command(command)
                
                if command == "takeoff":
                    is_flying = True
                elif command == "land":
                    is_flying = False
                
                last_command_time = current_time
                last_processed_gesture = gesture
                last_processed_time = current_time
            
            # 显示帧率
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Gesture: {gesture} ({confidence:.2f})", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示画面
            if show_window:
                cv2.imshow("Gesture Control - AirSim", debug_frame)
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q') or key == 27:
                    print("\n[INFO] 退出程序...")
                    break
                
                elif key == ord(' '):
                    if is_flying:
                        print("[INFO] 降落...")
                        controller.land()
                        is_flying = False
                    else:
                        print("[INFO] 起飞...")
                        controller.takeoff()
                        is_flying = True
                    time.sleep(0.5)
                
                elif key == ord('t') or key == ord('T'):
                    if not is_flying:
                        print("[INFO] 手动起飞...")
                        controller.takeoff()
                        is_flying = True
                
                elif key == ord('l') or key == ord('L'):
                    if is_flying:
                        print("[INFO] 手动降落...")
                        controller.land()
                        is_flying = False
                
                elif key == ord('h') or key == ord('H'):
                    print("[INFO] 悬停")
                    controller.hover()
            else:
                # 没有显示窗口时，使用一个简单的循环来模拟
                time.sleep(0.1)
                # 检查是否需要退出
                if frame_count > 300:  # 运行 5 秒后自动退出
                    print("\n[INFO] 自动退出程序...")
                    break
            
            # 显示状态
            if is_flying and frame_count % 30 == 0:
                state = controller.get_state()
                print(f"[状态] 高度: {state['position'][2]:.2f}米")
    
    except KeyboardInterrupt:
        print("\n[INFO] 程序中断")
    
    finally:
        print("\n[INFO] 清理资源...")
        
        if is_flying:
            print("[INFO] 正在降落...")
            controller.land()
        
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        controller.disconnect()
        
        if hasattr(detector, "release"):
            detector.release()
        
        print("[OK] 程序安全退出")


if __name__ == "__main__":
    main()