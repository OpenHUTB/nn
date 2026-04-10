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
from gesture_detector_enhanced import EnhancedGestureDetector


def main():
    """主函数"""
    print("=" * 70)
    print("手势控制无人机 - AirSim 真实模拟器版")
    print("=" * 70)
    print()
    
    # 1. 连接 AirSim
    print("[1/4] 正在连接 AirSim 模拟器...")
    controller = AirSimController(ip_address="127.0.0.1", port=41451)
    
    if not controller.connect():
        print("\n[ERROR] AirSim 连接失败")
        print("\n请检查:")
        print("  1. AirSim 模拟器是否运行")
        print("  2. 防火墙设置")
        print("\n按回车键退出...")
        input()
        return
    
    # 2. 初始化手势检测器
    print("\n[2/4] 正在初始化手势检测器...")
    detector = EnhancedGestureDetector(use_ml=False)
    print("[OK] 手势检测器就绪")
    
    # 3. 初始化摄像头
    print("\n[3/4] 正在初始化摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] 摄像头不可用")
        controller.disconnect()
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("[OK] 摄像头已就绪：640x480 @ 30fps")
    
    # 4. 系统就绪
    print("\n[4/4] 系统就绪！")
    print("\n" + "=" * 70)
    print("手势控制:")
    print("  张开手掌 - 悬停")
    print("  食指向上 - 上升")
    print("  食指向下 - 下降")
    print("  指向左侧 - 左移")
    print("  指向右侧 - 右移")
    print("  两指向前 - 前进")
    print("  握拳     - 降落")
    print("\n键盘控制:")
    print("  空格键 - 起飞/降落")
    print("  T     - 手动起飞")
    print("  L     - 手动降落")
    print("  H     - 悬停")
    print("  Q/ESC - 退出程序")
    print("=" * 70)
    print()
    
    # 主循环
    is_flying = False
    last_gesture_time = 0
    current_gesture = ""
    gesture_hold_duration = 1.0
    frame_count = 0
    start_time = time.time()
    
    print("[INFO] 按 空格键 或 T 键 起飞")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] 无法读取摄像头画面")
                break
            
            frame_count += 1
            
            # 手势识别
            debug_frame, gesture, confidence, _ = detector.detect_gestures(frame)
            
            # 显示帧率
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Gesture: {gesture} ({confidence:.2f})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 处理手势
            if gesture and gesture != "none" and gesture != "no_hand":
                current_time = time.time()
                
                if gesture == current_gesture:
                    if current_time - last_gesture_time > gesture_hold_duration:
                        print(f"[CMD] 手势：{gesture}")
                        
                        if gesture == 'fist':
                            if is_flying:
                                print("[INFO] 降落...")
                                controller.land()
                                is_flying = False
                            
                        elif gesture == 'point_up':
                            print("[INFO] 上升")
                            controller.move_by_velocity(0, 0, -1.0, duration=0.5)
                            
                        elif gesture == 'point_down':
                            print("[INFO] 下降")
                            controller.move_by_velocity(0, 0, 1.0, duration=0.5)
                            
                        elif gesture == 'point_left':
                            print("[INFO] 左移")
                            controller.move_by_velocity(-1.0, 0, 0, duration=0.5)
                            
                        elif gesture == 'point_right':
                            print("[INFO] 右移")
                            controller.move_by_velocity(1.0, 0, 0, duration=0.5)
                            
                        elif gesture == 'victory':
                            print("[INFO] 前进")
                            controller.move_by_velocity(0, 1.0, 0, duration=0.5)
                            
                        elif gesture == 'open_palm':
                            print("[INFO] 悬停")
                            controller.hover()
                        
                        last_gesture_time = current_time
                else:
                    current_gesture = gesture
                    last_gesture_time = current_time
            
            # 显示画面
            cv2.imshow('Gesture Control - AirSim', debug_frame)
            
            # 键盘控制
            key = cv2.waitKey(10) & 0xFF
            
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
            
            # 显示状态
            if is_flying and frame_count % 30 == 0:
                state = controller.get_state()
                print(f"[状态] 高度：{state['position'][2]:.2f}m")
    
    except KeyboardInterrupt:
        print("\n[INFO] 程序中断")
    
    finally:
        print("\n[INFO] 清理资源...")
        
        if is_flying:
            print("[INFO] 正在降落...")
            controller.land()
        
        cap.release()
        cv2.destroyAllWindows()
        controller.disconnect()
        
        print("[OK] 程序安全退出")


if __name__ == "__main__":
    main()
