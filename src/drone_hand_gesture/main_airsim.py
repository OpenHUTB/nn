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
from gesture_detector import GestureDetector


def main(show_window=True):
    """主函数"""
    print("=" * 70)
    print("手势控制无人机 - AirSim 真实模拟器版")
    print("=" * 70)
    print()
    
    # 1. 连接 AirSim
    print("[1/4] 正在连接 AirSim 模拟器...")
    print("[DEBUG] 正在创建 AirSimController 实例...")
    controller = AirSimController(ip_address="127.0.0.1", port=41451)
    print("[DEBUG] AirSimController 实例创建成功")
    
    print("[DEBUG] 正在连接 AirSim...")
    if not controller.connect():
        print("\n[ERROR] AirSim 连接失败")
        print("\n请检查:")
        print("  1. AirSim 模拟器是否运行")
        print("  2. 防火墙设置")
        print("\n按回车键退出...")
        input()
        return
    print("[DEBUG] AirSim 连接成功")
    
    # 2. 初始化手势检测器
    print("\n[2/4] 正在初始化手势检测器...")
    print("[DEBUG] 正在创建 GestureDetector 实例...")
    detector = GestureDetector()
    print("[DEBUG] GestureDetector 实例创建成功")
    print("[OK] 手势检测器就绪（规则检测）")
    
    # 3. 初始化摄像头
    print("\n[3/4] 正在初始化摄像头...")
    print("[DEBUG] 正在打开摄像头...")
    cap = cv2.VideoCapture(0)
    print("[DEBUG] 摄像头打开成功")
    
    if not cap.isOpened():
        print("[ERROR] 摄像头不可用")
        controller.disconnect()
        return
    
    print("[DEBUG] 正在设置摄像头参数...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("[DEBUG] 摄像头参数设置成功")
    print("[OK] 摄像头已就绪：640x480 @ 30fps")
    
    # 4. 系统就绪
    print("\n[4/4] 系统就绪！")
    print("\n" + "=" * 70)
    print("手势控制:")
    print("  张开手掌   - 起飞")
    print("  握拳       - 降落")
    print("  食指上指   - 上升")
    print("  食指向下   - 下降")
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
    
    # 主循环
    is_flying = False
    last_command_time = 0
    current_gesture = ""
    last_processed_gesture = ""
    last_processed_time = 0
    command_cooldown = 1.5  # 命令冷却时间（秒）
    gesture_threshold = 0.5  # 置信度阈值（gesture_detector返回0.75-0.95）
    frame_count = 0
    start_time = time.time()
    
    print("[INFO] 按 空格键 或 T 键 起飞")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] 无法读取摄像头画面")
                break
            
            # 镜像翻转画面，让操作更自然
            frame = cv2.flip(frame, 1)
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
            
            # 处理手势（持续移动模式）
            current_time = time.time()

            # 使用detector的get_command方法获取指令
            command = detector.get_command(gesture)

            # 持续移动模式：只要手势持续就持续移动
            if (gesture not in ["no_hand", "hand_detected", "none"]
                    and confidence > gesture_threshold
                    and command != "none"):

                # 移动类命令（持续执行）
                move_commands = ["up", "down", "left", "right", "forward", "backward"]
                
                if command in move_commands:
                    # AirSim NED 坐标系: X=前进, Y=右移, Z=下降
                    move_velocity = {
                        "forward":  (1.0, 0, 0),    # X正=前进
                        "backward": (-1.0, 0, 0),   # X负=后退
                        "right":    (0, 1.0, 0),    # Y正=右移
                        "left":     (0, -1.0, 0),   # Y负=左移
                        "down":     (0, 0, 1.0),    # Z正=下降
                        "up":       (0, 0, -1.0),   # Z负=上升
                    }
                    vx, vy, vz = move_velocity[command]
                    # 持续发送速度命令（0.1秒很短，只要手势持续就会不断发送）
                    controller.move_by_velocity(vx, vy, vz, duration=0.1)
                    
                    # 打印一次指令即可
                    if command != getattr(controller, '_last_command', None):
                        print(f"[CMD] 手势：{gesture} (置信度: {confidence:.2f}) -> 执行: {command}")
                        controller._last_command = command

                # 一次性命令（带冷却时间）
                elif command == "land":
                    if is_flying and current_time - last_command_time > command_cooldown:
                        print("[INFO] 降落...")
                        controller.land()
                        is_flying = False
                        last_command_time = current_time

                elif command == "hover":
                    if current_time - last_command_time > command_cooldown:
                        print("[INFO] 悬停")
                        controller.hover()
                        last_command_time = current_time

                elif command == "takeoff":
                    if not is_flying and current_time - last_command_time > command_cooldown:
                        print("[INFO] 起飞...")
                        controller.takeoff()
                        is_flying = True
                        last_command_time = current_time

                elif command == "stop":
                    if current_time - last_command_time > command_cooldown:
                        print("[INFO] 停止")
                        controller.hover()
                        last_command_time = current_time
            
            # 显示画面
            if show_window:
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
                print(f"[状态] 高度：{state['position'][2]:.2f}m")
    
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
        
        print("[OK] 程序安全退出")


if __name__ == "__main__":
    main()
