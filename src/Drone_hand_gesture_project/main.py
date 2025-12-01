import cv2
import numpy as np
import time
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gesture_detector import GestureDetector
from drone_controller import DroneController


def create_test_frame(message="Gesture Drone Control - VM Mode"):
    """创建测试帧"""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # 添加标题
    cv2.putText(frame, message, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 添加手势说明
    gestures = [
        "Gesture Commands:",
        "Open Palm - Takeoff",
        "Closed Fist - Land",
        "Victory - Forward",
        "Thumb Up - Backward",
        "Point Up - Up",
        "Point Down - Down",
        "OK Sign - Hover",
        "Thumb Down - Stop"
    ]

    for i, text in enumerate(gestures):
        y_pos = 90 + i * 25
        color = (0, 0, 255) if i == 0 else (0, 100, 0)
        cv2.putText(frame, text, (50, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(frame, "Press 'q' to quit", (50, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


def main():
    print("=" * 60)
    print("  手势控制无人机系统 - 虚拟机版本")
    print("=" * 60)
    print("程序已启动，正在尝试显示窗口...")
    print("如果看不到窗口，请检查虚拟机显示设置")
    print("=" * 60)

    detector = GestureDetector()
    controller = DroneController(simulation_mode=True)

    # 测试显示
    test_frame = create_test_frame("Testing Display...")
    cv2.imshow('Gesture Drone - VM', test_frame)
    cv2.waitKey(1000)  # 显示1秒

    # 尝试打开摄像头
    cap = None
    for cam_id in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            print(f"摄像头 {cam_id} 打开成功")
            break
        else:
            cap = None

    if cap is None:
        print("使用虚拟摄像头模式")

    last_command_time = time.time()
    frame_count = 0

    while True:
        frame_count += 1

        # 获取帧
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
            else:
                frame = create_test_frame("Camera Error - Virtual Mode")
        else:
            # 虚拟模式 - 创建动态测试帧
            if frame_count % 30 == 0:  # 每30帧切换消息
                messages = [
                    "Virtual Camera Mode - Make gestures",
                    "Hand Detection Active - VM",
                    "Gesture Recognition Ready"
                ]
                message = messages[(frame_count // 30) % len(messages)]
                frame = create_test_frame(message)
            else:
                frame = create_test_frame("Virtual Camera Mode - Make gestures")

        # 手势检测
        try:
            processed_frame, gesture, confidence = detector.detect_gestures(frame)
        except Exception as e:
            print(f"手势检测错误: {e}")
            processed_frame = frame
            gesture = "no_hand"

        # 处理命令
        current_time = time.time()
        if (gesture not in ["no_hand", "hand_detected"] and
                current_time - last_command_time > 2.0):

            command = detector.get_command(gesture)
            if command != "none":
                print(f"🎯 检测到手势: {gesture} -> 执行: {command}")
                controller.send_command(command)
                last_command_time = current_time

        # 显示帧
        cv2.imshow('Gesture Drone Control - VM', processed_frame)

        # 退出检测
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("切换摄像头...")
            if cap:
                cap.release()
            cap = None

    # 清理
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("程序退出")


if __name__ == "__main__":
    main()
