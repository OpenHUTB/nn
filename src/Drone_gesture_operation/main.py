#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import time


# 极简手势识别（仅保留拳头/点手势，极致流畅）
def main():
    # 1. 摄像头初始化（极简参数）
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)  # 极低分辨率，秒杀卡顿
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))  # 快速编码
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # 关闭缓存，降低延迟

    # 2. 固定参数（适配所有摄像头）
    skin_lower = np.array([0, 10, 10], np.uint8)
    skin_upper = np.array([30, 255, 180], np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    last_gesture = "None"
    gesture_count = 0

    print("✅ 极致轻量化手势识别 | ESC退出")
    print("💡 把手放在画面中间，握拳=Fist，伸食指=Point")

    while True:
        # 计时（极简FPS）
        t1 = time.time()

        # 3. 读取帧（跳过缓存帧）
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        frame_small = cv.resize(frame, (160, 120))  # 超小尺寸处理

        # 4. 极简手部检测
        hsv = cv.cvtColor(frame_small, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, skin_lower, skin_upper)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # 5. 找轮廓（只找最大的）
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        current_gesture = "None"
        if contours:
            cnt = max(contours, key=cv.contourArea)
            if cv.contourArea(cnt) > 1000:
                # 6. 极简分类（仅拳头/点手势）
                hull = cv.convexHull(cnt)
                solidity = cv.contourArea(cnt) / cv.contourArea(hull)
                current_gesture = "Fist" if solidity > 0.85 else "Point"

        # 7. 稳定输出（连续2帧相同）
        if current_gesture == last_gesture:
            gesture_count += 1
        else:
            gesture_count = 0
            last_gesture = current_gesture
        stable_gesture = last_gesture if gesture_count > 1 else "None"

        # 8. 绘制（极简UI，减少计算）
        cv.putText(frame, f"Gesture: {stable_gesture}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv.putText(frame, f"FPS: {int(1 / (time.time() - t1))}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 9. 显示（拉伸回原尺寸，保持清晰）
        frame_show = cv.resize(frame, (640, 480))
        cv.imshow("Ultra Light Gesture", frame_show)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()