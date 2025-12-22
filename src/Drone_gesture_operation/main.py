#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import time
from collections import deque


# ========== 稳定版手势识别（固定ROI+帧验证） ==========
class StableHandRecognizer:
    def __init__(self):
        # 1. 固定检测区域（画面右侧1/3）
        self.roi_x1, self.roi_y1 = 600, 100
        self.roi_x2, self.roi_y2 = 900, 500
        # 2. 肤色范围（扩大兼容）
        self.lower_skin = np.array([0, 20, 30], dtype=np.uint8)
        self.upper_skin = np.array([30, 255, 255], dtype=np.uint8)
        # 3. 手势缓存（连续3帧相同才确认）
        self.gesture_buffer = deque(maxlen=3)
        self.last_gesture = "None"

    def get_roi(self, image):
        """截取固定检测区域"""
        return image[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]

    def process(self, image):
        # 1. 截取ROI
        roi = self.get_roi(image)
        if roi.size == 0:
            return "None", []

        # 2. 预处理（降噪+肤色检测）
        blur = cv.GaussianBlur(roi, (7, 7), 0)
        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_skin, self.upper_skin)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # 3. 找手部轮廓
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.gesture_buffer.append("None")
            return self._get_stable_gesture(), []

        max_contour = max(contours, key=cv.contourArea)
        if cv.contourArea(max_contour) < 5000:
            self.gesture_buffer.append("None")
            return self._get_stable_gesture(), []

        # 4. 手势判断（仅保留拳头/点手势，最稳定）
        hull = cv.convexHull(max_contour)
        solidity = cv.contourArea(max_contour) / cv.contourArea(hull)
        current_gesture = "Fist" if solidity > 0.85 else "Point"
        self.gesture_buffer.append(current_gesture)

        # 5. 稳定输出（连续3帧相同）
        return self._get_stable_gesture(), max_contour

    def _get_stable_gesture(self):
        """只有连续3帧相同才输出"""
        if len(self.gesture_buffer) < 3:
            return self.last_gesture
        if len(set(self.gesture_buffer)) == 1:
            self.last_gesture = self.gesture_buffer[0]
        return self.last_gesture


# ========== 主函数（带ROI框+稳定显示） ==========
def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv.CAP_PROP_EXPOSURE, -6)  # 固定曝光，避免过曝
    recognizer = StableHandRecognizer()
    fps_calc = deque(maxlen=10)

    while True:
        # 计时算FPS
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        debug_frame = frame.copy()

        # 1. 绘制ROI框（提示用户把手放在这里）
        cv.rectangle(debug_frame, (recognizer.roi_x1, recognizer.roi_y1),
                     (recognizer.roi_x2, recognizer.roi_y2), (0, 255, 255), 2)
        cv.putText(debug_frame, "Put hand here", (recognizer.roi_x1 + 10, recognizer.roi_y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 2. 识别手势
        gesture, contour = recognizer.process(frame)

        # 3. 绘制结果（固定位置，不闪烁）
        cv.putText(debug_frame, f"Stable Gesture: {gesture}", (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        # 4. 绘制手部轮廓（ROI内）
        if contour is not None and len(contour) > 0:
            # 转换轮廓坐标到全局画面
            contour[:, :, 0] += recognizer.roi_x1
            contour[:, :, 1] += recognizer.roi_y1
            cv.drawContours(debug_frame, [contour], -1, (0, 255, 0), 2)

        # 计算FPS
        fps = 1 / (time.time() - start)
        fps_calc.append(fps)
        cv.putText(debug_frame, f"FPS: {int(np.mean(fps_calc))}", (50, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        # 显示
        cv.imshow("Stable Hand Gesture", debug_frame)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()