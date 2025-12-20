#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import time

import cv2 as cv
import numpy as np


# ========== FPS计算 ==========
class CvFpsCalc:
    def __init__(self, buffer_len=10):
        self.buffer_len = buffer_len
        self.times = deque(maxlen=buffer_len)

    def get(self):
        self.times.append(time.perf_counter())
        if len(self.times) < 2:
            return 0
        return int(len(self.times) / (self.times[-1] - self.times[0]))


# ========== 手势分类器（简化版） ==========
class KeyPointClassifier:
    def __call__(self, landmark_list):
        return 7  # 模拟点手势


class PointHistoryClassifier:
    def __call__(self, point_history):
        return 0


# ========== 参数解析 ==========
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


# ========== 辅助函数 ==========
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    if key == ord('n'):
        mode = 0
    if key == ord('k'):
        mode = 1
    if key == ord('h'):
        mode = 2
    return number, mode


def calc_bounding_rect(image):
    """模拟手部边界框（屏幕中心）"""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    bw, bh = 200, 200  # 边界框大小
    return [cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2]


def calc_landmark_list(image):
    """模拟21个手部关键点（适配原代码逻辑）"""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    landmark_list = []

    # 手掌中心（0号点）
    landmark_list.append([cx, cy])

    # 拇指（1-4号点）
    landmark_list.append([cx - 50, cy - 30])
    landmark_list.append([cx - 80, cy - 60])
    landmark_list.append([cx - 100, cy - 90])
    landmark_list.append([cx - 110, cy - 110])

    # 食指（5-8号点）
    landmark_list.append([cx + 50, cy - 30])
    landmark_list.append([cx + 80, cy - 60])
    landmark_list.append([cx + 100, cy - 90])
    landmark_list.append([cx + 110, cy - 110])  # 8号点（点手势关键）

    # 中指（9-12号点）
    landmark_list.append([cx + 30, cy - 10])
    landmark_list.append([cx + 50, cy - 40])
    landmark_list.append([cx + 70, cy - 70])
    landmark_list.append([cx + 80, cy - 90])

    # 无名指（13-16号点）
    landmark_list.append([cx + 10, cy + 10])
    landmark_list.append([cx + 20, cy - 20])
    landmark_list.append([cx + 30, cy - 50])
    landmark_list.append([cx + 40, cy - 70])

    # 小指（17-20号点）
    landmark_list.append([cx - 10, cy + 10])
    landmark_list.append([cx - 20, cy - 20])
    landmark_list.append([cx - 30, cy - 50])
    landmark_list.append([cx - 40, cy - 70])

    return landmark_list


def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    if not temp:
        return []
    base_x, base_y = temp[0][0], temp[0][1]
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    temp = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, temp)) if temp else 1
    return [x / max_val for x in temp]


def pre_process_point_history(image, point_history):
    temp = copy.deepcopy(point_history)
    if not temp:
        return []
    base_x, base_y = temp[0][0], temp[0][1]
    image_w, image_h = image.shape[1], image.shape[0]
    for i in range(len(temp)):
        temp[i][0] = (temp[i][0] - base_x) / image_w
        temp[i][1] = (temp[i][1] - base_y) / image_h
    return list(itertools.chain.from_iterable(temp))


def draw_landmarks(image, landmark_list):
    """绘制模拟关键点"""
    if len(landmark_list) == 0:
        return image
    # 绘制手指连线
    links = [(2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
             (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
             (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
    for (p1, p2) in links:
        if p1 < len(landmark_list) and p2 < len(landmark_list):
            cv.line(image, tuple(landmark_list[p1]), tuple(landmark_list[p2]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_list[p1]), tuple(landmark_list[p2]), (255, 255, 255), 2)
    # 绘制关键点
    for i, (x, y) in enumerate(landmark_list):
        size = 8 if i in [4, 8, 12, 16, 20] else 5
        cv.circle(image, (x, y), size, (255, 255, 255), -1)
        cv.circle(image, (x, y), size, (0, 0, 0), 1)
    return image


def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
    return image


def draw_info_text(image, brect, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1] - 30), (brect[2], brect[1]), (0, 255, 0), -1)
    info = f"Hand: {hand_sign_text}"
    cv.putText(image, info, (brect[0] + 5, brect[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if finger_gesture_text:
        cv.putText(image, f"Gesture: {finger_gesture_text}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return image


def draw_point_history(image, point_history):
    for i, (x, y) in enumerate(point_history):
        if x != 0 and y != 0:
            cv.circle(image, (x, y), 2 + i // 2, (0, 255, 0), -1)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    mode_text = ["Idle", "Log Keypoint", "Log Point History"][mode] if 0 <= mode <= 2 else "Idle"
    cv.putText(image, f"Mode: {mode_text}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if 0 <= number <= 9:
        cv.putText(image, f"Num: {number}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


# ========== 主函数 ==========
def main():
    args = get_args()
    # 初始化摄像头
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # 初始化工具类
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # 标签和历史数据
    keypoint_labels = ["None", "Point", "Fist", "OK", "Peace", "ThumbUp", "ThumbDown", "PointGesture"]
    point_history_labels = ["None", "MoveUp", "MoveDown", "MoveLeft", "MoveRight"]
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    mode = 0

    while True:
        # FPS计算
        fps = cvFpsCalc.get()

        # 按键处理
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC退出
            break
        number, mode = select_mode(key, mode)

        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)  # 镜像显示
        debug_frame = copy.deepcopy(frame)

        # 【核心修改】跳过真实检测，直接模拟手部数据
        brect = calc_bounding_rect(debug_frame)
        landmark_list = calc_landmark_list(debug_frame)

        # 预处理
        pre_landmark = pre_process_landmark(landmark_list)
        pre_point_history = pre_process_point_history(debug_frame, point_history)

        # 手势分类
        hand_sign_id = keypoint_classifier(pre_landmark)
        point_history.append(landmark_list[8] if hand_sign_id == 7 else [0, 0])

        # 手指手势分类
        finger_gesture_id = 0
        if len(pre_point_history) == history_length * 2:
            finger_gesture_id = point_history_classifier(pre_point_history)
        finger_gesture_history.append(finger_gesture_id)
        most_common = Counter(finger_gesture_history).most_common(1)

        # 绘制UI
        debug_frame = draw_bounding_rect(debug_frame, brect)
        debug_frame = draw_landmarks(debug_frame, landmark_list)
        debug_frame = draw_info_text(
            debug_frame, brect,
            keypoint_labels[hand_sign_id] if hand_sign_id < len(keypoint_labels) else "Unknown",
            point_history_labels[most_common[0][0]] if most_common else "Unknown"
        )

        # 绘制辅助信息
        debug_frame = draw_point_history(debug_frame, point_history)
        debug_frame = draw_info(debug_frame, fps, mode, number)

        # 显示窗口
        cv.imshow('Hand Gesture Recognition (ESC to exit)', debug_frame)

    # 释放资源
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()