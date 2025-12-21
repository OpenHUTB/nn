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


# ========== FPS计算（还原初始版本，无多余逻辑） ==========
class CvFpsCalc:
    def __init__(self, buffer_len=10):
        self.buffer_len = buffer_len
        self.times = deque(maxlen=buffer_len)

    def get(self):
        self.times.append(time.perf_counter())
        if len(self.times) < 2:
            return 0
        return int(len(self.times) / (self.times[-1] - self.times[0]))


# ========== 手势分类器（还原初始版本） ==========
class KeyPointClassifier:
    def __call__(self, landmark_list):
        return 7  # 模拟点手势


class PointHistoryClassifier:
    def __call__(self, point_history):
        return 0


# ========== 参数解析（还原初始版本） ==========
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


# ========== 辅助函数（仅修复按键响应，不改动逻辑） ==========
def select_mode(key, mode):
    """仅修复按键捕获，保留初始逻辑"""
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    # 还原初始按键判断，仅增加ASCII码兼容（不影响帧率）
    if key == ord('n') or key == 110:
        mode = 0
    elif key == ord('k') or key == 107:
        mode = 1
    elif key == ord('h') or key == 104:
        mode = 2
    return number, mode


def calc_bounding_rect(image):
    """还原初始逻辑"""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    bw, bh = 200, 200
    return [cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2]


def calc_landmark_list(image):
    """还原初始逻辑"""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    landmark_list = []

    landmark_list.append([cx, cy])
    landmark_list.append([cx - 50, cy - 30])
    landmark_list.append([cx - 80, cy - 60])
    landmark_list.append([cx - 100, cy - 90])
    landmark_list.append([cx - 110, cy - 110])
    landmark_list.append([cx + 50, cy - 30])
    landmark_list.append([cx + 80, cy - 60])
    landmark_list.append([cx + 100, cy - 90])
    landmark_list.append([cx + 110, cy - 110])
    landmark_list.append([cx + 30, cy - 10])
    landmark_list.append([cx + 50, cy - 40])
    landmark_list.append([cx + 70, cy - 70])
    landmark_list.append([cx + 80, cy - 90])
    landmark_list.append([cx + 10, cy + 10])
    landmark_list.append([cx + 20, cy - 20])
    landmark_list.append([cx + 30, cy - 50])
    landmark_list.append([cx + 40, cy - 70])
    landmark_list.append([cx - 10, cy + 10])
    landmark_list.append([cx - 20, cy - 20])
    landmark_list.append([cx - 30, cy - 50])
    landmark_list.append([cx - 40, cy - 70])

    return landmark_list


def pre_process_landmark(landmark_list):
    """还原初始逻辑"""
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
    """还原初始逻辑"""
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
    """还原初始绘制逻辑（不改动）"""
    if len(landmark_list) == 0:
        return image
    links = [(2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
             (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
             (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
    for (p1, p2) in links:
        if p1 < len(landmark_list) and p2 < len(landmark_list):
            cv.line(image, tuple(landmark_list[p1]), tuple(landmark_list[p2]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_list[p1]), tuple(landmark_list[p2]), (255, 255, 255), 2)
    for i, (x, y) in enumerate(landmark_list):
        size = 8 if i in [4, 8, 12, 16, 20] else 5
        cv.circle(image, (x, y), size, (255, 255, 255), -1)
        cv.circle(image, (x, y), size, (0, 0, 0), 1)
    return image


def draw_bounding_rect(image, brect):
    """还原初始逻辑"""
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
    return image


def draw_info_text(image, brect, hand_sign_text, finger_gesture_text):
    """还原初始逻辑"""
    cv.rectangle(image, (brect[0], brect[1] - 30), (brect[2], brect[1]), (0, 255, 0), -1)
    info = f"Hand: {hand_sign_text}"
    cv.putText(image, info, (brect[0] + 5, brect[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if finger_gesture_text:
        cv.putText(image, f"Gesture: {finger_gesture_text}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return image


def draw_point_history(image, point_history):
    """还原初始逻辑"""
    for i, (x, y) in enumerate(point_history):
        if x != 0 and y != 0:
            cv.circle(image, (x, y), 2 + i // 2, (0, 255, 0), -1)
    return image


def draw_info(image, fps, mode, number):
    """仅修复模式显示的可视化，不改动绘制逻辑"""
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    mode_text = ["Idle", "Log Keypoint", "Log Point History"][mode] if 0 <= mode <= 2 else "Idle"
    cv.putText(image, f"Mode: {mode_text}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if 0 <= number <= 9:
        cv.putText(image, f"Num: {number}", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


# ========== 主函数（仅修复退出/按键BUG，不改动核心逻辑） ==========
def main():
    args = get_args()
    # 还原初始摄像头初始化（无多余硬件加速配置）
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # 还原初始初始化逻辑
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    keypoint_labels = ["None", "Point", "Fist", "OK", "Peace", "ThumbUp", "ThumbDown", "PointGesture"]
    point_history_labels = ["None", "MoveUp", "MoveDown", "MoveLeft", "MoveRight"]
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    mode = 0

    print("✅ 还原初始版本（30帧）| ESC退出 | n/k/h切换模式")

    try:
        while True:
            # 还原初始帧率计算
            fps = cvFpsCalc.get()

            # 修复按键响应（仅捕获，无多余逻辑）
            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC退出
                break

            # 还原初始模式切换
            number, mode = select_mode(key, mode)

            # 还原初始帧读取（无多余异常捕获）
            ret, frame = cap.read()
            if not ret:
                break

            # 还原初始镜像+拷贝逻辑
            frame = cv.flip(frame, 1)
            debug_frame = copy.deepcopy(frame)

            # 还原初始核心逻辑（不改动）
            brect = calc_bounding_rect(debug_frame)
            landmark_list = calc_landmark_list(debug_frame)
            pre_landmark = pre_process_landmark(landmark_list)
            pre_point_history = pre_process_point_history(debug_frame, point_history)

            hand_sign_id = keypoint_classifier(pre_landmark)
            point_history.append(landmark_list[8] if hand_sign_id == 7 else [0, 0])

            finger_gesture_id = 0
            if len(pre_point_history) == history_length * 2:
                finger_gesture_id = point_history_classifier(pre_point_history)
            finger_gesture_history.append(finger_gesture_id)
            most_common = Counter(finger_gesture_history).most_common(1)

            # 还原初始绘制逻辑
            debug_frame = draw_bounding_rect(debug_frame, brect)
            debug_frame = draw_landmarks(debug_frame, landmark_list)
            debug_frame = draw_info_text(
                debug_frame, brect,
                keypoint_labels[hand_sign_id] if hand_sign_id < len(keypoint_labels) else "Unknown",
                point_history_labels[most_common[0][0]] if most_common else "Unknown"
            )
            debug_frame = draw_point_history(debug_frame, point_history)
            debug_frame = draw_info(debug_frame, fps, mode, number)

            # 还原初始窗口显示
            cv.imshow('Hand Gesture Recognition', debug_frame)

    except KeyboardInterrupt:
        pass
    finally:
        # 还原初始资源释放
        cap.release()
        cv.destroyAllWindows()
        print(f"✅ 退出 | 最终帧率：{fps}")


if __name__ == '__main__':
    main()