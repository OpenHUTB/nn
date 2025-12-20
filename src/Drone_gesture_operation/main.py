#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 功能：无人机手势识别模拟程序（适配Python3.13+Windows）
# 说明：移除MediaPipe/TensorFlow依赖，通过模拟手部数据实现核心逻辑展示
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import time

import cv2 as cv
import numpy as np


# ===================== 工具类：FPS计算 =====================
class CvFpsCalc:
    """帧率计算类，基于时间队列滑动平均计算FPS"""

    def __init__(self, buffer_len=10):
        self.buffer_len = buffer_len  # 缓存长度（计算最近N帧的平均FPS）
        self.times = deque(maxlen=buffer_len)  # 时间戳队列

    def get(self):
        """获取当前FPS值"""
        self.times.append(time.perf_counter())  # 记录当前时间戳
        if len(self.times) < 2:  # 至少需要2个时间戳才能计算
            return 0
        # 计算平均FPS：帧数 / 总时间
        fps = len(self.times) / (self.times[-1] - self.times[0])
        return int(fps)


# ===================== 模拟分类器（适配原逻辑） =====================
class KeyPointClassifier:
    """关键点分类器（简化版）"""

    def __call__(self, landmark_list):
        # 固定返回7（对应PointGesture点手势，模拟分类结果）
        return 7


class PointHistoryClassifier:
    """轨迹点分类器（简化版）"""

    def __call__(self, point_history):
        # 固定返回0（对应None，模拟分类结果）
        return 0

    # ===================== 参数解析 =====================


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="手部手势识别模拟程序")
    parser.add_argument("--device", type=int, default=0, help="摄像头设备号（默认0）")
    parser.add_argument("--width", type=int, default=960, help="摄像头画面宽度")
    parser.add_argument("--height", type=int, default=540, help="摄像头画面高度")
    return parser.parse_args()


# ===================== 核心辅助函数 =====================
def select_mode(key, mode):
    """根据按键切换操作模式
    Args:
        key: 按键值
        mode: 当前模式（0:空闲 1:记录关键点 2:记录轨迹点）
    Returns:
        number: 按键数字（0-9），-1表示非数字键
        mode: 更新后的模式
    """
    number = -1
    if 48 <= key <= 57:  # 数字键0-9
        number = key - 48
    if key == ord('n'):  # n键：切换到空闲模式
        mode = 0
    if key == ord('k'):  # k键：切换到记录关键点模式
        mode = 1
    if key == ord('h'):  # h键：切换到记录轨迹点模式
        mode = 2
    return number, mode


def calc_bounding_rect(image):
    """生成模拟手部边界框（屏幕中心固定位置）
    Args:
        image: 摄像头帧画面
    Returns:
        brect: 边界框坐标 [x1, y1, x2, y2]
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2  # 屏幕中心
    bw, bh = 200, 200  # 边界框尺寸
    return [cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2]


def calc_landmark_list(image):
    """生成模拟21个手部关键点（适配原代码21点逻辑）
    Args:
        image: 摄像头帧画面
    Returns:
        landmark_list: 21个关键点坐标列表 [[x1,y1], [x2,y2], ...]
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2  # 屏幕中心（关键点基准位置）
    landmark_list = []

    # 0号点：手掌中心
    landmark_list.append([cx, cy])
    # 1-4号点：拇指
    landmark_list.extend([[cx - 50, cy - 30], [cx - 80, cy - 60], [cx - 100, cy - 90], [cx - 110, cy - 110]])
    # 5-8号点：食指（8号点为指尖，点手势关键）
    landmark_list.extend([[cx + 50, cy - 30], [cx + 80, cy - 60], [cx + 100, cy - 90], [cx + 110, cy - 110]])
    # 9-12号点：中指
    landmark_list.extend([[cx + 30, cy - 10], [cx + 50, cy - 40], [cx + 70, cy - 70], [cx + 80, cy - 90]])
    # 13-16号点：无名指
    landmark_list.extend([[cx + 10, cy + 10], [cx + 20, cy - 20], [cx + 30, cy - 50], [cx + 40, cy - 70]])
    # 17-20号点：小指
    landmark_list.extend([[cx - 10, cy + 10], [cx - 20, cy - 20], [cx - 30, cy - 50], [cx - 40, cy - 70]])

    return landmark_list


def pre_process_landmark(landmark_list):
    """关键点预处理：相对坐标转换+归一化（适配原逻辑）
    Args:
        landmark_list: 原始关键点列表
    Returns:
        预处理后的一维归一化列表
    """
    temp = copy.deepcopy(landmark_list)
    if not temp:  # 空值保护
        return []

    # 相对坐标：以0号点（手掌中心）为基准
    base_x, base_y = temp[0][0], temp[0][1]
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y

    # 一维化 + 归一化（消除尺度影响）
    temp = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, temp)) if temp else 1  # 除零保护
    return [x / max_val for x in temp]


def pre_process_point_history(image, point_history):
    """轨迹点预处理：相对坐标转换+归一化（适配原逻辑）
    Args:
        image: 摄像头帧画面
        point_history: 轨迹点历史列表
    Returns:
        预处理后的一维归一化列表
    """
    temp = copy.deepcopy(point_history)
    if not temp:  # 空值保护
        return []

    # 相对坐标：以第一个点为基准
    base_x, base_y = temp[0][0], temp[0][1]
    image_w, image_h = image.shape[1], image.shape[0]
    for i in range(len(temp)):
        temp[i][0] = (temp[i][0] - base_x) / image_w  # 归一化到[0,1]
        temp[i][1] = (temp[i][1] - base_y) / image_h

    # 一维化
    return list(itertools.chain.from_iterable(temp))


# ===================== 绘制函数（UI展示） =====================
def draw_landmarks(image, landmark_list):
    """绘制手部关键点和连线
    Args:
        image: 待绘制的画面
        landmark_list: 关键点列表
    Returns:
        绘制后的画面
    """
    if len(landmark_list) == 0:
        return image

    # 手指连线定义（关键点索引对）
    links = [(2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
             (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
             (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]

    # 绘制连线：黑色粗线+白色细线（立体效果）
    for (p1, p2) in links:
        if p1 < len(landmark_list) and p2 < len(landmark_list):  # 索引保护
            cv.line(image, tuple(landmark_list[p1]), tuple(landmark_list[p2]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_list[p1]), tuple(landmark_list[p2]), (255, 255, 255), 2)

    # 绘制关键点：指尖8号/12号等用大圆点，其余用小圆点
    for i, (x, y) in enumerate(landmark_list):
        size = 8 if i in [4, 8, 12, 16, 20] else 5
        cv.circle(image, (x, y), size, (255, 255, 255), -1)  # 白色填充
        cv.circle(image, (x, y), size, (0, 0, 0), 1)  # 黑色边框
    return image


def draw_bounding_rect(image, brect):
    """绘制手部边界框
    Args:
        image: 待绘制的画面
        brect: 边界框坐标 [x1,y1,x2,y2]
    Returns:
        绘制后的画面
    """
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
    return image


def draw_info_text(image, brect, hand_sign_text, finger_gesture_text):
    """绘制手势信息文本
    Args:
        image: 待绘制的画面
        brect: 边界框坐标
        hand_sign_text: 手部手势文本
        finger_gesture_text: 手指轨迹手势文本
    Returns:
        绘制后的画面
    """
    # 绘制背景框（覆盖边界框上方）
    cv.rectangle(image, (brect[0], brect[1] - 30), (brect[2], brect[1]), (0, 255, 0), -1)
    # 绘制手部手势标签
    info = f"Hand: {hand_sign_text}"
    cv.putText(image, info, (brect[0] + 5, brect[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # 绘制轨迹手势标签
    if finger_gesture_text:
        cv.putText(image, f"Gesture: {finger_gesture_text}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return image


def draw_point_history(image, point_history):
    """绘制轨迹点历史（指尖移动轨迹）
    Args:
        image: 待绘制的画面
        point_history: 轨迹点列表
    Returns:
        绘制后的画面
    """
    for i, (x, y) in enumerate(point_history):
        if x != 0 and y != 0:  # 跳过无效点
            # 轨迹点大小随索引递增（视觉层次感）
            cv.circle(image, (x, y), 2 + i // 2, (0, 255, 0), -1)
    return image


def draw_info(image, fps, mode, number):
    """绘制全局信息（FPS、模式、数字）
    Args:
        image: 待绘制的画面
        fps: 当前帧率
        mode: 当前模式
        number: 当前数字
    Returns:
        绘制后的画面
    """
    # 绘制FPS
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    # 绘制模式
    mode_text = ["Idle", "Log Keypoint", "Log Point History"][mode] if 0 <= mode <= 2 else "Idle"
    cv.putText(image, f"Mode: {mode_text}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # 绘制数字
    if 0 <= number <= 9:
        cv.putText(image, f"Num: {number}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


# ===================== 主程序入口 =====================
def main():
    # 1. 初始化参数和资源
    args = get_args()
    cap = cv.VideoCapture(args.device)  # 打开摄像头
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)  # 设置宽度
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)  # 设置高度

    # 初始化工具类
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # 初始化标签和历史数据
    keypoint_labels = ["None", "Point", "Fist", "OK", "Peace", "ThumbUp", "ThumbDown", "PointGesture"]
    point_history_labels = ["None", "MoveUp", "MoveDown", "MoveLeft", "MoveRight"]
    history_length = 16  # 轨迹点缓存长度
    point_history = deque(maxlen=history_length)  # 指尖轨迹缓存
    finger_gesture_history = deque(maxlen=history_length)  # 手势分类结果缓存
    mode = 0  # 初始模式：空闲

    # 2. 主循环（摄像头帧处理）
    while True:
        # 计算当前FPS
        fps = cvFpsCalc.get()

        # 按键处理（ESC退出）
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break
        number, mode = select_mode(key, mode)

        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:  # 帧读取失败则退出
            break
        frame = cv.flip(frame, 1)  # 镜像翻转（符合视觉习惯）
        debug_frame = copy.deepcopy(frame)  # 用于绘制的帧副本

        # 3. 核心逻辑：模拟手部数据生成 + 预处理 + 分类
        brect = calc_bounding_rect(debug_frame)  # 生成边界框
        landmark_list = calc_landmark_list(debug_frame)  # 生成关键点
        pre_landmark = pre_process_landmark(landmark_list)  # 关键点预处理
        pre_point_history = pre_process_point_history(debug_frame, point_history)  # 轨迹点预处理

        # 手势分类
        hand_sign_id = keypoint_classifier(pre_landmark)  # 关键点分类
        # 记录指尖轨迹（点手势时记录8号点，否则记录无效点）
        point_history.append(landmark_list[8] if hand_sign_id == 7 else [0, 0])

        # 轨迹手势分类（缓存满16*2个点时分类）
        finger_gesture_id = 0
        if len(pre_point_history) == history_length * 2:
            finger_gesture_id = point_history_classifier(pre_point_history)
        finger_gesture_history.append(finger_gesture_id)
        # 取最频繁的手势分类结果（防抖）
        most_common = Counter(finger_gesture_history).most_common(1)

        # 4. UI绘制
        debug_frame = draw_bounding_rect(debug_frame, brect)  # 绘制边界框
        debug_frame = draw_landmarks(debug_frame, landmark_list)  # 绘制关键点
        debug_frame = draw_info_text(  # 绘制手势信息
            debug_frame, brect,
            keypoint_labels[hand_sign_id] if hand_sign_id < len(keypoint_labels) else "Unknown",
            point_history_labels[most_common[0][0]] if most_common else "Unknown"
        )
        debug_frame = draw_point_history(debug_frame, point_history)  # 绘制轨迹
        debug_frame = draw_info(debug_frame, fps, mode, number)  # 绘制全局信息

        # 显示画面
        cv.imshow('Hand Gesture Recognition (ESC to exit)', debug_frame)

    # 3. 资源释放
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()