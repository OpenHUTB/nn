#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 导入必要的库
import copy  # 深拷贝库，用于数据副本创建
import argparse  # 命令行参数解析库
import itertools  # 迭代工具库，用于数据扁平化
from collections import Counter  # 计数工具，用于手势历史统计
from collections import deque  # 双端队列，用于FPS计算和历史数据存储
import time  # 时间库，用于FPS计算

import cv2 as cv  # OpenCV库，核心视觉处理
import numpy as np  # 数值计算库，用于数组操作


# ========== FPS计算类（还原初始版本，无多余逻辑） ==========
class CvFpsCalc:
    """
    FPS（帧率）计算类
    功能：基于时间戳队列计算实时帧率，缓冲区长度控制计算稳定性
    """

    def __init__(self, buffer_len=10):
        """
        初始化FPS计算器
        :param buffer_len: 时间戳缓冲区长度，默认10帧
        """
        self.buffer_len = buffer_len  # 缓冲区长度
        self.times = deque(maxlen=buffer_len)  # 存储时间戳的双端队列

    def get(self):
        """
        计算并返回当前帧率
        :return: 整数型帧率值（FPS）
        """
        # 记录当前时间戳
        self.times.append(time.perf_counter())
        # 缓冲区数据不足时返回0
        if len(self.times) < 2:
            return 0
        # 帧率计算公式：帧数 / 总时间（秒）
        return int(len(self.times) / (self.times[-1] - self.times[0]))


# ========== 手势分类器（简化版，模拟点手势识别） ==========
class KeyPointClassifier:
    """
    关键点分类器（简化版）
    功能：模拟手势分类，固定返回点手势标识（7）
    """

    def __call__(self, landmark_list):
        """
        分类调用方法
        :param landmark_list: 手部关键点列表（未实际使用）
        :return: 固定返回7（代表点手势）
        """
        return 7  # 模拟点手势分类结果


class PointHistoryClassifier:
    """
    轨迹历史分类器（简化版）
    功能：模拟轨迹分类，固定返回0
    """

    def __call__(self, point_history):
        """
        分类调用方法
        :param point_history: 关键点轨迹历史（未实际使用）
        :return: 固定返回0
        """
        return 0  # 模拟轨迹分类结果


# ========== 命令行参数解析函数 ==========
def get_args():
    """
    解析命令行参数
    :return: 解析后的参数对象
    参数说明：
        --device: 摄像头设备号，默认0（内置摄像头）
        --width: 摄像头采集宽度，默认960像素
        --height: 摄像头采集高度，默认540像素
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加摄像头设备号参数
    parser.add_argument("--device", type=int, default=0)
    # 添加采集宽度参数
    parser.add_argument("--width", type=int, default=960)
    # 添加采集高度参数
    parser.add_argument("--height", type=int, default=540)
    # 解析参数并返回
    return parser.parse_args()


# ========== 辅助函数（仅修复按键响应，不改动核心逻辑） ==========
def select_mode(key, mode):
    """
    按键模式选择函数
    功能：根据按键值切换程序运行模式，兼容ASCII码和字符判断
    :param key: 按键ASCII码值
    :param mode: 当前模式（0=Idle/空闲, 1=Log Keypoint/关键点记录, 2=Log Point History/轨迹记录）
    :return: (数字编号, 切换后的模式)
    """
    number = -1  # 初始化数字编号（0-9按键对应）
    # 判断是否为数字按键（0-9）
    if 48 <= key <= 57:
        number = key - 48  # 转换为数字（ASCII码48对应0）

    # 模式切换逻辑（兼容字符和ASCII码）
    if key == ord('n') or key == 110:  # n键：切换到空闲模式
        mode = 0
    elif key == ord('k') or key == 107:  # k键：切换到关键点记录模式
        mode = 1
    elif key == ord('h') or key == 104:  # h键：切换到轨迹记录模式
        mode = 2
    return number, mode


def calc_bounding_rect(image):
    """
    计算手部边界框（模拟）
    功能：以画面中心为基准，生成200×200像素的正方形边界框
    :param image: 输入图像（用于获取宽高）
    :return: 边界框坐标 [x1, y1, x2, y2]
    """
    # 获取图像宽高
    h, w = image.shape[:2]
    # 计算画面中心坐标
    cx, cy = w // 2, h // 2
    # 边界框尺寸（200×200）
    bw, bh = 200, 200
    # 返回边界框坐标（左上x, 左上y, 右下x, 右下y）
    return [cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2]


def calc_landmark_list(image):
    """
    生成模拟手部关键点列表
    功能：以画面中心为基准，生成21个预设的手部关键点坐标（对应手部骨骼）
    :param image: 输入图像（用于获取宽高）
    :return: 21个关键点的坐标列表 [[x1,y1], [x2,y2], ..., [x21,y21]]
    关键点说明：
        0: 手掌中心
        1-4: 拇指
        5-8: 食指
        9-12: 中指
        13-16: 无名指
        17-20: 小指
    """
    # 获取图像宽高
    h, w = image.shape[:2]
    # 画面中心坐标
    cx, cy = w // 2, h // 2
    # 初始化关键点列表
    landmark_list = []

    # 0: 手掌中心
    landmark_list.append([cx, cy])
    # 1-4: 拇指关键点
    landmark_list.append([cx - 50, cy - 30])
    landmark_list.append([cx - 80, cy - 60])
    landmark_list.append([cx - 100, cy - 90])
    landmark_list.append([cx - 110, cy - 110])
    # 5-8: 食指关键点
    landmark_list.append([cx + 50, cy - 30])
    landmark_list.append([cx + 80, cy - 60])
    landmark_list.append([cx + 100, cy - 90])
    landmark_list.append([cx + 110, cy - 110])
    # 9-12: 中指关键点
    landmark_list.append([cx + 30, cy - 10])
    landmark_list.append([cx + 50, cy - 40])
    landmark_list.append([cx + 70, cy - 70])
    landmark_list.append([cx + 80, cy - 90])
    # 13-16: 无名指关键点
    landmark_list.append([cx + 10, cy + 10])
    landmark_list.append([cx + 20, cy - 20])
    landmark_list.append([cx + 30, cy - 50])
    landmark_list.append([cx + 40, cy - 70])
    # 17-20: 小指关键点
    landmark_list.append([cx - 10, cy + 10])
    landmark_list.append([cx - 20, cy - 20])
    landmark_list.append([cx - 30, cy - 50])
    landmark_list.append([cx - 40, cy - 70])

    return landmark_list


def pre_process_landmark(landmark_list):
    """
    关键点预处理函数
    功能：归一化关键点坐标（以手掌中心为原点，缩放至-1~1范围）
    :param landmark_list: 原始关键点列表
    :return: 归一化后的一维数组
    """
    # 深拷贝关键点列表（避免修改原数据）
    temp = copy.deepcopy(landmark_list)
    # 空列表直接返回
    if not temp:
        return []
    # 以手掌中心（第一个关键点）为原点
    base_x, base_y = temp[0][0], temp[0][1]
    # 所有关键点减去原点坐标（相对化）
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    # 将二维列表扁平化为一维数组
    temp = list(itertools.chain.from_iterable(temp))
    # 计算最大绝对值（用于缩放）
    max_val = max(map(abs, temp)) if temp else 1
    # 归一化到-1~1范围
    return [x / max_val for x in temp]


def pre_process_point_history(image, point_history):
    """
    轨迹历史预处理函数
    功能：归一化轨迹坐标（以第一个轨迹点为原点，缩放至图像宽高比例）
    :param image: 输入图像（用于获取宽高）
    :param point_history: 轨迹点历史列表
    :return: 归一化后的一维数组
    """
    # 深拷贝轨迹列表
    temp = copy.deepcopy(point_history)
    # 空列表直接返回
    if not temp:
        return []
    # 以第一个轨迹点为原点
    base_x, base_y = temp[0][0], temp[0][1]
    # 获取图像宽高
    image_w, image_h = image.shape[1], image.shape[0]
    # 归一化坐标（相对图像比例）
    for i in range(len(temp)):
        temp[i][0] = (temp[i][0] - base_x) / image_w
        temp[i][1] = (temp[i][1] - base_y) / image_h
    # 扁平化数组并返回
    return list(itertools.chain.from_iterable(temp))


def draw_landmarks(image, landmark_list):
    """
    绘制手部关键点和连线
    功能：在图像上绘制21个关键点（圆）和骨骼连线（线条）
    :param image: 输入图像（画布）
    :param landmark_list: 关键点列表
    :return: 绘制后的图像
    """
    # 空列表直接返回原图像
    if len(landmark_list) == 0:
        return image
    # 定义手部骨骼连线（关键点索引对）
    links = [(2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
             (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
             (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
    # 绘制骨骼连线
    for (p1, p2) in links:
        # 确保索引有效
        if p1 < len(landmark_list) and p2 < len(landmark_list):
            # 绘制黑色粗线（底层）
            cv.line(image, tuple(landmark_list[p1]), tuple(landmark_list[p2]), (0, 0, 0), 6)
            # 绘制白色细线（上层，模拟骨骼）
            cv.line(image, tuple(landmark_list[p1]), tuple(landmark_list[p2]), (255, 255, 255), 2)
    # 绘制关键点（圆）
    for i, (x, y) in enumerate(landmark_list):
        # 指尖关键点（4/8/12/16/20）绘制更大的圆
        size = 8 if i in [4, 8, 12, 16, 20] else 5
        # 白色实心圆（底层）
        cv.circle(image, (x, y), size, (255, 255, 255), -1)
        # 黑色描边（上层）
        cv.circle(image, (x, y), size, (0, 0, 0), 1)
    return image


def draw_bounding_rect(image, brect):
    """
    绘制手部边界框
    功能：在图像上绘制绿色矩形边界框
    :param image: 输入图像
    :param brect: 边界框坐标 [x1, y1, x2, y2]
    :return: 绘制后的图像
    """
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
    return image


def draw_info_text(image, brect, hand_sign_text, finger_gesture_text):
    """
    绘制手势信息文本
    功能：在边界框上方绘制手势类型文本，在画面左上角绘制轨迹类型文本
    :param image: 输入图像
    :param brect: 边界框坐标
    :param hand_sign_text: 手势类型文本（如Point）
    :param finger_gesture_text: 轨迹类型文本（如None）
    :return: 绘制后的图像
    """
    # 绘制手势类型背景框（绿色）
    cv.rectangle(image, (brect[0], brect[1] - 30), (brect[2], brect[1]), (0, 255, 0), -1)
    # 手势类型文本内容
    info = f"Hand: {hand_sign_text}"
    # 绘制手势类型文本（白色）
    cv.putText(image, info, (brect[0] + 5, brect[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # 绘制轨迹类型文本（红色）
    if finger_gesture_text:
        cv.putText(image, f"Gesture: {finger_gesture_text}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return image


def draw_point_history(image, point_history):
    """
    绘制轨迹历史
    功能：在图像上绘制关键点的运动轨迹（渐变大小的绿色圆）
    :param image: 输入图像
    :param point_history: 轨迹点历史列表
    :return: 绘制后的图像
    """
    for i, (x, y) in enumerate(point_history):
        # 非空轨迹点才绘制
        if x != 0 and y != 0:
            # 轨迹点大小随索引递增（模拟轨迹深度）
            cv.circle(image, (x, y), 2 + i // 2, (0, 255, 0), -1)
    return image


def draw_info(image, fps, mode, number):
    """
    绘制系统信息（FPS/模式/数字）
    功能：在画面左上角绘制帧率、运行模式、数字编号
    :param image: 输入图像
    :param fps: 当前帧率
    :param mode: 当前运行模式
    :param number: 数字编号（0-9）
    :return: 绘制后的图像
    """
    # 绘制帧率（红色）
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    # 模式文本映射
    mode_text = ["Idle", "Log Keypoint", "Log Point History"][mode] if 0 <= mode <= 2 else "Idle"
    # 绘制运行模式（白色）
    cv.putText(image, f"Mode: {mode_text}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # 绘制数字编号（白色，仅当有效时）
    if 0 <= number <= 9:
        cv.putText(image, f"Num: {number}", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


# ========== 主函数（程序入口，仅修复退出/按键BUG，不改动核心逻辑） ==========
def main():
    """
    程序主函数
    执行流程：
    1. 解析命令行参数
    2. 初始化摄像头和核心组件
    3. 主循环：采集图像→处理数据→绘制画面→响应按键
    4. 资源释放与退出
    """
    # 1. 解析命令行参数
    args = get_args()

    # 2. 初始化摄像头（原生模式，无硬件加速）
    cap = cv.VideoCapture(args.device)  # 打开摄像头
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)  # 设置采集宽度
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)  # 设置采集高度

    # 3. 初始化核心组件
    cvFpsCalc = CvFpsCalc(buffer_len=10)  # FPS计算器
    keypoint_classifier = KeyPointClassifier()  # 关键点分类器
    point_history_classifier = PointHistoryClassifier()  # 轨迹分类器

    # 手势标签映射（用于显示）
    keypoint_labels = ["None", "Point", "Fist", "OK", "Peace", "ThumbUp", "ThumbDown", "PointGesture"]
    point_history_labels = ["None", "MoveUp", "MoveDown", "MoveLeft", "MoveRight"]
    # 轨迹历史长度（控制队列大小）
    history_length = 16
    # 初始化轨迹历史队列
    point_history = deque(maxlen=history_length)
    # 初始化手势历史队列（用于统计）
    finger_gesture_history = deque(maxlen=history_length)
    # 初始运行模式（0=Idle）
    mode = 0

    # 打印启动信息
    print("✅ 还原初始版本（30帧）| ESC退出 | n/k/h切换模式")

    try:
        # 4. 主循环（持续采集和处理）
        while True:
            # 4.1 计算当前帧率
            fps = cvFpsCalc.get()

            # 4.2 按键响应（1ms等待，避免卡死）
            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC键：退出主循环
                break

            # 4.3 切换运行模式
            number, mode = select_mode(key, mode)

            # 4.4 采集摄像头图像
            ret, frame = cap.read()
            if not ret:  # 采集失败则退出循环
                break

            # 4.5 图像预处理（镜像翻转+深拷贝）
            frame = cv.flip(frame, 1)  # 水平镜像（符合人眼习惯）
            debug_frame = copy.deepcopy(frame)  # 拷贝图像用于绘制（避免修改原数据）

            # 4.6 核心数据处理
            # 计算手部边界框
            brect = calc_bounding_rect(debug_frame)
            # 生成模拟关键点列表
            landmark_list = calc_landmark_list(debug_frame)
            # 关键点归一化
            pre_landmark = pre_process_landmark(landmark_list)
            # 轨迹历史归一化
            pre_point_history = pre_process_point_history(debug_frame, point_history)

            # 手势分类（模拟）
            hand_sign_id = keypoint_classifier(pre_landmark)
            # 记录食指关键点轨迹
            point_history.append(landmark_list[8] if hand_sign_id == 7 else [0, 0])

            # 轨迹分类（模拟，仅当历史数据足够时）
            finger_gesture_id = 0
            if len(pre_point_history) == history_length * 2:
                finger_gesture_id = point_history_classifier(pre_point_history)
            # 记录手势分类历史
            finger_gesture_history.append(finger_gesture_id)
            # 统计最频繁的手势（模拟分类结果）
            most_common = Counter(finger_gesture_history).most_common(1)

            # 4.7 画面绘制
            debug_frame = draw_bounding_rect(debug_frame, brect)  # 绘制边界框
            debug_frame = draw_landmarks(debug_frame, landmark_list)  # 绘制关键点和连线
            # 绘制手势信息
            debug_frame = draw_info_text(
                debug_frame, brect,
                keypoint_labels[hand_sign_id] if hand_sign_id < len(keypoint_labels) else "Unknown",
                point_history_labels[most_common[0][0]] if most_common else "Unknown"
            )
            debug_frame = draw_point_history(debug_frame, point_history)  # 绘制轨迹
            debug_frame = draw_info(debug_frame, fps, mode, number)  # 绘制系统信息

            # 4.8 显示画面
            cv.imshow('Hand Gesture Recognition', debug_frame)

    # 捕获Ctrl+C中断（手动终止程序）
    except KeyboardInterrupt:
        pass
    # 最终资源释放（无论是否异常，都执行）
    finally:
        cap.release()  # 释放摄像头资源
        cv.destroyAllWindows()  # 关闭所有OpenCV窗口
        print(f"✅ 退出 | 最终帧率：{fps}")  # 打印退出信息


# 程序入口
if __name__ == '__main__':
    main()