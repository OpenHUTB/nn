import cv2
import numpy as np
from config import config


def draw_safe_zone(image):
    """
    绘制驾驶安全走廊范围 (辅助调试)
    """
    h, w = image.shape[:2]
    center_x = w // 2

    # 计算安全区域宽度的一半
    half_width = int((w * config.SAFE_ZONE_RATIO) / 2)

    # 左边界和右边界
    left_x = center_x - half_width
    right_x = center_x + half_width

    # 颜色 (BGR): 蓝色
    color = (255, 0, 0)
    thickness = 2

    # 画两条竖线
    cv2.line(image, (left_x, 0), (left_x, h), color, thickness)
    cv2.line(image, (right_x, 0), (right_x, h), color, thickness)

    # 在上方标注文字
    cv2.putText(image, "Driving Corridor", (left_x + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    return image


def draw_results(image, results, classes):
    """
    绘制检测结果 (边界框 + 类别标签 + 置信度)
    """
    # 颜色库 (BGR)
    COLOR_BOX = (0, 255, 0)  # 绿色框
    COLOR_TEXT = (0, 0, 0)  # 黑色文字
    COLOR_BG = (0, 255, 0)  # 绿色背景条

    for (x, y, w, h, class_id, conf) in results:
        label = str(classes[class_id])
        confidence = f"{conf:.2f}"

        # 1. 画矩形框
        cv2.rectangle(image, (x, y), (x + w, y + h), COLOR_BOX, 2)

        # 2. 准备标签文字
        text = f"{label} {confidence}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 3. 画文字背景条
        cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y), COLOR_BG, -1)

        # 4. 写字
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    return image