import cv2
import numpy as np


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

        # 3. 画文字背景条 (为了字看得更清楚)
        cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y), COLOR_BG, -1)

        # 4. 写字
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    return image