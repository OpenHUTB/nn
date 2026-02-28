"""
process.py：辅助函数（处理坐标、距离、标注）
"""
import cv2
import numpy as np


def process_box_coords(box, scale_x, scale_y):
    """将YOLO输出的坐标缩放回原始帧尺寸"""
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    )


def get_box_center(x1, y1, x2, y2):
    """计算目标框的中心坐标"""
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def calculate_euclidean_distance(pt1, pt2):
    """计算两个点的欧式距离"""
    return np.linalg.norm(np.array(pt1) - np.array(pt2))


def draw_annotations(frame, detected_objects, is_accident, language="zh"):
    """
    绘制标注（无需扩展库，用拼音+中文注释避免乱码）
    """
    # 类别映射：拼音+中文（OpenCV默认支持英文/拼音）
    class_map = {
        "person": "Ren（人）" if language == "zh" else "Person",
        "car": "Xiao Che（小车）" if language == "zh" else "Car",
        "truck": "Ka Che（卡车）" if language == "zh" else "Truck"
    }

    # OpenCV默认英文字体（无需扩展库）
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 绘制目标框+标签
    for obj in detected_objects:
        cls_name, x1, y1, x2, y2 = obj
        display_name = class_map.get(cls_name, cls_name)
        # 绘制绿色框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制标签（避免超出画面）
        label_y = y1 - 10 if y1 > 20 else y1 + 20
        cv2.putText(frame, display_name, (x1, label_y), font, 0.8, (0, 255, 0), 2)

    # 绘制事故提示（红色）
    if is_accident:
        accident_text = "Shi Gu!（事故！）" if language == "zh" else "Accident Detected!"
        cv2.putText(frame, accident_text, (50, 50), font, 1.2, (0, 0, 255), 3)

    return frame
