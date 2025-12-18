"""
辅助处理工具：新增多语言标注功能，解决单一语言问题
"""
import numpy as np
import torch
import cv2

def process_box_coords(box, scale_x, scale_y):
    """
    安全处理YOLOv8检测框坐标，解决张量类型错误
    :param box: YOLOv8的检测框对象
    :param scale_x: 宽度缩放比例
    :param scale_y: 高度缩放比例
    :return: 转换后的坐标(x1, y1, x2, y2)
    """
    # 兼容张量和numpy数组
    if isinstance(box.xyxy[0], torch.Tensor):
        box_xyxy = box.xyxy[0].cpu().numpy()
    else:
        box_xyxy = np.array(box.xyxy[0])
    # 缩放坐标
    scaled_box = box_xyxy * [scale_x, scale_y, scale_x, scale_y]
    # 转换为整数
    return map(int, scaled_box)

def draw_annotations(frame, detected_objects, is_accident, language="zh"):  # 新增：语言参数
    """
    多语言标注：支持中文(zh)/英文(en)，根据语言动态切换文本
    :param frame: 原始视频帧
    :param detected_objects: 检测到的目标列表
    :param is_accident: 是否检测到事故
    :param language: 标注语言（zh/en）
    :return: 标注后的帧
    """
    # 新增：多语言文本配置字典（可扩展更多语言）
    lang_text = {
        "zh": {
            "accident_warn": "⚠️ 检测到事故！",
            "person": "行人",
            "car": "汽车",
            "truck": "卡车"
        },
        "en": {
            "accident_warn": "⚠️ ACCIDENT DETECTED!",
            "person": "Person",
            "car": "Car",
            "truck": "Truck"
        }
    }
    # 确保语言有效，默认中文
    text = lang_text.get(language, lang_text["zh"])

    # 绘制目标检测框（使用多语言类别名）
    for (cls_name, x1, y1, x2, y2) in detected_objects:
        # 转换为当前语言的类别名称
        display_cls = text.get(cls_name, cls_name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, display_cls, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 绘制事故警告（多语言）
    if is_accident:
        cv2.putText(frame, text["accident_warn"], (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return frame

# 供外部导入的函数
__all__ = ["process_box_coords", "draw_annotations"]
