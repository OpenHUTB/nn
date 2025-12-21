"""
辅助处理工具：负责坐标转换、帧处理等通用逻辑
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

def draw_annotations(frame, detected_objects, is_accident):
    """
    在帧上绘制检测标注和事故警告
    :param frame: 原始视频帧
    :param detected_objects: 检测到的目标列表
    :param is_accident: 是否检测到事故
    :return: 标注后的帧
    """
    # 绘制目标检测框
    for (cls_name, x1, y1, x2, y2) in detected_objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, cls_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # 绘制事故警告
    if is_accident:
        cv2.putText(frame, "⚠️ 检测到事故！", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return frame

# 供外部导入的函数
__all__ = ["process_box_coords", "draw_annotations"]