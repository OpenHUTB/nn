import cv2
from config import config


def _clamp_ratio(value):
    return max(0.0, min(1.0, float(value)))


def _normalize_box(result):
    if result is None or len(result) < 4:
        return None

    try:
        x, y, w, h = (float(value) for value in result[:4])
    except (TypeError, ValueError):
        return None

    if w <= 0 or h <= 0:
        return None

    class_id = result[4] if len(result) > 4 else None
    confidence = result[5] if len(result) > 5 else None
    return x, y, w, h, class_id, confidence


def _resolve_label(classes, class_id):
    if class_id is None:
        return "unknown"
    try:
        class_index = int(class_id)
    except (TypeError, ValueError):
        return "unknown"
    if 0 <= class_index < len(classes):
        return str(classes[class_index])
    return f"class_{class_index}"


def draw_safe_zone(image):
    """
    绘制驾驶安全走廊范围 (辅助调试)
    """
    if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
        return image

    h, w = image.shape[:2]
    center_x = w // 2

    # 计算安全区域宽度的一半
    half_width = int((w * _clamp_ratio(config.SAFE_ZONE_RATIO)) / 2)

    # 左边界和右边界
    left_x = max(0, center_x - half_width)
    right_x = min(w - 1, center_x + half_width)

    # 颜色 (BGR): 蓝色
    color = (255, 0, 0)
    thickness = 2

    # 画两条竖线
    cv2.line(image, (left_x, 0), (left_x, h), color, thickness)
    cv2.line(image, (right_x, 0), (right_x, h), color, thickness)

    # 在上方标注文字
    label_x = min(max(0, left_x + 5), max(0, w - 150))
    cv2.putText(image, "Driving Corridor", (label_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    return image


def draw_results(image, results, classes):
    """
    绘制检测结果 (边界框 + 类别标签 + 置信度)
    """
    if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
        return image

    image_h, image_w = image.shape[:2]
    classes = [] if classes is None else classes
    results = [] if results is None else results

    # 颜色库 (BGR)
    COLOR_BOX = (0, 255, 0)  # 绿色框
    COLOR_TEXT = (0, 0, 0)  # 黑色文字
    COLOR_BG = (0, 255, 0)  # 绿色背景条

    for result in results:
        normalized_box = _normalize_box(result)
        if normalized_box is None:
            continue

        x, y, w, h, class_id, conf = normalized_box
        label = _resolve_label(classes, class_id)
        try:
            confidence_value = 0.0 if conf is None else float(conf)
        except (TypeError, ValueError):
            confidence_value = 0.0
        confidence = f"{confidence_value:.2f}"

        x1 = max(0, min(image_w - 1, int(round(x))))
        y1 = max(0, min(image_h - 1, int(round(y))))
        x2 = max(0, min(image_w - 1, int(round(x + w))))
        y2 = max(0, min(image_h - 1, int(round(y + h))))
        if x2 <= x1 or y2 <= y1:
            continue

        # 1. 画矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_BOX, 2)

        # 2. 准备标签文字
        text = f"{label} {confidence}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 3. 画文字背景条
        text_bg_left = x1
        text_bg_top = max(0, y1 - text_h - baseline - 8)
        text_bg_right = min(image_w - 1, x1 + text_w)
        text_bg_bottom = y1
        cv2.rectangle(image, (text_bg_left, text_bg_top), (text_bg_right, text_bg_bottom), COLOR_BG, -1)

        # 4. 写字
        text_y = max(text_h + 2, y1 - 5)
        cv2.putText(image, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    return image
