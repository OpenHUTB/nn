"""步骤6：车道偏离预警系统。

基于曲率半径和车辆偏移量，判断当前驾驶风险等级，返回对应的
预警颜色、车道区域填充色和人类可读的预警消息。
"""

from config import CONFIG


def compute_warning_level(metrics):
    """根据曲率与偏移量判定当前驾驶风险等级。

    三级预警：
      - safe（安全）：偏移小、曲率半径大、车道线检测正常
      - caution（注意）：偏移中等、进入弯道
      - danger（危险）：偏移过大、急弯、车道线丢失

    Args:
        metrics: compute_lane_metrics 返回的字典

    Returns:
        dict: {
            "level": "safe" | "caution" | "danger",
            "label": 中文预警标签,
            "lane_color": 车道区域填充 BGR 颜色,
            "text_color": 叠加文字 BGR 颜色,
            "flashing": 是否需要闪烁（仅 danger 级别），
            "reasons": [触发预警的具体原因列表],
        }
    """
    level = "safe"
    reasons = []

    offset = metrics.get("offset")
    avg_r = metrics.get("avg_curvature")
    curve_dir = metrics.get("curve_direction", "直行")
    left_fit = metrics.get("left_fit")
    right_fit = metrics.get("right_fit")

    # 车道线丢失检查
    if left_fit is None and right_fit is None:
        return {
            "level": "danger",
            "label": "车道线丢失",
            "lane_color": (0, 0, 255),
            "text_color": (0, 0, 255),
            "flashing": True,
            "reasons": ["双侧车道线均未检测到"],
        }

    if left_fit is None or right_fit is None:
        level = "danger"
        reasons.append("单侧车道线丢失")

    # 偏移过大检查
    if offset is not None:
        if abs(offset) > CONFIG["warning_offset_danger"]:
            level = "danger"
            reasons.append(f"偏移过大 ({abs(offset):.2f}m)")
        elif abs(offset) > CONFIG["warning_offset_caution"]:
            if level != "danger":
                level = "caution"
            reasons.append(f"偏移偏大 ({abs(offset):.2f}m)")

    # 急弯检查
    if avg_r is not None:
        if avg_r < CONFIG["warning_curve_danger"]:
            level = "danger"
            reasons.append(f"急弯 (曲率 {avg_r:.0f}m)")
        elif avg_r < CONFIG["warning_curve_caution"]:
            if level != "danger":
                level = "caution"
            reasons.append(f"弯道 (曲率 {avg_r:.0f}m)")

    # 按等级组装返回
    if level == "danger":
        return {
            "level": "danger",
            "label": "危险",
            "lane_color": (0, 0, 255),       # 红色
            "text_color": (0, 0, 255),
            "flashing": True,
            "reasons": reasons,
        }
    elif level == "caution":
        return {
            "level": "caution",
            "label": "注意",
            "lane_color": (0, 255, 255),     # 黄色
            "text_color": (0, 255, 255),
            "flashing": False,
            "reasons": reasons,
        }
    else:
        return {
            "level": "safe",
            "label": "安全",
            "lane_color": (0, 255, 0),       # 绿色
            "text_color": (0, 255, 0),
            "flashing": False,
            "reasons": [],
        }


def get_warning_lane_color(metrics):
    """获取基于预警级别的车道区域填充色。

    便捷函数，直接返回 BGR 颜色元组。
    """
    warning = compute_warning_level(metrics)
    return warning["lane_color"]


def get_warning_text_color(metrics):
    """获取基于预警级别的叠加文字颜色。

    便捷函数，直接返回 BGR 颜色元组。
    """
    warning = compute_warning_level(metrics)
    return warning["text_color"]