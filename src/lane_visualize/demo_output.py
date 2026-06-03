"""
演示脚本：生成车道线检测 + 风险评估系统运行效果图
无需实际视频输入，模拟生成可视化界面
"""

import cv2
import numpy as np
import math
import time
from pathlib import Path


def create_demo_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """创建模拟的道路场景帧"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # 天空渐变
    for y in range(height // 2):
        ratio = y / (height // 2)
        frame[y, :] = [int(180 + 50 * ratio), int(200 + 30 * ratio), int(220 + 20 * ratio)]

    # 地面
    frame[height // 2:, :] = [80, 90, 80]

    # 道路
    road_top = int(height * 0.45)
    pts = np.array([
        [0, height],
        [width // 2 - 50, road_top],
        [width // 2 + 50, road_top],
        [width, height]
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (60, 60, 65))

    # 车道线
    lane_color = (200, 200, 200)
    # 左车道线
    cv2.line(frame, (width // 2 - 200, height), (width // 2 - 30, road_top + 50), lane_color, 3)
    # 右车道线
    cv2.line(frame, (width // 2 + 200, height), (width // 2 + 30, road_top + 50), lane_color, 3)
    # 中心虚线
    for i in range(10):
        y1 = height - i * 60
        y2 = y1 - 30
        if y2 > road_top:
            cv2.line(frame, (width // 2, y1), (width // 2, y2), (255, 255, 0), 2)

    return frame


def draw_lane_overlay(frame: np.ndarray, deviation: float) -> np.ndarray:
    """绘制车道区域叠加层"""
    overlay = np.zeros_like(frame)
    h, w = frame.shape[:2]
    road_top = int(h * 0.45)

    # 绿色车道区域
    pts = np.array([
        [w // 2 - 180, h],
        [w // 2 - 25, road_top + 60],
        [w // 2 + 25, road_top + 60],
        [w // 2 + 180, h]
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], (0, 200, 0))

    return cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)


def draw_detection_box(frame: np.ndarray, box: tuple, label: str, risk_level: str, risk_score: float, ttc: float):
    """绘制检测框和风险信息"""
    x1, y1, x2, y2 = box

    if risk_level == "DANGER":
        color = (0, 0, 255)
        thickness = 3
    elif risk_level == "WARNING":
        color = (0, 255, 255)
        thickness = 2
    else:
        color = (0, 255, 0)
        thickness = 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # 标签
    text = f"{label} {risk_level} {risk_score:.0f}"
    cv2.putText(frame, text, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # TTC
    ttc_text = f"TTC:{ttc:.1f}s" if np.isfinite(ttc) else "TTC:--"
    cv2.putText(frame, ttc_text, (x1, min(frame.shape[0] - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)

    # 风险警告
    if risk_level == "DANGER":
        cv2.putText(frame, "BRAKE!", (x1, max(45, y1 - 35)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    elif risk_level == "WARNING":
        cv2.putText(frame, "WARNING", (x1, max(45, y1 - 35)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def draw_dashboard(frame: np.ndarray, deviation: float, steer_angle: float,
                   fps: float, status: str, max_risk: float, min_ttc: float):
    """绘制仪表盘"""
    h, w = frame.shape[:2]

    # 底部黑色背景
    cv2.rectangle(frame, (0, h - 90), (w, h), (0, 0, 0), -1)

    # 虚拟方向盘
    center = (w // 2, h - 45)
    radius = 30
    display_angle = steer_angle * 4 * 1.5
    display_angle = max(-90, min(90, display_angle))
    rad = math.radians(display_angle - 90)
    end_x = int(center[0] + radius * math.cos(rad))
    end_y = int(center[1] + radius * math.sin(rad))

    cv2.circle(frame, center, radius, (200, 200, 200), 2)
    cv2.line(frame, center, (end_x, end_y), (0, 0, 255), 3)
    cv2.putText(frame, "STEER", (center[0] - 22, center[1] + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 偏移指示条
    cv2.rectangle(frame, (w // 2 - 100, h - 82), (w // 2 + 100, h - 77), (50, 50, 50), -1)
    marker_x = int(w // 2 + deviation * w)
    marker_x = max(w // 2 - 100, min(w // 2 + 100, marker_x))
    dev_color = (0, 255, 0) if abs(deviation) < 0.05 else (0, 0, 255)
    cv2.circle(frame, (marker_x, h - 79), 6, dev_color, -1)

    # 状态颜色
    if status == "DANGER":
        status_color = (0, 0, 255)
    elif status == "WARNING":
        status_color = (0, 255, 255)
    else:
        status_color = (0, 255, 0)

    # 左侧数据
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(frame, f"RISK: {max_risk:.1f}", (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

    # 右侧状态
    ttc_text = f"TTC: {min_ttc:.1f}s" if np.isfinite(min_ttc) else "TTC: --"
    cv2.putText(frame, ttc_text, (w - 190, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"STATUS: {status}", (w - 190, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)


def draw_top_bar(frame: np.ndarray, frame_idx: int):
    """绘制顶部信息栏"""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 42), (0, 0, 0), -1)
    cv2.putText(frame, "SOURCE: DEMO", (18, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"FRAME: {frame_idx}", (190, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, "AutoPilot V3.2 Risk Optimized", (400, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)


def generate_demo_screenshot():
    """生成演示截图"""
    output_dir = Path(__file__).parent / "demo_output"
    output_dir.mkdir(exist_ok=True)

    # 场景1：正常行驶
    frame1 = create_demo_frame()
    frame1 = draw_lane_overlay(frame1, 0.02)

    # 前方车辆 - 安全
    draw_detection_box(frame1, (550, 280, 730, 420), "car", "SAFE", 25.0, 8.5)

    draw_dashboard(frame1, 0.02, 5.0, 28.5, "CRUISING", 25.0, 8.5)
    draw_top_bar(frame1, 120)
    cv2.imwrite(str(output_dir / "scene1_cruising.jpg"), frame1)

    # 场景2：预警状态
    frame2 = create_demo_frame()
    frame2 = draw_lane_overlay(frame2, 0.03)

    # 前方车辆 - 预警
    draw_detection_box(frame2, (480, 250, 800, 480), "car", "WARNING", 52.0, 2.8)
    # 侧方车辆 - 安全
    draw_detection_box(frame2, (100, 300, 250, 430), "car", "SAFE", 15.0, 12.0)

    draw_dashboard(frame2, 0.03, -10.0, 25.2, "WARNING", 52.0, 2.8)
    draw_top_bar(frame2, 256)
    cv2.imwrite(str(output_dir / "scene2_warning.jpg"), frame2)

    # 场景3：危险状态
    frame3 = create_demo_frame()
    frame3 = draw_lane_overlay(frame3, 0.01)

    # 前方车辆 - 危险
    draw_detection_box(frame3, (400, 200, 880, 550), "truck", "DANGER", 82.0, 1.2)
    # 行人 - 预警
    draw_detection_box(frame3, (200, 320, 280, 480), "person", "WARNING", 48.0, 3.5)

    draw_dashboard(frame3, 0.01, 2.0, 22.8, "DANGER", 82.0, 1.2)
    draw_top_bar(frame3, 489)

    # 添加 SNAPSHOT SAVED 提示
    cv2.putText(frame3, "SNAPSHOT SAVED", (500, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    cv2.imwrite(str(output_dir / "scene3_danger.jpg"), frame3)

    print(f"已生成 3 张演示截图到: {output_dir}")
    print("  - scene1_cruising.jpg  : 正常行驶状态")
    print("  - scene2_warning.jpg   : 预警状态")
    print("  - scene3_danger.jpg    : 危险状态")

    return output_dir


if __name__ == "__main__":
    generate_demo_screenshot()