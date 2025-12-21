import cv2
import numpy as np
import sys
import time
import os
import math
from collections import deque
from yolo_det import ObjectDetector

# ================= 配置区 =================
CANNY_LOW, CANNY_HIGH = 50, 150
ROI_TOP, ROI_HEIGHT = 0.40, 0.60
SKIP_FRAMES = 3  # 跳帧数
WARNING_RATIO = 0.20  # 碰撞预警阈值
STEER_SENSITIVITY = 1.5  # 转向灵敏度 (越大方向盘转得越快)


# ==========================================

class EventLogger:
    """黑匣子：负责记录危险事件"""

    def __init__(self, save_dir="events"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.last_save_time = 0
        self.cooldown = 2.0  # 两次抓拍最小间隔(秒)

    def log_danger(self, frame, obj_name):
        now = time.time()
        if now - self.last_save_time > self.cooldown:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/danger_{timestamp}_{obj_name}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 危险已抓拍: {filename}")
            self.last_save_time = now
            return True
        return False


class LaneSystem:
    def __init__(self):
        self.left_fit_avg = None
        self.right_fit_avg = None
        self.vertices = None

    def get_lane_info(self, frame):
        """返回: (lane_mask, deviation_percent, curvature_angle)"""
        if frame is None: return None, 0, 0

        # 1. 视觉处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

        # 2. ROI
        h, w = frame.shape[:2]
        if self.vertices is None:
            top_w = w * ROI_TOP
            self.vertices = np.array([[
                (0, h),
                (int(w * 0.5 - top_w / 2), int(h * ROI_HEIGHT)),
                (int(w * 0.5 + top_w / 2), int(h * ROI_HEIGHT)),
                (w, h)
            ]], dtype=np.int32)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self.vertices, 255)
        roi = cv2.bitwise_and(edges, mask)

        # 3. 霍夫变换
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=100)

        # 4. 计算拟合线
        l_fit, r_fit = self.avg_lines(lines)
        self.left_fit_avg = self.smooth(self.left_fit_avg, l_fit)
        self.right_fit_avg = self.smooth(self.right_fit_avg, r_fit)

        # 5. 计算绘制点
        y_min = int(h * ROI_HEIGHT) + 40
        l_pts = self.make_pts(self.left_fit_avg, y_min, h)
        r_pts = self.make_pts(self.right_fit_avg, y_min, h)

        # 6. 计算偏离度与转向角
        deviation = 0
        angle = 0
        if l_pts and r_pts:
            lane_center = (l_pts[0][0] + r_pts[1][0]) / 2
            screen_center = w / 2
            # 偏离度: +右偏, -左偏
            deviation = (lane_center - screen_center) / w

            # 简单估算弯道角度 (基于左右线斜率的平均值)
            l_slope = self.left_fit_avg[0]
            r_slope = self.right_fit_avg[0]
            # 将斜率转换为角度 (这是一个近似值)
            angle = math.degrees(math.atan((l_slope + r_slope) / 2))

            # 7. 绘制
        lane_layer = np.zeros_like(frame)
        if l_pts and r_pts:
            pts = np.array([l_pts[0], l_pts[1], r_pts[1], r_pts[0]], dtype=np.int32)
            cv2.fillPoly(lane_layer, [pts], (0, 255, 0))

        return lane_layer, deviation, angle

    def avg_lines(self, lines):
        lefts, rights = [], []
        if lines is None: return None, None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if abs(slope) < 0.3 or abs(slope) > 5: continue
            if slope < 0:
                lefts.append((slope, intercept))
            else:
                rights.append((slope, intercept))
        return (np.mean(lefts, axis=0) if lefts else None,
                np.mean(rights, axis=0) if rights else None)

    def smooth(self, curr, new_val):
        return curr * 0.8 + new_val * 0.2 if curr is not None and new_val is not None else new_val

    def make_pts(self, line, y1, y2):
        if line is None: return None
        s, i = line
        if abs(s) < 1e-3: return None
        try:
            return ((int((y1 - i) / s), y1), (int((y2 - i) / s), y2))
        except:
            return None


def draw_dashboard(img, deviation, steer_angle, fps, status):
    """绘制高科技仪表盘 (方向盘 + 数据)"""
    h, w = img.shape[:2]

    # 1. 底部黑色面板
    cv2.rectangle(img, (0, h - 80), (w, h), (0, 0, 0), -1)

    # 2. 虚拟方向盘 (Steering Wheel)
    center = (w // 2, h - 40)
    radius = 30
    # 计算旋转后的终点
    # steer_angle 是基于路面斜率的，我们需要把它放大一点显示
    display_angle = steer_angle * 5 * STEER_SENSITIVITY
    # 限制最大转角
    display_angle = max(-90, min(90, display_angle))

    rad = math.radians(display_angle - 90)  # -90因为OpenCV里0度是3点钟方向
    end_x = int(center[0] + radius * math.cos(rad))
    end_y = int(center[1] + radius * math.sin(rad))

    # 画圆和指针
    cv2.circle(img, center, radius, (200, 200, 200), 2)
    cv2.line(img, center, (end_x, end_y), (0, 0, 255), 3)
    cv2.putText(img, "STEER", (center[0] - 20, center[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 3. 偏离指示条
    bar_width = 200
    cv2.rectangle(img, (w // 2 - 100, h - 75), (w // 2 + 100, h - 70), (50, 50, 50), -1)
    # 指示点
    marker_x = int(w // 2 + deviation * w)  # 偏差值映射到像素
    marker_x = max(w // 2 - 100, min(w // 2 + 100, marker_x))
    color = (0, 255, 0) if abs(deviation) < 0.05 else (0, 0, 255)
    cv2.circle(img, (marker_x, h - 72), 6, color, -1)

    # 4. 数据显示
    cv2.putText(img, f"FPS: {fps:.1f}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(img, f"STATUS: {status}", (w - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "sample.hevc"
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): return

    lane_sys = LaneSystem()
    yolo_sys = ObjectDetector()
    logger = EventLogger()

    print("🚀 AutoPilot V3.0: 启动虚拟控制与取证系统...")

    frame_count = 0
    current_dets = []

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        display = frame.copy()
        h, w = frame.shape[:2]

        # --- A. 感知层: YOLO检测 (跳帧) ---
        if frame_count % (SKIP_FRAMES + 1) == 0:
            _, current_dets = yolo_sys.detect(frame)

        is_danger = False
        danger_obj = ""

        # --- B. 决策层: 碰撞分析 ---
        for det in current_dets:
            x1, y1, x2, y2 = det['box']
            width_ratio = det['width'] / w

            color = (0, 255, 0)
            if width_ratio > WARNING_RATIO:
                color = (0, 0, 255)
                is_danger = True
                danger_obj = det['class']
                # 画面中心大字警告
                cv2.putText(display, "BRAKE!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, det['class'], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # --- C. 黑匣子: 自动抓拍 ---
        if is_danger:
            # 传入原始未画框的帧，还是画了框的？画了框的更直观
            if logger.log_danger(display, danger_obj):
                cv2.putText(display, "SNAPSHOT SAVED", (w // 2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)

        # --- D. 控制层: 车道与转向 ---
        lane_layer, deviation, steer_angle = lane_sys.get_lane_info(frame)
        if lane_layer is not None:
            display = cv2.addWeighted(display, 1, lane_layer, 0.4, 0)

        # --- E. 交互层: 仪表盘 ---
        fps = 1.0 / (time.time() - t_start)
        status = "DANGER" if is_danger else "CRUISING"
        draw_dashboard(display, deviation, steer_angle, fps, status)

        cv2.imshow('AutoPilot V3.0 - Control Dashboard', display)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()