import cv2
import numpy as np
import sys
import time
import os
import math
from collections import deque  # 【新增】引入双端队列用于历史缓冲
from yolo_det import ObjectDetector

# ================= 配置区 =================
# --- 视觉参数 (新) ---
# Canny 边缘检测阈值 (在颜色过滤后应用，可以适当降低)
CANNY_LOW, CANNY_HIGH = 30, 100
# ROI 区域设置
ROI_TOP, ROI_HEIGHT = 0.40, 0.60

# --- HLS 颜色阈值 (关键优化点) ---
# 白色掩码范围 (对亮度L要求高，对色相H和饱和度S要求低)
# 注意：OpenCV中 H范围是0-180, L和S是0-255
WHITE_LOWER = np.array([0, 160, 0], dtype=np.uint8)
WHITE_UPPER = np.array([180, 255, 100], dtype=np.uint8)
# 黄色掩码范围 (H通常在15-35之间，S要够高)
YELLOW_LOWER = np.array([15, 80, 80], dtype=np.uint8)
YELLOW_UPPER = np.array([35, 255, 255], dtype=np.uint8)

# --- 平滑参数 (新) ---
# 历史记录长度：越大越稳，但滞后感越强。8-12帧通常是不错的平衡点。
HISTORY_LEN = 10

# --- 交互参数 ---
SKIP_FRAMES = 3  # 跳帧数
WARNING_RATIO = 0.20  # 碰撞预警阈值
STEER_SENSITIVITY = 1.5  # 转向灵敏度


# ==========================================

class EventLogger:
    """黑匣子：负责记录危险事件 (保持 V3.0 功能不变)"""

    def __init__(self, save_dir="events"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.last_save_time = 0
        self.cooldown = 2.0

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
    """【重构】车道线系统：颜色过滤 + 加权拟合 + 历史平滑"""

    def __init__(self):
        # 使用双端队列存储最近 N 帧的拟合结果 (斜率, 截距)
        self.left_history = deque(maxlen=HISTORY_LEN)
        self.right_history = deque(maxlen=HISTORY_LEN)
        # 存储最近一次稳定的检测结果，用于兜底
        self.last_stable_left = None
        self.last_stable_right = None
        self.vertices = None

    def color_filter(self, frame):
        """【核心优化1】HLS颜色空间过滤，只提取白色和黄色"""
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        white_mask = cv2.inRange(hls, WHITE_LOWER, WHITE_UPPER)
        yellow_mask = cv2.inRange(hls, YELLOW_LOWER, YELLOW_UPPER)
        # 合并掩码
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        # 轻微膨胀，连接断续的线段
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        return combined_mask

    def get_lane_info(self, frame):
        """返回: (lane_mask, deviation_percent, curvature_angle)"""
        if frame is None: return None, 0, 0
        h, w = frame.shape[:2]

        # 1. 视觉处理流程升级
        # 旧版: Gray -> Canny
        # 新版: HLS Color Mask -> Canny (只在筛选出的颜色区域找边缘)
        color_mask = self.color_filter(frame)
        edges = cv2.Canny(color_mask, CANNY_LOW, CANNY_HIGH)

        # 2. ROI (感兴趣区域)
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

        # 3. 霍夫变换 (参数微调：提高最小长度要求，过滤噪点)
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=20, minLineLength=30, maxLineGap=100)

        # 4. 【核心优化2】处理线段：分类、过滤、加权拟合
        curr_l_fit, curr_r_fit = self.process_lines(lines, h)

        # 5. 【核心优化3】历史缓冲平滑
        # 如果当前帧检测到了，加入历史队列
        if curr_l_fit is not None: self.left_history.append(curr_l_fit)
        if curr_r_fit is not None: self.right_history.append(curr_r_fit)

        # 计算平滑后的结果（取历史平均值）
        # 如果历史队列为空，尝试使用上一次的稳定值兜底
        smooth_left = np.mean(self.left_history, axis=0) if self.left_history else self.last_stable_left
        smooth_right = np.mean(self.right_history, axis=0) if self.right_history else self.last_stable_right

        # 更新稳定兜底值
        if smooth_left is not None: self.last_stable_left = smooth_left
        if smooth_right is not None: self.last_stable_right = smooth_right

        # 6. 计算绘制点 (使用平滑后的结果)
        y_min = int(h * ROI_HEIGHT) + 40
        # 只有当左右两条线都有稳定结果时才绘制
        l_pts = self.make_pts(smooth_left, y_min, h) if smooth_left is not None else None
        r_pts = self.make_pts(smooth_right, y_min, h) if smooth_right is not None else None

        # 7. 计算偏离度与转向角
        deviation = 0
        angle = 0
        lane_layer = np.zeros_like(frame)

        if l_pts and r_pts:
            # 绘制绿色车道区域
            pts = np.array([l_pts[0], l_pts[1], r_pts[1], r_pts[0]], dtype=np.int32)
            cv2.fillPoly(lane_layer, [pts], (0, 255, 0))

            # 计算控制数据
            lane_center = (l_pts[0][0] + r_pts[1][0]) / 2
            screen_center = w / 2
            deviation = (lane_center - screen_center) / w

            # 计算转向角：取左右斜率的平均值代表道路趋势
            l_slope = smooth_left[0]
            r_slope = smooth_right[0]
            avg_slope = (l_slope + r_slope) / 2
            # 转换为角度，取反以适配仪表盘显示习惯（负斜率=左转=正角度）
            angle = -math.degrees(math.atan(avg_slope))

        return lane_layer, deviation, angle

    def process_lines(self, lines, height):
        """【核心优化2实现】加权拟合，长线段权重更大"""
        left_lines, right_lines = [], []
        left_weights, right_weights = [], []  # 权重列表

        if lines is None: return None, None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue  # 防止垂直线除零

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # 计算线段长度作为权重

            # 过滤掉过于水平或垂直的噪点线段
            if abs(slope) < 0.3 or abs(slope) > 3.0: continue

            # 简单的左右分类
            if slope < 0:  # 左车道线通常斜率为负
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:  # 右车道线通常斜率为正
                right_lines.append((slope, intercept))
                right_weights.append(length)

        # 使用加权平均计算最终拟合线
        # 如果某侧没有检测到线，返回 None
        left_fit = np.average(left_lines, axis=0, weights=left_weights) if left_lines else None
        right_fit = np.average(right_lines, axis=0, weights=right_weights) if right_lines else None

        return left_fit, right_fit

    def make_pts(self, line, y1, y2):
        if line is None: return None
        s, i = line
        # 防止斜率过小导致计算出的x坐标飞出天际
        if abs(s) < 1e-2: return None
        try:
            x1 = int((y1 - i) / s)
            x2 = int((y2 - i) / s)
            # 增加边界检查，防止绘制错误
            if x1 < -2000 or x1 > 4000 or x2 < -2000 or x2 > 4000: return None
            return ((x1, y1), (x2, y2))
        except:
            return None


def draw_dashboard(img, deviation, steer_angle, fps, status):
    """绘制高科技仪表盘 (保持 V3.0 功能不变)"""
    h, w = img.shape[:2]
    # 1. 底部黑色面板
    cv2.rectangle(img, (0, h - 80), (w, h), (0, 0, 0), -1)
    # 2. 虚拟方向盘
    center = (w // 2, h - 40)
    radius = 30
    display_angle = steer_angle * 4 * STEER_SENSITIVITY  # 调整了显示系数
    display_angle = max(-90, min(90, display_angle))
    rad = math.radians(display_angle - 90)
    end_x = int(center[0] + radius * math.cos(rad))
    end_y = int(center[1] + radius * math.sin(rad))
    cv2.circle(img, center, radius, (200, 200, 200), 2)
    cv2.line(img, center, (end_x, end_y), (0, 0, 255), 3)
    cv2.putText(img, "STEER", (center[0] - 20, center[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    # 3. 偏离指示条
    cv2.rectangle(img, (w // 2 - 100, h - 75), (w // 2 + 100, h - 70), (50, 50, 50), -1)
    marker_x = int(w // 2 + deviation * w)
    marker_x = max(w // 2 - 100, min(w // 2 + 100, marker_x))
    color = (0, 255, 0) if abs(deviation) < 0.05 else (0, 0, 255)
    cv2.circle(img, (marker_x, h - 72), 6, color, -1)
    # 4. 数据显示
    cv2.putText(img, f"FPS: {fps:.1f}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(img, f"STATUS: {status}", (w - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "sample.hevc"
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return

    # 初始化新的车道系统
    lane_sys = LaneSystem()
    yolo_sys = ObjectDetector()
    logger = EventLogger()

    print("🚀 AutoPilot V3.1 (Stable): 启动优化版控制系统...")

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
                cv2.putText(display, "BRAKE!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, det['class'], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # --- C. 黑匣子: 自动抓拍 ---
        if is_danger:
            if logger.log_danger(display, danger_obj):
                cv2.putText(display, "SNAPSHOT SAVED", (w // 2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)

        # --- D. 控制层: 优化的车道与转向 ---
        # 这里调用的是新的 LaneSystem 逻辑
        lane_layer, deviation, steer_angle = lane_sys.get_lane_info(frame)
        if lane_layer is not None:
            display = cv2.addWeighted(display, 1, lane_layer, 0.4, 0)

        # --- E. 交互层: 仪表盘 ---
        fps = 1.0 / (time.time() - t_start)
        status = "DANGER" if is_danger else "CRUISING"
        draw_dashboard(display, deviation, steer_angle, fps, status)

        cv2.imshow('AutoPilot V3.1 (Stable)', display)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()