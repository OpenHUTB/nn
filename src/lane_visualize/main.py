import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

class LaneDetector:
    def __init__(self):
        # 霍夫变换参数
        self.rho = 1
        self.theta = np.pi / 180
        self.threshold = 15
        self.min_line_length = 40
        self.max_line_gap = 20
        self.vertices = None

    def region_of_interest(self, img):
        """定义感兴趣区域（ROI）"""
        mask = np.zeros_like(img)
        if self.vertices is None:
            height, width = img.shape
            # 调整梯形区域以适应大多数行车记录仪视角
            self.vertices = np.array([[
                (width * 0.1, height),            # 左下
                (width * 0.45, height * 0.6),     # 左上
                (width * 0.55, height * 0.6),     # 右上
                (width * 0.9, height)             # 右下
            ]], dtype=np.int32)
        cv2.fillPoly(mask, self.vertices, 255)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    def detect_edges(self, img):
        """Canny边缘检测"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def detect_lines(self, edges):
        """霍夫变换检测直线"""
        lines = cv2.HoughLinesP(edges, self.rho, self.theta, self.threshold,
                                np.array([]), minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap)
        return lines

    def average_slope_intercept(self, lines):
        """计算左右车道线的平均斜率和截距"""
        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0: continue
                slope = (y2 - y1) / (x2 - x1)
                
                # 过滤不合理的斜率 (太水平或太垂直)
                if abs(slope) < 0.5 or abs(slope) > 2:
                    continue

                intercept = y1 - slope * x1
                # 图像坐标系中，y轴向下，所以斜率为负是左车道，正为右车道
                if slope < 0:
                    left_lines.append((slope, intercept))
                else:
                    right_lines.append((slope, intercept))

        left_avg = np.average(left_lines, axis=0) if left_lines else None
        right_avg = np.average(right_lines, axis=0) if right_lines else None
        return left_avg, right_avg

    def make_line_points(self, avg_line, y_min, y_max):
        """生成绘制用的坐标点"""
        if avg_line is None:
            return None
        slope, intercept = avg_line
        # 防止除以0
        if slope == 0: 
            return None
            
        try:
            x_min = int((y_min - intercept) / slope)
            x_max = int((y_max - intercept) / slope)
            return [(x_min, y_min), (x_max, y_max)]
        except OverflowError:
            return None

    def draw_lane(self, img, left_line, right_line):
        """绘制半透明车道区域"""
        lane_img = np.zeros_like(img)
        
        # 确保两条线都检测到了才画多边形
        if left_line is not None and right_line is not None:
            left_pts = np.array([left_line[0], left_line[1]], dtype=np.int32)
            right_pts = np.array([right_line[0], right_line[1]], dtype=np.int32)
            
            # 创建多边形顶点
            pts = np.vstack([left_pts, np.flipud(right_pts)])
            cv2.fillPoly(lane_img, [pts], (0, 255, 0)) # 绿色填充
            
        # 无论是否形成区域，都尝试画线
        if left_line is not None:
            cv2.line(lane_img, left_line[0], left_line[1], (255, 0, 0), 10) # 蓝色线
        if right_line is not None:
            cv2.line(lane_img, right_line[0], right_line[1], (255, 0, 0), 10) # 蓝色线

        result = cv2.addWeighted(img, 0.8, lane_img, 0.4, 0)
        return result

    def process_frame(self, frame):
        if frame is None: return None
        edges = self.detect_edges(frame)
        roi_edges = self.region_of_interest(edges)
        lines = self.detect_lines(roi_edges)
        
        height, width = frame.shape[:2]
        left_avg, right_avg = self.average_slope_intercept(lines)
        
        y_min = int(height * 0.65) # 稍作调整，不要画太远
        y_max = height

        left_line = self.make_line_points(left_avg, y_min, y_max)
        right_line = self.make_line_points(right_avg, y_min, y_max)
        
      import cv2
import numpy as np
import sys
import os
from collections import deque

class LaneDetector:
    def __init__(self):
        # --- 1. 霍夫变换参数 (经过微调) ---
        self.rho = 1              # 距离分辨率
        self.theta = np.pi / 180  # 角度分辨率
        self.threshold = 20       # 阈值：被认为是一条直线所需的最小投票数
        self.min_line_length = 20 # 最小线段长度
        self.max_line_gap = 300   # 允许线段断裂的最大距离 (设大一点可以连接虚线)

        # --- 2. 历史缓存 (用于平滑防抖) ---
        # 队列长度为10，表示平均过去10帧的结果
        self.left_lines_buffer = deque(maxlen=10)
        self.right_lines_buffer = deque(maxlen=10)

        self.vertices = None

    def color_filter(self, img):
        """
        颜色过滤：将图像转换为HLS空间，专门提取白色和黄色
        这比单纯的灰度边缘检测抗干扰能力强得多
        """
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # 1. 白色过滤
        # L通道(亮度) > 200 基本就是白色
        lower_white = np.array([0, 200, 0])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(hls, lower_white, upper_white)

        # 2. 黄色过滤
        # H通道(色相) 在 10-40 之间是黄色/橙色
        lower_yellow = np.array([10, 0, 100])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

        # 合并掩码
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # 将原图和掩码进行按位与，提取出只有颜色的部分
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        return masked_img

    def region_of_interest(self, img):
        mask = np.zeros_like(img)
        
        # 动态定义ROI：针对行车记录仪视角优化
        if self.vertices is None:
            height, width = img.shape
            self.vertices = np.array([[
                (0, height),                    # 左下 (贴边)
                (width * 0.45, height * 0.6),   # 左上 (远处中心左)
                (width * 0.55, height * 0.6),   # 右上 (远处中心右)
                (width, height)                 # 右下 (贴边)
            ]], dtype=np.int32)

        cv2.fillPoly(mask, self.vertices, 255)
        return cv2.bitwise_and(img, mask)

    def detect_edges(self, img):
        # 先进行颜色过滤
        color_filtered = self.color_filter(img)
        
        # 转灰度
        gray = cv2.cvtColor(color_filtered, cv2.COLOR_BGR2GRAY)
        
        # 强高斯模糊，消除路面纹理噪声
        blur = cv2.GaussianBlur(gray, (7, 7), 0) # 增大核大小
        
        # Canny边缘检测
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def average_slope_intercept(self, lines):
        left_lines = []
        right_lines = []

        if lines is None:
            return None, None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue # 避免垂直线除以0
            
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # --- 3. 斜率过滤 (关键) ---
            # 左车道斜率通常在 -0.9 到 -0.3 之间
            # 右车道斜率通常在 0.3 到 0.9 之间
            # 过滤掉过于水平或过于垂直的线
            if -0.9 < slope < -0.3:
                left_lines.append((slope, intercept))
            elif 0.3 < slope < 0.9:
                right_lines.append((slope, intercept))

        # 计算当前帧的平均值
        left_avg = np.average(left_lines, axis=0) if left_lines else None
        right_avg = np.average(right_lines, axis=0) if right_lines else None
        
        return left_avg, right_avg

    def smooth_lines(self, left_current, right_current):
        """
        平滑算法：结合历史帧数据，避免线条跳变
        """
        # 处理左线
        if left_current is not None:
            self.left_lines_buffer.append(left_current)
        
        # 处理右线
        if right_current is not None:
            self.right_lines_buffer.append(right_current)

        # 计算缓冲区平均值
        left_smooth = np.average(self.left_lines_buffer, axis=0) if self.left_lines_buffer else None
        right_smooth = np.average(self.right_lines_buffer, axis=0) if self.right_lines_buffer else None

        return left_smooth, right_smooth

    def make_line_points(self, line, y_min, y_max):
        if line is None: return None
        slope, intercept = line
        
        if abs(slope) < 1e-3: return None # 防止水平线

        try:
            x_min = int((y_min - intercept) / slope)
            x_max = int((y_max - intercept) / slope)
            return ((x_min, y_min), (x_max, y_max))
        except (OverflowError, ValueError):
            return None

    def process_frame(self, frame):
        if frame is None: return None

        # 1. 边缘检测 (含颜色过滤)
        edges = self.detect_edges(frame)
        
        # 2. ROI 裁剪
        roi = self.region_of_interest(edges)
        
        # 3. 霍夫直线检测
        lines = cv2.HoughLinesP(roi, self.rho, self.theta, self.threshold, 
                                np.array([]), minLineLength=self.min_line_length, 
                                maxLineGap=self.max_line_gap)
        
        # 4. 分类并计算平均线
        left_raw, right_raw = self.average_slope_intercept(lines)
        
        # 5. 历史平滑 (这是减少抖动的关键)
        left_avg, right_avg = self.smooth_lines(left_raw, right_raw)

        # 6. 生成绘制坐标
        height = frame.shape[0]
        y_min = int(height * 0.65)
        y_max = height
        
        left_pts = self.make_line_points(left_avg, y_min, y_max)
        right_pts = self.make_line_points(right_avg, y_min, y_max)

        # 7. 绘制
        return self.draw_lane(frame, left_pts, right_pts)

    def draw_lane(self, img, left_pts, right_pts):
        lane_img = np.zeros_like(img)
        
        # 绘制半透明多边形
        if left_pts is not None and right_pts is not None:
            pts = np.array([left_pts[0], left_pts[1], right_pts[1], right_pts[0]], dtype=np.int32)
            cv2.fillPoly(lane_img, [pts], (0, 255, 0))
        
        # 绘制实线
        if left_pts:
            cv2.line(lane_img, left_pts[0], left_pts[1], (255, 0, 0), 10)
        if right_pts:
            cv2.line(lane_img, right_pts[0], right_pts[1], (0, 0, 255), 10)
            
        return cv2.addWeighted(img, 1, lane_img, 0.3, 0)

def main():
    detector = LaneDetector()
    
    # 自动判断参数
    input_source = 0
    if len(sys.argv) > 1:
        input_source = sys.argv[1]
    
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error opening source: {input_source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 处理并显示
        result = detector.process_frame(frame)
        
        # 调整窗口大小以便观察
        cv2.namedWindow('Lane Optimization', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Lane Optimization', 1280, 720)
        
        if result is not None:
            cv2.imshow('Lane Optimization', result)
        else:
            cv2.imshow('Lane Optimization', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()