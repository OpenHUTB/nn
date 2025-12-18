import cv2
import numpy as np
import sys
from collections import deque

class LaneDetector:
    def __init__(self):
        # --- 1. 放宽霍夫变换参数 ---
        # 即使是断断续续的线也能被连起来
        self.rho = 2              # 稍微降低精度以换取更多检测
        self.theta = np.pi / 180
        self.threshold = 15       # 极低阈值：只要有15个点在一条线上就认为是线
        self.min_line_length = 10 # 允许非常短的线段
        self.max_line_gap = 50    # 允许线段之间有缺口

        # 历史缓存
        self.left_lines_buffer = deque(maxlen=10)
        self.right_lines_buffer = deque(maxlen=10)
        self.vertices = None

    def process_image_pipeline(self, img):
        """
        处理流水线，返回边缘图以供调试
        """
        # 1. 灰度化 (放弃HLS颜色过滤，改用经典的鲁棒灰度法)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. 提高对比度 (直方图均衡化)，应对阴天或昏暗环境
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)

        # 3. 高斯模糊，去噪
        blur = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)

        # 4. Canny 边缘检测 (动态阈值)
        # 自动计算图像的中值亮度，根据亮度调整阈值
        v = np.median(blur)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(blur, lower, upper)

        # 5. ROI 区域掩码
        roi_edges = self.region_of_interest(edges)
        
        return roi_edges

    def region_of_interest(self, img):
        mask = np.zeros_like(img)
        if self.vertices is None:
            height, width = img.shape
            # 定义一个更宽的梯形，防止漏掉旁边的线
            self.vertices = np.array([[
                (width * 0.1, height),          # 左下
                (width * 0.4, height * 0.6),    # 左上
                (width * 0.6, height * 0.6),    # 右上
                (width * 0.9, height)           # 右下
            ]], dtype=np.int32)
        cv2.fillPoly(mask, self.vertices, 255)
        return cv2.bitwise_and(img, mask)

    def average_slope_intercept(self, lines):
        left_lines = []
        right_lines = []

        if lines is None:
            return None, None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue
            
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # 放宽斜率限制：只要不是绝对水平或绝对垂直都算
            # 很多弯道斜率会很小
            if abs(slope) < 0.1 or abs(slope) > 10:
                continue

            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

        left_avg = np.average(left_lines, axis=0) if left_lines else None
        right_avg = np.average(right_lines, axis=0) if right_lines else None
        return left_avg, right_avg

    def smooth_lines(self, left_current, right_current):
        if left_current is not None:
            self.left_lines_buffer.append(left_current)
        if right_current is not None:
            self.right_lines_buffer.append(right_current)

        left_smooth = np.average(self.left_lines_buffer, axis=0) if self.left_lines_buffer else None
        right_smooth = np.average(self.right_lines_buffer, axis=0) if self.right_lines_buffer else None
        return left_smooth, right_smooth

    def make_line_points(self, line, y_min, y_max):
        if line is None: return None
        slope, intercept = line
        if abs(slope) < 1e-3: return None
        try:
            x_min = int((y_min - intercept) / slope)
            x_max = int((y_max - intercept) / slope)
            return ((x_min, y_min), (x_max, y_max))
        except:
            return None

    def process_frame(self, frame):
        if frame is None: return None, None

        # 获取调试用的边缘图
        edges = self.process_image_pipeline(frame)
        
        # 霍夫检测
        lines = cv2.HoughLinesP(edges, self.rho, self.theta, self.threshold, 
                                np.array([]), minLineLength=self.min_line_length, 
                                maxLineGap=self.max_line_gap)
        
        left_raw, right_raw = self.average_slope_intercept(lines)
        left_avg, right_avg = self.smooth_lines(left_raw, right_raw)

        height = frame.shape[0]
        y_min = int(height * 0.65)
        y_max = height
        
        left_pts = self.make_line_points(left_avg, y_min, y_max)
        right_pts = self.make_line_points(right_avg, y_min, y_max)

        result_frame = self.draw_lane(frame, left_pts, right_pts)
        
        # 将边缘图转为彩色以便显示
        debug_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 在调试图上画出ROI区域，方便你看是否对准了
        cv2.polylines(debug_frame, [self.vertices], True, (0, 0, 255), 2)

        return result_frame, debug_frame

    def draw_lane(self, img, left_pts, right_pts):
        lane_img = np.zeros_like(img)
        if left_pts is not None and right_pts is not None:
            pts = np.array([left_pts[0], left_pts[1], right_pts[1], right_pts[0]], dtype=np.int32)
            cv2.fillPoly(lane_img, [pts], (0, 255, 0))
        
        if left_pts: cv2.line(lane_img, left_pts[0], left_pts[1], (255, 0, 0), 10)
        if right_pts: cv2.line(lane_img, right_pts[0], right_pts[1], (0, 0, 255), 10)
            
        return cv2.addWeighted(img, 1, lane_img, 0.3, 0)

def main():
    detector = LaneDetector()
    input_source = sys.argv[1] if len(sys.argv) > 1 else 0
    
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print("无法打开视频")
        return

    print("调试模式已启动：请观察 'Debug View' 窗口")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 获取结果和调试图
        result, debug_view = detector.process_frame(frame)
        
        # 拼接图像显示 (左右并排)
        # 将原图缩小一点以便并排显示
        h, w = frame.shape[:2]
        small_result = cv2.resize(result, (w//2, h//2))
        small_debug = cv2.resize(debug_view, (w//2, h//2))
        
        # 纵向拼接：上面是结果，下面是调试（黑白边缘图）
        combined = np.vstack((small_result, small_debug))
        
        cv2.imshow('Lane Detection Debugger', combined)

        # 慢动作播放 (50ms延时)，方便看清
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()