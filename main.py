import cv2
import numpy as np
import sys
from collections import deque
import time

# 引入我们刚才写的 YOLO 模块
from yolo_det import ObjectDetector

# ==========================================
# 👇 参数配置区 (填入你之前 Tuner 调好的参数)
# ==========================================
CANNY_LOW = 50        
CANNY_HIGH = 150      
ROI_TOP_WIDTH = 0.40  
ROI_HEIGHT_POS = 0.60 
HOUGH_THRESH = 20     
MIN_LINE_LEN = 20     
MAX_LINE_GAP = 100    
# ==========================================

class LaneDetector:
    def __init__(self):
        self.left_lines_buffer = deque(maxlen=10)
        self.right_lines_buffer = deque(maxlen=10)
        self.vertices = None

    def region_of_interest(self, img):
        mask = np.zeros_like(img)
        if self.vertices is None:
            height, width = img.shape
            top_w = width * ROI_TOP_WIDTH
            top_x_center = width * 0.5
            bl = (0, height)
            tl = (int(top_x_center - top_w/2), int(height * ROI_HEIGHT_POS))
            tr = (int(top_x_center + top_w/2), int(height * ROI_HEIGHT_POS))
            br = (width, height)
            self.vertices = np.array([[bl, tl, tr, br]], dtype=np.int32)
        cv2.fillPoly(mask, self.vertices, 255)
        return cv2.bitwise_and(img, mask)

    def process_lane(self, frame):
        if frame is None: return frame
        
        # 1. 边缘检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        
        # 2. ROI
        roi = self.region_of_interest(edges)
        
        # 3. 霍夫变换
        lines = cv2.HoughLinesP(roi, 1, np.pi/180, HOUGH_THRESH, 
                                minLineLength=MIN_LINE_LEN, 
                                maxLineGap=MAX_LINE_GAP)
        
        # 4. 计算与平滑
        left_raw, right_raw = self.average_slope_intercept(lines)
        left_avg, right_avg = self.smooth_lines(left_raw, right_raw)

        # 5. 绘制
        height = frame.shape[0]
        y_min = int(height * ROI_HEIGHT_POS) + 50
        y_max = height
        left_pts = self.make_line_points(left_avg, y_min, y_max)
        right_pts = self.make_line_points(right_avg, y_min, y_max)

        return self.draw_lane(frame, left_pts, right_pts)

    def average_slope_intercept(self, lines):
        left_lines = []
        right_lines = []
        if lines is None: return None, None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if abs(slope) < 0.3 or abs(slope) > 10: continue
            if slope < 0: left_lines.append((slope, intercept))
            else: right_lines.append((slope, intercept))
        left_avg = np.average(left_lines, axis=0) if left_lines else None
        right_avg = np.average(right_lines, axis=0) if right_lines else None
        return left_avg, right_avg

    def smooth_lines(self, left_current, right_current):
        if left_current is not None: self.left_lines_buffer.append(left_current)
        if right_current is not None: self.right_lines_buffer.append(right_current)
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
        except: return None

    def draw_lane(self, img, left_pts, right_pts):
        lane_img = np.zeros_like(img)
        if left_pts is not None and right_pts is not None:
            pts = np.array([left_pts[0], left_pts[1], right_pts[1], right_pts[0]], dtype=np.int32)
            cv2.fillPoly(lane_img, [pts], (0, 255, 0))
        if left_pts: cv2.line(lane_img, left_pts[0], left_pts[1], (255, 0, 0), 10)
        if right_pts: cv2.line(lane_img, right_pts[0], right_pts[1], (0, 0, 255), 10)
        # 注意：这里我们返回 lane_img (纯遮罩)，而不是叠加后的图，方便后面统一叠加
        return lane_img

def main():
    input_source = sys.argv[1] if len(sys.argv) > 1 else "sample.hevc"
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print("无法打开视频")
        return

    # 1. 初始化两个检测器
    lane_detector = LaneDetector()
    yolo_detector = ObjectDetector(model_name='yolov8n.pt') # 首次运行会自动下载 6MB 的模型

    print("系统启动中...")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # -----------------------------------------------------
        # 步骤 A: 运行 YOLO 车辆检测 (返回画好框的图)
        # -----------------------------------------------------
        frame_with_cars = yolo_detector.detect(frame)

        # -----------------------------------------------------
        # 步骤 B: 运行车道线检测 (返回纯车道层)
        # -----------------------------------------------------
        lane_layer = lane_detector.process_lane(frame)

        # -----------------------------------------------------
        # 步骤 C: 合并图层
        # -----------------------------------------------------
        # 将半透明车道层 叠加到 画了车的图上
        final_result = cv2.addWeighted(frame_with_cars, 1, lane_layer, 0.4, 0)

        # 计算并显示 FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(final_result, f"FPS: {fps:.1f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('AutoPilot System V2.0', final_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()