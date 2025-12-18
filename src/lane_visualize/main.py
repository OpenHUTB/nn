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
        
        result = self.draw_lane(frame, left_line, right_line)
        return result

def main():
    detector = LaneDetector()
    
    # 逻辑：如果有命令行参数，读取视频；否则读取摄像头
    input_source = 0 # 默认摄像头
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if os.path.exists(input_path):
            input_source = input_path
            print(f"正在打开视频文件: {input_path}")
        else:
            print(f"错误: 找不到文件 {input_path}")
            return
    else:
        print("未提供视频路径，正在打开默认摄像头...")

    cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        print("错误: 无法打开视频源")
        return

    print("按 'q' 键退出程序")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放结束或无法读取帧")
            break

        # 处理帧
        processed_frame = detector.process_frame(frame)

        # 显示结果
        if processed_frame is not None:
            cv2.imshow('Lane Detection System', processed_frame)
        else:
            cv2.imshow('Lane Detection System', frame)

        # 按 'q' 退出，设置 25ms 延时（约 40fps）
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()