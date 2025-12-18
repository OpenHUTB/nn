import cv2
import numpy as np
import sys
from collections import deque

# ==========================================
# 👇 请在这里填入你在 Tuner 中调出的“最好”的数值
# ==========================================
CANNY_LOW = 50  # 你的 Canny Low
CANNY_HIGH = 150  # 你的 Canny High

ROI_TOP_WIDTH = 0.40  # 你的 ROI Top W (例如滑动条是40，这里写 0.40)
ROI_HEIGHT_POS = 0.60  # 你的 ROI Height (例如滑动条是60，这里写 0.60)

HOUGH_THRESH = 20  # 你的 Hough Thresh
MIN_LINE_LEN = 20  # 你的 Min Length
MAX_LINE_GAP = 100  # 你的 Max Gap


# ==========================================

class LaneDetector:
    def __init__(self):
        # 历史缓存 (用于平滑防抖)
        self.left_lines_buffer = deque(maxlen=10)
        self.right_lines_buffer = deque(maxlen=10)
        self.vertices = None

    def region_of_interest(self, img):
        mask = np.zeros_like(img)
        if self.vertices is None:
            height, width = img.shape

            # 使用填入的参数计算梯形
            top_w = width * ROI_TOP_WIDTH
            top_x_center = width * 0.5

            # 梯形四个顶点
            bl = (0, height)  # 左下
            tl = (int(top_x_center - top_w / 2), int(height * ROI_HEIGHT_POS))  # 左上
            tr = (int(top_x_center + top_w / 2), int(height * ROI_HEIGHT_POS))  # 右上
            br = (width, height)  # 右下

            self.vertices = np.array([[bl, tl, tr, br]], dtype=np.int32)

        cv2.fillPoly(mask, self.vertices, 255)
        return cv2.bitwise_and(img, mask)

    def process_frame(self, frame):
        if frame is None: return None

        # 1. 图像预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. 边缘检测 (使用你的参数)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

        # 3. ROI 裁剪
        roi = self.region_of_interest(edges)

        # 4. 霍夫变换 (使用你的参数)
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, HOUGH_THRESH,
                                minLineLength=MIN_LINE_LEN,
                                maxLineGap=MAX_LINE_GAP)

        # 5. 计算平均线
        left_raw, right_raw = self.average_slope_intercept(lines)

        # 6. 平滑处理
        left_avg, right_avg = self.smooth_lines(left_raw, right_raw)

        # 7. 绘制
        height = frame.shape[0]
        y_min = int(height * ROI_HEIGHT_POS) + 50  # 稍微画低一点，不要画到消失点
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

            # 斜率过滤：排除水平线和垂直线
            if abs(slope) < 0.3 or abs(slope) > 10:
                continue

            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

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
        except:
            return None

    def draw_lane(self, img, left_pts, right_pts):
        lane_img = np.zeros_like(img)
        if left_pts is not None and right_pts is not None:
            pts = np.array([left_pts[0], left_pts[1], right_pts[1], right_pts[0]], dtype=np.int32)
            cv2.fillPoly(lane_img, [pts], (0, 255, 0))

        if left_pts: cv2.line(lane_img, left_pts[0], left_pts[1], (255, 0, 0), 10)
        if right_pts: cv2.line(lane_img, right_pts[0], right_pts[1], (0, 0, 255), 10)
        return cv2.addWeighted(img, 1, lane_img, 0.3, 0)


def main():
    # 自动读取 sample.hevc 或摄像头
    input_source = sys.argv[1] if len(sys.argv) > 1 else "sample.hevc"
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print("无法打开视频，请检查路径。")
        return

    detector = LaneDetector()
    print("正在运行... 按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            # 循环播放
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        result = detector.process_frame(frame)
        cv2.imshow('Final Lane Detection', result)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()