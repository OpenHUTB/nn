import cv2
import numpy as np
import matplotlib.pyplot as plt


class LaneDetector:
    def __init__(self):
        # 霍夫变换参数
        self.rho = 1
        self.theta = np.pi / 180
        self.threshold = 15
        self.min_line_length = 40
        self.max_line_gap = 20

        # 车道区域顶点（需要根据摄像头位置调整）
        self.vertices = None

    def region_of_interest(self, img):
        """
        定义感兴趣区域（ROI），只处理图像中可能包含车道的部分
        """
        mask = np.zeros_like(img)

        # 如果顶点未定义，使用默认值
        if self.vertices is None:
            height, width = img.shape
            # 定义梯形区域，底部为图像底部，顶部为图像中心附近
            self.vertices = np.array([[
                (width * 0.1, height),
                (width * 0.45, height * 0.6),
                (width * 0.55, height * 0.6),
                (width * 0.9, height)
            ]], dtype=np.int32)

        # 填充多边形区域
        cv2.fillPoly(mask, self.vertices, 255)

        # 应用掩码
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    def detect_edges(self, img):
        """
        使用Canny边缘检测算法检测图像中的边缘
        """
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 高斯模糊以减少噪声
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny边缘检测
        edges = cv2.Canny(blur, 50, 150)

        return edges

    def detect_lines(self, edges):
        """
        使用霍夫变换检测直线
        """
        lines = cv2.HoughLinesP(edges, self.rho, self.theta, self.threshold,
                                np.array([]), minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap)
        return lines

    def average_slope_intercept(self, lines):
        """
        将检测到的线段按左右车道线分组，并计算平均斜率和截距
        """
        left_lines = []  # 左车道线
        right_lines = []  # 右车道线

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # 计算斜率
                if x2 - x1 == 0:  # 避免除以零
                    continue
                slope = (y2 - y1) / (x2 - x1)

                # 过滤掉水平线（斜率接近0）和垂直线（斜率非常大）
                if abs(slope) < 0.5 or abs(slope) > 2:
                    continue

                # 计算截距
                intercept = y1 - slope * x1

                # 根据斜率正负区分左右车道线
                if slope < 0:  # 左车道线斜率为负
                    left_lines.append((slope, intercept))
                else:  # 右车道线斜率为正
                    right_lines.append((slope, intercept))

        # 计算平均斜率和截距
        left_avg = np.average(left_lines, axis=0) if left_lines else None
        right_avg = np.average(right_lines, axis=0) if right_lines else None

        return left_avg, right_avg

    def make_line_points(self, avg_line, y_min, y_max):
        """
        根据斜率和截距生成直线的端点
        """
        if avg_line is None:
            return None

        slope, intercept = avg_line

        # 计算直线与y_min和y_max的交点
        x_min = int((y_min - intercept) / slope)
        x_max = int((y_max - intercept) / slope)

        return [(x_min, y_min), (x_max, y_max)]

    def draw_lane(self, img, left_line, right_line):
        """
        在图像上绘制车道区域
        """
        # 创建空白图像
        lane_img = np.zeros_like(img)

        if left_line is not None and right_line is not None:
            # 提取点坐标
            left_pts = np.array([left_line[0], left_line[1]], dtype=np.int32)
            right_pts = np.array([right_line[0], right_line[1]], dtype=np.int32)

            # 创建车道多边形
            pts = np.vstack([left_pts, np.flipud(right_pts)])

            # 填充车道区域
            cv2.fillPoly(lane_img, [pts], (0, 255, 0))

            # 绘制车道线
            cv2.line(lane_img, left_pts[0], left_pts[1], (255, 0, 0), 5)
            cv2.line(lane_img, right_pts[0], right_pts[1], (255, 0, 0), 5)

        # 将车道图像叠加到原图上
        result = cv2.addWeighted(img, 0.8, lane_img, 0.4, 0)
        return result

    def process_frame(self, frame):
        """
        处理单帧图像，检测车道并绘制结果
        """
        # 检测边缘
        edges = self.detect_edges(frame)

        # 提取感兴趣区域
        roi_edges = self.region_of_interest(edges)

        # 检测直线
        lines = self.detect_lines(roi_edges)

        # 获取图像尺寸
        height, width = frame.shape[:2]

        # 计算平均车道线
        left_avg, right_avg = self.average_slope_intercept(lines)

        # 生成车道线端点
        y_min = int(height * 0.6)  # 车道线上端点
        y_max = height  # 车道线下端点

        left_line = self.make_line_points(left_avg, y_min, y_max)
        right_line = self.make_line_points(right_avg, y_min, y_max)

        # 绘制车道
        result = self.draw_lane(frame, left_line, right_line)

        return result


def main():
    # 创建车道检测器
    detector = LaneDetector()

    # 打开摄像头（0为默认摄像头）
    cap = cv2.VideoCapture(0)

    # 或者打开视频文件
    # cap = cv2.VideoCapture('road_video.mp4')

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        # 读取帧
        ret, frame = cap.read()

        if not ret:
            print("无法读取帧，退出...")
            break

        # 处理帧
        processed_frame = detector.process_frame(frame)

        # 显示结果
        cv2.imshow('Lane Detection', processed_frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()