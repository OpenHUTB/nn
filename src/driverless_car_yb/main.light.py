import cv2
import numpy as np


class TrafficLightDetector:
    def __init__(self):
        # 定义红绿灯颜色范围（HSV 空间，抗光照干扰更强）
        self.color_ranges = {
            'red': [
                (0, 120, 70),  # 红色低阈值（低色相范围）
                (10, 255, 255),  # 红色高阈值（低色相范围）
                (170, 120, 70),  # 红色低阈值（高色相范围，因HSV色相环0和180相邻）
                (180, 255, 255)  # 红色高阈值（高色相范围）
            ],
            'yellow': [
                (20, 120, 70),  # 黄色低阈值
                (30, 255, 255)  # 黄色高阈值
            ],
            'green': [
                (35, 120, 70),  # 绿色低阈值
                (77, 255, 255)  # 绿色高阈值
            ]
        }

    def preprocess_image(self, frame):
        """图像预处理：降噪 + 尺寸缩放"""
        # 缩放图像（加快处理速度）
        frame = cv2.resize(frame, (640, 480))
        # 高斯模糊降噪
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        # 转换为 HSV 颜色空间（更适合颜色分割）
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        return frame, hsv

    def get_color_mask(self, hsv, color):
        """根据颜色获取掩码（筛选目标颜色区域）"""
        if color == 'red':
            # 红色需合并两个范围的掩码
            lower1 = np.array(self.color_ranges[color][0])
            upper1 = np.array(self.color_ranges[color][1])
            lower2 = np.array(self.color_ranges[color][2])
            upper2 = np.array(self.color_ranges[color][3])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower = np.array(self.color_ranges[color][0])
            upper = np.array(self.color_ranges[color][1])
            mask = cv2.inRange(hsv, lower, upper)

        # 形态学操作：去除小噪点 + 强化目标区域
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)  # 腐蚀
        mask = cv2.dilate(mask, kernel, iterations=2)  # 膨胀
        return mask

    def detect_light(self, frame, hsv):
        """检测红绿灯状态"""
        max_area = 0
        detected_color = "none"

        # 遍历三种颜色，寻找最大的有效颜色区域（红绿灯灯芯）
        for color in ['red', 'yellow', 'green']:
            mask = self.get_color_mask(hsv, color)
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                # 计算轮廓面积（过滤小噪点）
                area = cv2.contourArea(cnt)
                if area > 100:  # 最小面积阈值（可根据实际调整）
                    # 计算轮廓的外接圆（红绿灯近似圆形）
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    radius = int(radius)

                    # 圆度筛选（确保是近似圆形的灯芯）
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > 0.5:  # 圆度阈值（0.5~1.0，值越大越圆）
                            if area > max_area:
                                max_area = area
                                detected_color = color
                            # 在图像上绘制灯芯位置
                            cv2.circle(frame, center, radius, self.get_color_bgr(color), 2)
                            cv2.putText(frame, color, (center[0] - 20, center[1] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.get_color_bgr(color), 2)

        # 在图像顶部显示检测结果
        cv2.putText(frame, f"Traffic Light: {detected_color.upper()}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame, detected_color

    def get_color_bgr(self, color):
        """将颜色名称转换为 BGR 格式（OpenCV 绘图用）"""
        color_map = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'none': (255, 255, 255)
        }
        return color_map[color]

    def detect_image(self, image_path):
        """识别单张图片中的红绿灯"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"错误：无法读取图片 {image_path}")
            return

        frame, hsv = self.preprocess_image(frame)
        result_frame, color = self.detect_light(frame, hsv)

        print(f"检测结果：{color.upper()}")
        # 显示结果
        cv2.imshow("Traffic Light Detection (Image)", result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_video(self, camera_index=0):
        """实时摄像头识别红绿灯"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return

        print("实时检测中，按 'q' 退出...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取摄像头画面")
                break

            frame, hsv = self.preprocess_image(frame)
            result_frame, color = self.detect_light(frame, hsv)

            # 显示结果
            cv2.imshow("Traffic Light Detection (Video)", result_frame)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = TrafficLightDetector()

    # 选择识别模式（二选一）
    # 1. 识别单张图片（替换为你的红绿灯图片路径）
    # detector.detect_image("traffic_light.jpg")

    # 2. 实时摄像头识别（默认使用电脑内置摄像头）
    detector.detect_video()