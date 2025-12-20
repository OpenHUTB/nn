# 导入必要库：OpenCV用于图像处理，numpy用于数值计算
import cv2
import numpy as np


class TrafficLightDetector:
    """
    红绿灯检测器类
    功能：识别单张图片或实时摄像头中的红绿灯状态（红/黄/绿）
    核心原理：HSV颜色空间分割 + 形态学处理 + 轮廓检测 + 圆度筛选
    """

    def __init__(self):
        """
        初始化函数：定义红绿灯颜色的HSV阈值范围
        选择HSV空间的原因：相比RGB空间，HSV对光照变化的鲁棒性更强，颜色分割效果更稳定
        每个颜色的阈值范围通过实际场景调试得到，可根据环境光照调整
        """
        # 颜色范围字典：key为颜色名称，value为HSV阈值（H:色相0-180, S:饱和度0-255, V:明度0-255）
        self.color_ranges = {
            'red': [
                (0, 120, 70),  # 红色低阈值（低色相区间：0-10°）
                (10, 255, 255),  # 红色高阈值（低色相区间）
                (170, 120, 70),  # 红色低阈值（高色相区间：170-180°）
                (180, 255, 255)  # 红色高阈值（高色相区间）
                # 红色在HSV色相环中跨0°点，需分两个区间才能完整覆盖
            ],
            'yellow': [
                (20, 120, 70),  # 黄色低阈值（色相20-30°）
                (30, 255, 255)  # 黄色高阈值
            ],
            'green': [
                (35, 120, 70),  # 绿色低阈值（色相35-77°）
                (77, 255, 255)  # 绿色高阈值
            ]
        }

    def preprocess_image(self, frame):
        """
        图像预处理：优化图像质量，为后续颜色分割做准备
        步骤：尺寸缩放 → 高斯模糊 → 颜色空间转换
        Args:
            frame: 输入图像（BGR格式，OpenCV默认读取格式）
        Returns:
            frame: 缩放后的原图像（用于后续绘图显示）
            hsv: 转换后的HSV图像（用于颜色分割）
        """
        # 缩放图像到640x480：减小图像尺寸，加快处理速度，同时保持一定分辨率
        frame = cv2.resize(frame, (640, 480))
        # 高斯模糊：使用5x5卷积核，标准差0，去除图像噪声（减少颜色分割的干扰）
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        # 颜色空间转换：BGR → HSV（OpenCV默认BGR，需转换为HSV进行颜色筛选）
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        return frame, hsv

    def get_color_mask(self, hsv, color):
        """
        根据目标颜色生成掩码（二值图像）：筛选出图像中属于目标颜色的区域
        步骤：生成颜色掩码 → 形态学操作（腐蚀+膨胀）
        Args:
            hsv: HSV格式图像
            color: 目标颜色名称（'red'/'yellow'/'green'）
        Returns:
            mask: 二值掩码图像（白色：目标颜色区域，黑色：非目标颜色区域）
        """
        if color == 'red':
            # 红色特殊处理：合并两个色相区间的掩码
            lower1 = np.array(self.color_ranges[color][0])  # 低色相区间低阈值
            upper1 = np.array(self.color_ranges[color][1])  # 低色相区间高阈值
            lower2 = np.array(self.color_ranges[color][2])  # 高色相区间低阈值
            upper2 = np.array(self.color_ranges[color][3])  # 高色相区间高阈值

            # 生成两个区间的掩码：cv2.inRange()返回二值图（在阈值内为255，否则为0）
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            # 合并两个掩码：使用按位或操作，得到完整的红色区域
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # 黄/绿色处理：单个色相区间
            lower = np.array(self.color_ranges[color][0])
            upper = np.array(self.color_ranges[color][1])
            mask = cv2.inRange(hsv, lower, upper)

        # 形态学操作：优化掩码质量，去除小噪点，强化目标区域
        kernel = np.ones((5, 5), np.uint8)  # 5x5的操作核（uint8：无符号8位整数，符合OpenCV要求）
        mask = cv2.erode(mask, kernel, iterations=1)  # 腐蚀：缩小白色区域，去除细小噪点
        mask = cv2.dilate(mask, kernel, iterations=2)  # 膨胀：扩大白色区域，恢复目标区域大小（腐蚀后补偿）
        return mask

    def get_color_bgr(self, color):
        """
        辅助函数：将颜色名称转换为BGR格式（OpenCV绘图函数要求BGR颜色）
        Args:
            color: 颜色名称（'red'/'yellow'/'green'/'none'）
        Returns:
            BGR格式的颜色值（元组）
        """
        color_map = {
            'red': (0, 0, 255),  # 红色：B=0, G=0, R=255
            'yellow': (0, 255, 255),  # 黄色：B=0, G=255, R=255
            'green': (0, 255, 0),  # 绿色：B=0, G=255, R=0
            'none': (255, 255, 255)  # 无检测结果：白色
        }
        return color_map[color]

    def detect_light(self, frame, hsv):
        """
        核心检测函数：识别红绿灯的状态
        逻辑：遍历三种颜色 → 生成掩码 → 查找轮廓 → 筛选有效灯芯（面积+圆度）→ 确定最大有效区域
        Args:
            frame: 预处理后的原图像（用于绘图）
            hsv: HSV格式图像（用于颜色分割）
        Returns:
            frame: 绘制了检测结果的图像
            detected_color: 检测到的红绿灯颜色（'red'/'yellow'/'green'/'none'）
        """
        max_area = 0  # 记录最大有效区域的面积
        detected_color = "none"  # 初始检测结果：无

        # 遍历三种红绿灯颜色，依次检测
        for color in ['red', 'yellow', 'green']:
            # 1. 获取当前颜色的掩码
            mask = self.get_color_mask(hsv, color)
            # 2. 查找掩码中的轮廓（轮廓：目标区域的边界线条）
            # cv2.RETR_EXTERNAL：只检测最外层轮廓；cv2.CHAIN_APPROX_SIMPLE：压缩轮廓点，减少计算量
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 遍历每个轮廓，筛选有效灯芯
            for cnt in contours:
                # 3. 面积筛选：过滤面积小于100的小轮廓（排除小噪点）
                area = cv2.contourArea(cnt)
                if area > 100:  # 阈值可根据实际场景调整（如远距离红绿灯需减小阈值）
                    # 4. 计算轮廓的最小外接圆（红绿灯灯芯近似为圆形）
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x), int(y))  # 圆心坐标（转换为整数，用于绘图）
                    radius = int(radius)  # 半径（转换为整数）

                    # 5. 圆度筛选：确保轮廓是近似圆形（排除不规则形状的干扰）
                    perimeter = cv2.arcLength(cnt, True)  # 计算轮廓周长（True表示闭合轮廓）
                    if perimeter > 0:  # 避免周长为0导致除零错误
                        # 圆度公式：4πA/P²（A=面积，P=周长），完美圆形的圆度=1.0
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > 0.5:  # 圆度阈值：0.5~1.0，值越大越接近圆形
                            # 6. 确定最大有效区域：面积最大的灯芯即为当前亮灯的颜色
                            if area > max_area:
                                max_area = area
                                detected_color = color
                            # 7. 在图像上绘制检测结果：外接圆 + 颜色标签
                            cv2.circle(frame, center, radius, self.get_color_bgr(color), 2)  # 绘制圆（线宽2）
                            cv2.putText(
                                frame, color, (center[0] - 20, center[1] - 20),  # 文字位置（圆心左上方）
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,  # 字体 + 字号
                                self.get_color_bgr(color), 2  # 文字颜色 + 线宽
                            )

        # 8. 在图像顶部显示最终检测结果（白色文字，线宽2）
        cv2.putText(
            frame, f"Traffic Light: {detected_color.upper()}", (20, 40),  # 文字位置（左上角）
            cv2.FONT_HERSHEY_SIMPLEX, 1,  # 字体 + 字号
            (255, 255, 255), 2  # 白色文字 + 线宽2
        )
        return frame, detected_color

    def detect_image(self, image_path):
        """
        单张图片检测接口：读取图片 → 预处理 → 检测 → 显示结果
        Args:
            image_path: 图片文件路径（相对/绝对路径均可）
        """
        # 读取图片（OpenCV默认读取为BGR格式）
        frame = cv2.imread(image_path)
        if frame is None:  # 图片读取失败（路径错误/文件损坏）
            print(f"错误：无法读取图片 {image_path}，请检查路径是否正确")
            return

        # 预处理 + 核心检测
        frame, hsv = self.preprocess_image(frame)
        result_frame, color = self.detect_light(frame, hsv)

        # 输出检测结果（控制台）
        print(f"检测结果：{color.upper()}")
        # 显示结果图像
        cv2.imshow("Traffic Light Detection (Image)", result_frame)
        cv2.waitKey(0)  # 等待按键输入（0表示无限等待，按任意键关闭）
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口，释放资源

    def detect_video(self, camera_index=0):
        """
        实时摄像头检测接口：打开摄像头 → 逐帧处理 → 实时显示结果
        Args:
            camera_index: 摄像头索引（默认0：电脑内置摄像头；外接摄像头可尝试1、2等）
        """
        # 打开摄像头（VideoCapture对象：用于读取摄像头画面）
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():  # 摄像头打开失败（无权限/设备未连接）
            print("错误：无法打开摄像头，请检查设备连接或权限设置")
            return

        print("实时检测中，按 'q' 退出...")
        # 循环读取摄像头帧（实时处理）
        while True:
            # 读取一帧画面：ret表示读取成功与否，frame为当前帧图像
            ret, frame = cap.read()
            if not ret:  # 帧读取失败（摄像头断开/无画面）
                print("错误：无法读取摄像头画面")
                break

            # 预处理 + 核心检测
            frame, hsv = self.preprocess_image(frame)
            result_frame, color = self.detect_light(frame, hsv)

            # 实时显示检测结果
            cv2.imshow("Traffic Light Detection (Video)", result_frame)

            # 按键检测：按 'q' 退出循环（waitKey(1)：等待1ms，保证实时性）
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放资源：关闭摄像头 + 关闭所有窗口
        cap.release()
        cv2.destroyAllWindows()


# 主程序入口：当脚本直接运行时执行
if __name__ == "__main__":
    # 创建红绿灯检测器实例
    detector = TrafficLightDetector()

    # 选择检测模式（二选一，取消注释对应行即可）
    # 1. 单张图片检测：替换为你的红绿灯图片路径
    # detector.detect_image("traffic_light.jpg")

    # 2. 实时摄像头检测：使用默认内置摄像头（index=0）
    detector.detect_video()