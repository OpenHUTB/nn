import cv2
import numpy as np
import os
import time


def traffic_light_real_time_detection():
    """
    摄像头实时交通信号灯三色识别函数
    """
    # 1. 初始化摄像头（0为默认内置摄像头，1为外接摄像头，可根据实际调整）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头，请检查摄像头是否正常连接或被其他程序占用.")
        return "摄像头初始化失败."

    # 设置摄像头分辨率（提升识别效率，可根据摄像头性能调整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 创建保存目录（用于保存抓拍结果）
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_time_detection_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    red_lower1 = np.array([0, 100, 80])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 80])
    red_upper2 = np.array([180, 255, 255])

    # 黄色
    yellow_lower = np.array([15, 120, 80])
    yellow_upper = np.array([35, 255, 255])

    # 绿色
    green_lower = np.array([40, 80, 80])
    green_upper = np.array([80, 255, 255])

    # 形态学操作核
    kernel = np.ones((3, 3), np.uint8)
    # 轮廓检测参数
    contour_params = (cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 自适应最小面积（基于摄像头分辨率）
    min_area = (640 * 480) / 1000

    print("摄像头已启动，实时交通信号灯检测中...")
    print("操作提示：")
    print("  1. 按下 's' 键保存当前检测结果图像")
    print("  2. 按下 'ESC' 键退出程序")

    def filter_valid_contours(contours, min_area_val):
        """筛选有效轮廓（面积+圆度，适配信号灯圆形特征）"""
        valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_val:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity > 0.6:  # 圆度筛选，排除非圆形噪声
                valid.append(cnt)
        return valid

    while True:
        # 3. 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("警告：无法读取摄像头帧，可能是摄像头断开连接.")
            break

        # 复制帧用于绘制结果
        frame_result = frame.copy()
        # 转换为HSV色彩空间（抗光照干扰）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 4. 颜色掩码生成
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        # 5. 形态学操作优化掩码
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 6. 轮廓检测与筛选
        red_contours, _ = cv2.findContours(red_mask, *contour_params)
        yellow_contours, _ = cv2.findContours(yellow_mask, *contour_params)
        green_contours, _ = cv2.findContours(green_mask, *contour_params)

        red_valid = filter_valid_contours(red_contours, min_area)
        yellow_valid = filter_valid_contours(yellow_contours, min_area)
        green_valid = filter_valid_contours(green_contours, min_area)

        # 7. 信号灯判断与可视化绘制
        light_color = "未检测到信号灯"
        color_configs = [
            (red_valid, "红色信号灯", "Red", (0, 0, 255)),
            (yellow_valid, "黄色信号灯", "Yellow", (0, 255, 255)),
            (green_valid, "绿色信号灯", "Green", (0, 255, 0))
        ]

        for valid_contours, color_name, label, bgr_color in color_configs:
            if len(valid_contours) > 0:
                light_color = color_name
                # 绘制轮廓和最小外接圆
                cv2.drawContours(frame_result, valid_contours, -1, bgr_color, 2)
                for cnt in valid_contours:
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(frame_result, center, radius, bgr_color, 2)
                    # 绘制文字标签（避免超出图像边界）
                    text_y = int(y - radius - 10) if (y - radius - 10) > 10 else int(y + radius + 20)
                    cv2.putText(frame_result, label, (int(x) - 20, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)
                break  # 通常仅一种信号灯点亮，找到后退出循环

        # 8. 在帧上添加信息提示
        cv2.putText(frame_result, f"Status: {light_color}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_result, "Press 's' to save, 'ESC' to exit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 9. 显示实时结果
        cv2.namedWindow("Real-Time Traffic Light Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real-Time Traffic Light Detection", 640, 480)
        cv2.imshow("Real-Time Traffic Light Detection", frame_result)

        # 10. 按键事件处理
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键退出
            print("程序已退出")
            break
        elif key == ord('s'):  # 's'键保存结果
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            save_path = os.path.join(save_dir, f"real_time_result_{timestamp}.jpg")
            cv2.imwrite(save_path, frame_result)
            print(f"当前检测结果已保存：{save_path}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    return "实时检测结束"


# 主函数调用
if __name__ == "__main__":
    traffic_light_real_time_detection()