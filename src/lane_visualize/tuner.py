import cv2
import numpy as np
import sys


def nothing(x):
    pass


def main():
    # 读取视频
    input_source = "sample.hevc"
    if len(sys.argv) > 1:
        input_source = sys.argv[1]

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"无法打开文件: {input_source}")
        return

    # 创建一个窗口
    cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tuner", 1000, 800)

    # === 创建滑动条 (Sliders) ===
    # 1. Canny 边缘检测阈值 (控制看到的细节多少)
    cv2.createTrackbar("Canny Low", "Tuner", 50, 255, nothing)
    cv2.createTrackbar("Canny High", "Tuner", 150, 255, nothing)

    # 2. ROI 区域 (控制梯形的高度和宽度)
    # 调整梯形顶部的宽度 (Top Width) 和 底部的高度 (Bottom Cut)
    cv2.createTrackbar("ROI Top W", "Tuner", 40, 100, nothing)  # 默认40%
    cv2.createTrackbar("ROI Height", "Tuner", 60, 100, nothing)  # 默认60%位置

    # 3. 霍夫变换 (控制直线的判定标准)
    cv2.createTrackbar("Hough Thresh", "Tuner", 20, 100, nothing)  # 最小投票数
    cv2.createTrackbar("Min Length", "Tuner", 20, 200, nothing)  # 最短线长度
    cv2.createTrackbar("Max Gap", "Tuner", 100, 500, nothing)  # 允许的最大断裂

    print("按 'SPACE' 暂停/继续 | 按 'r' 重播 | 按 'q' 退出")

    is_paused = False

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
                continue

            # 缓存当前帧用于暂停时持续处理
            current_frame = frame.copy()

        # --- 获取滑动条当前值 ---
        canny_low = cv2.getTrackbarPos("Canny Low", "Tuner")
        canny_high = cv2.getTrackbarPos("Canny High", "Tuner")
        roi_top_w = cv2.getTrackbarPos("ROI Top W", "Tuner") / 100.0
        roi_h = cv2.getTrackbarPos("ROI Height", "Tuner") / 100.0
        h_thresh = max(1, cv2.getTrackbarPos("Hough Thresh", "Tuner"))
        min_len = max(1, cv2.getTrackbarPos("Min Length", "Tuner"))
        max_gap = cv2.getTrackbarPos("Max Gap", "Tuner")

        # --- 图像处理管道 ---
        # 1. 灰度与降噪
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. 边缘检测 (实时应用滑动条的值)
        edges = cv2.Canny(blur, canny_low, canny_high)

        # 3. ROI 遮罩 (可视化)
        height, width = edges.shape
        mask = np.zeros_like(edges)

        # 动态计算梯形顶点
        # 梯形：底边铺满，顶边宽度由滑动条控制，高度由滑动条控制
        top_left_x = int(width * (0.5 - roi_top_w / 2))
        top_right_x = int(width * (0.5 + roi_top_w / 2))
        top_y = int(height * roi_h)

        vertices = np.array([[
            (0, height),
            (top_left_x, top_y),
            (top_right_x, top_y),
            (width, height)
        ]], dtype=np.int32)

        cv2.fillPoly(mask, vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # 4. 霍夫变换
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, h_thresh,
                                minLineLength=min_len, maxLineGap=max_gap)

        # --- 绘制结果 ---
        # 创建彩色图用于显示
        display = current_frame.copy()

        # A. 画出ROI范围 (红色细线) - 让你知道机器在看哪里
        cv2.polylines(display, [vertices], True, (0, 0, 255), 2)

        # B. 画出检测到的所有线段 (绿色)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 拼接显示：左边是边缘图(黑白)，右边是结果图(彩色)
        # 将edges转为3通道以便拼接
        edges_bgr = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((edges_bgr, display))

        # 缩小一点以适应屏幕
        scale = 0.6
        h_show, w_show = combined.shape[:2]
        show_img = cv2.resize(combined, (int(w_show * scale), int(h_show * scale)))

        cv2.imshow("Tuner", show_img)

        # 键盘控制
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # 空格键暂停
            is_paused = not is_paused
        elif key == ord('r'):  # 重播
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()