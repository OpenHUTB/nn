import cv2
import numpy as np
import os

# ============ 可调参数 ============
CANNY_LOW = 50
CANNY_HIGH = 150
GAUSSIAN_KERNEL = 5
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 15
HOUGH_MIN_LINE_LEN = 20
HOUGH_MAX_LINE_GAP = 80
SLOPE_THRESHOLD = 0.5   # 过滤水平噪声线
ROI_TOP_RATIO = 0.6     # ROI顶部占图片高度的比例
WIN_SIZE = (640, 400)


def load_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"找不到文件 {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise IOError(f"无法读取 {img_path}，文件可能损坏或格式不支持")
    return img


def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    return cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)


def region_of_interest(edges, height, width):
    roi_vertices = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.45), int(height * ROI_TOP_RATIO)),
        (int(width * 0.55), int(height * ROI_TOP_RATIO)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    return cv2.bitwise_and(edges, mask)


def detect_lines(roi_img):
    return cv2.HoughLinesP(
        roi_img,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )


def classify_lines(lines, height):
    """按斜率将线段分为左车道线和右车道线"""
    left_lines, right_lines = [], []
    if lines is None:
        return left_lines, right_lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < SLOPE_THRESHOLD:
            continue
        if slope < 0:
            left_lines.append(line[0])
        else:
            right_lines.append(line[0])
    return left_lines, right_lines


def average_line(lines, height):
    """将多条线段拟合为一条平均线，返回两端点"""
    if not lines:
        return None
    xs, ys = [], []
    for x1, y1, x2, y2 in lines:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    poly = np.polyfit(ys, xs, 1)  # x = f(y)
    y_top = int(height * ROI_TOP_RATIO)
    y_bottom = height
    x_top = int(np.polyval(poly, y_top))
    x_bottom = int(np.polyval(poly, y_bottom))
    return (x_top, y_top, x_bottom, y_bottom)


def draw_lanes(img, left_line, right_line):
    """在图像上绘制左右车道线，带半透明叠加层"""
    overlay = img.copy()
    h, w = img.shape[:2]
    lane_pts = []

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 4)
        lane_pts.append((x1, y1))
        lane_pts.append((x2, y2))

    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 4)
        lane_pts.append((x2, y2))
        lane_pts.append((x1, y1))

    # 用半透明多边形填充车道区域
    if len(lane_pts) == 4:
        pts = np.array(lane_pts, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))

    return cv2.addWeighted(overlay, 0.3, img, 0.7, 0)


def main():
    img_path = 'carla_test.jpg'
    img = load_image(img_path)
    h, w = img.shape[:2]
    print(f"图片读取成功，尺寸：{w}x{h}")

    edges = edge_detection(img)
    roi = region_of_interest(edges, h, w)
    lines = detect_lines(roi)

    left_lines, right_lines = classify_lines(lines, h)
    print(f"左车道线段: {len(left_lines)}, 右车道线段: {len(right_lines)}")

    left_lane = average_line(left_lines, h)
    right_lane = average_line(right_lines, h)

    result = draw_lanes(img, left_lane, right_lane)

    # 保存结果图
    result_path = 'result.jpg'
    cv2.imwrite(result_path, result)
    print(f"结果已保存到 {result_path}")

    # 组合四张图用于展示
    def resize(img, size):
        return cv2.resize(img, size)

    gray_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    roi_display = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    top = np.hstack([resize(img, (640, 360)), resize(gray_display, (640, 360))])
    bottom = np.hstack([resize(roi_display, (640, 360)), resize(result, (640, 360))])
    combined = np.vstack([top, bottom])
    cv2.imwrite('combined_result.jpg', combined)
    print("组合结果已保存到 combined_result.jpg")

    # 显示结果
    for name, img_show in [("01-原图", img), ("02-边缘检测", edges),
                            ("03-ROI区域", roi), ("04-车道线检测结果", result)]:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, *WIN_SIZE)
        cv2.imshow(name, img_show)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
