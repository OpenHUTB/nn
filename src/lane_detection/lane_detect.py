"""步骤2：HSV 车道线提取、双黄线中心轴与左右车道划分。"""
import cv2
import numpy as np

from config import DEFAULT_IMAGE


def extract_all_lane_lines(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([10, 70, 70])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    lane_mask = cv2.bitwise_or(yellow_mask, white_mask)
    kernel = np.ones((3, 3), np.uint8)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(lane_mask, 50, 150)
    return edges, yellow_mask, white_mask


def region_of_interest(image):
    height, width = image.shape[:2]
    vertices = np.array(
        [[
            (width * 0.1, height),
            (width * 0.4, height * 0.6),
            (width * 0.6, height * 0.6),
            (width * 0.9, height),
        ]],
        dtype=np.int32,
    )
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)


def detect_all_lines(edges):
    return cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=30,
        maxLineGap=50,
    )


def find_center_double_yellow_line(lines, width):
    center_candidates = []
    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2
        if abs(mid_x - width / 2) < width * 0.15 and abs(slope) > 0.5:
            center_candidates.append((x1, y1, x2, y2))

    if not center_candidates:
        return None

    x1_avg = int(np.mean([line[0] for line in center_candidates]))
    y1_avg = int(np.mean([line[1] for line in center_candidates]))
    x2_avg = int(np.mean([line[2] for line in center_candidates]))
    y2_avg = int(np.mean([line[3] for line in center_candidates]))
    return (x1_avg, y1_avg, x2_avg, y2_avg)


def split_lane_lines(lines, center_line, width):
    left_lane_lines = []
    right_lane_lines = []
    if lines is None or center_line is None:
        return left_lane_lines, right_lane_lines

    cx1, cy1, cx2, cy2 = center_line
    center_mid_x = (cx1 + cx2) / 2

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2
        if abs(slope) < 0.3 or abs(slope) > 2:
            continue
        if mid_x < center_mid_x:
            left_lane_lines.append((x1, y1, x2, y2))
        else:
            right_lane_lines.append((x1, y1, x2, y2))

    return left_lane_lines, right_lane_lines


def fit_and_sort_lanes(lines, height, is_left_side):
    if not lines:
        return []

    fitted_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        z = np.polyfit([y1, y2], [x1, x2], 1)
        f = np.poly1d(z)
        y_start = height
        y_end = int(height * 0.6)
        fitted_lines.append((int(f(y_start)), y_start, int(f(y_end)), y_end))

    if is_left_side:
        fitted_lines.sort(key=lambda x: (x[0] + x[2]) / 2, reverse=True)
    else:
        fitted_lines.sort(key=lambda x: (x[0] + x[2]) / 2)
    return fitted_lines


def draw_all_lanes(image, left_fitted, right_fitted, center_line):
    lane_mask = np.zeros_like(image)

    if center_line:
        cx1, cy1, cx2, cy2 = center_line
        cv2.line(lane_mask, (cx1, cy1), (cx2, cy2), (0, 0, 0), 6)

    for line in left_fitted:
        cv2.line(lane_mask, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 4)

    for line in right_fitted:
        cv2.line(lane_mask, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 4)

    return cv2.addWeighted(image, 0.9, lane_mask, 1.0, 0)


def run_hsv_pipeline(img_path=None, save_dir=None):
    """运行 HSV 多车道检测流水线。"""
    path = str(img_path or DEFAULT_IMAGE)
    img = cv2.imread(path)
    if img is None:
        print(f"错误：无法读取图片 {path}")
        return None

    height, width = img.shape[:2]
    edges, yellow_mask, white_mask = extract_all_lane_lines(img)
    roi_edges = region_of_interest(edges)
    all_lines = detect_all_lines(roi_edges)
    center_line = find_center_double_yellow_line(all_lines, width)
    left_lines, right_lines = split_lane_lines(all_lines, center_line, width)
    left_fitted = fit_and_sort_lanes(left_lines, height, is_left_side=True)
    right_fitted = fit_and_sort_lanes(right_lines, height, is_left_side=False)
    final_img = draw_all_lanes(img, left_fitted, right_fitted, center_line)

    if save_dir:
        save_dir = str(save_dir)
        cv2.imwrite(f"{save_dir}/step02_input.jpg", img)
        cv2.imwrite(f"{save_dir}/step02_yellow_mask.jpg", yellow_mask)
        cv2.imwrite(f"{save_dir}/step02_white_mask.jpg", white_mask)
        cv2.imwrite(f"{save_dir}/step02_result.jpg", final_img)

    return final_img
