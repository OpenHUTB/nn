import cv2
import numpy as np

# ====================== 1. 预处理：提取所有车道线（黄色+白色） ======================
def extract_all_lane_lines(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 提取黄色（双黄线）
    lower_yellow = np.array([10, 70, 70])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 提取白色（车道分隔线）
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 合并黄色和白色掩码
    lane_mask = cv2.bitwise_or(yellow_mask, white_mask)
    
    # 形态学操作，让线条更连续
    kernel = np.ones((3, 3), np.uint8)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
    
    # 边缘检测
    edges = cv2.Canny(lane_mask, 50, 150)
    return edges, yellow_mask, white_mask

# ====================== 2. ROI：只保留路面区域 ======================
def region_of_interest(image):
    height, width = image.shape[:2]
    vertices = np.array([[
        (width * 0.1, height),
        (width * 0.4, height * 0.6),
        (width * 0.6, height * 0.6),
        (width * 0.9, height)
    ]], dtype=np.int32)
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked

# ====================== 3. 霍夫变换检测所有直线 ======================
def detect_all_lines(edges):
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=30,
        maxLineGap=50
    )
    return lines

# ====================== 4. 【核心】识别双黄线作为中心轴 ======================
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
        
        # 双黄线特征：画面中间、接近垂直
        if abs(mid_x - width/2) < width * 0.15 and abs(slope) > 0.5:
            center_candidates.append((x1, y1, x2, y2))
    
    if not center_candidates:
        return None
    
    # 取平均位置作为中心轴
    x1_avg = int(np.mean([l[0] for l in center_candidates]))
    y1_avg = int(np.mean([l[1] for l in center_candidates]))
    x2_avg = int(np.mean([l[2] for l in center_candidates]))
    y2_avg = int(np.mean([l[3] for l in center_candidates]))
    
    return (x1_avg, y1_avg, x2_avg, y2_avg)

# ====================== 5. 【核心】以双黄线为界，划分左右车道线 ======================
def split_lane_lines(lines, center_line, width):
    left_lane_lines = []  # 双黄线左侧：对向（来车）车道线
    right_lane_lines = [] # 双黄线右侧：本向（行驶）车道线
    
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
        
        # 过滤掉水平和过陡的线
        if abs(slope) < 0.3 or abs(slope) > 2:
            continue
        
        # 以双黄线为界，划分左右
        if mid_x < center_mid_x:
            left_lane_lines.append((x1, y1, x2, y2))
        else:
            right_lane_lines.append((x1, y1, x2, y2))
    
    return left_lane_lines, right_lane_lines

# ====================== 6. 拟合车道线并排序 ======================
def fit_and_sort_lanes(lines, height, is_left_side):
    if not lines:
        return []
    
    # 拟合每条车道线
    fitted_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        z = np.polyfit([y1, y2], [x1, x2], 1)
        f = np.poly1d(z)
        
        y_start = height
        y_end = int(height * 0.6)
        x_start = int(f(y_start))
        x_end = int(f(y_end))
        
        fitted_lines.append((x_start, y_start, x_end, y_end))
    
    # 排序：左侧车道从右到左，右侧车道从左到右
    if is_left_side:
        fitted_lines.sort(key=lambda x: (x[0] + x[2]) / 2, reverse=True)
    else:
        fitted_lines.sort(key=lambda x: (x[0] + x[2]) / 2)
    
    return fitted_lines

# ====================== 7. 绘制所有车道（已去掉绿色底色） ======================
def draw_all_lanes(image, left_fitted, right_fitted, center_line):
    lane_mask = np.zeros_like(image)
    height = image.shape[0]
    
    # 绘制双黄线：黑色
    if center_line:
        cx1, cy1, cx2, cy2 = center_line
        cv2.line(lane_mask, (cx1, cy1), (cx2, cy2), (0, 0, 0), 6)
    
    # 绘制左侧（来车）车道线：蓝色
    for line in left_fitted:
        cv2.line(lane_mask, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 4)
    
    # 绘制右侧（行驶）车道线：红色
    for line in right_fitted:
        cv2.line(lane_mask, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 4)
    
    # 融合原图
    result = cv2.addWeighted(image, 0.9, lane_mask, 1.0, 0)
    return result

# ====================== 主程序 ======================
if __name__ == "__main__":
    img = cv2.imread("carla_test.jpg")
    height, width = img.shape[:2]
    
    # 1. 提取所有车道线
    edges, yellow_mask, white_mask = extract_all_lane_lines(img)
    # 2. ROI
    roi_edges = region_of_interest(edges)
    # 3. 检测所有直线
    all_lines = detect_all_lines(roi_edges)
    # 4. 识别双黄线中心轴
    center_line = find_center_double_yellow_line(all_lines, width)
    # 5. 划分左右车道线
    left_lines, right_lines = split_lane_lines(all_lines, center_line, width)
    # 6. 拟合并排序
    left_fitted = fit_and_sort_lanes(left_lines, height, is_left_side=True)
    right_fitted = fit_and_sort_lanes(right_lines, height, is_left_side=False)
    # 7. 绘制所有车道
    final_img = draw_all_lanes(img, left_fitted, right_fitted, center_line)
    
    # 显示结果
    cv2.imshow("Lane Detection (No Background)", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()