"""步骤3：透视变换 + 滑动窗口 + 二次多项式曲线拟合车道线检测。

基于鸟瞰图视角，通过直方图定位车道线基点，利用滑动窗口搜索车道像素，
使用二次多项式拟合弯道曲线，最后反透视变换回原图绘制。
"""
import cv2
import numpy as np

from config import CONFIG, DEFAULT_IMAGE


# ---- 透视变换 ----

def compute_perspective_matrix(width, height):
    """根据图像尺寸计算透视变换矩阵（原图 → 鸟瞰图）及其逆矩阵。"""
    src = np.float32([
        (width * 0.15, height * 0.65),
        (width * 0.43, height * 0.65),
        (width * 0.90, height * 0.95),
        (width * 0.05, height * 0.95),
    ])
    dst = np.float32([
        (width * 0.20, 0),
        (width * 0.80, 0),
        (width * 0.80, height),
        (width * 0.20, height),
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def warp_to_birdseye(image, M, width, height):
    """将图像变换为鸟瞰图。"""
    return cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)


# ---- 车道线像素提取 ----

def extract_lane_pixels(binary_warped):
    """通过直方图找到左右车道线基点，然后滑动窗口搜索车道像素。

    Returns:
        (left_x, left_y, right_x, right_y, out_img)
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50
    window_height = binary_warped.shape[0] // nwindows

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        if len(good_left) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right]))

    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=np.int64)
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=np.int64)

    leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
    lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
    rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
    righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])

    return leftx, lefty, rightx, righty, out_img


# ---- 多项式拟合 ----

def fit_polynomial(binary_warped, leftx, lefty, rightx, righty):
    """对左右车道线像素进行二次多项式拟合，返回拟合曲线坐标和可视化图像。"""
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    left_fit = None
    right_fit = None
    left_fitx = None
    right_fitx = None

    if len(lefty) > 0 and len(leftx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        out_img[lefty, leftx] = [255, 0, 0]

    if len(righty) > 0 and len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        out_img[righty, rightx] = [0, 0, 255]

    return left_fit, right_fit, left_fitx, right_fitx, ploty, out_img


# ---- 反透视绘制 ----

def draw_lane_on_original(original_img, binary_warped, Minv, left_fitx, right_fitx, ploty):
    """在鸟瞰图上绘制车道区域，再反透视变换叠加回原图。"""
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    if left_fitx is not None and right_fitx is not None:
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)

    if left_fitx is not None:
        for i in range(0, len(ploty) - 1, 5):
            pt1 = (int(left_fitx[i]), int(ploty[i]))
            pt2 = (int(left_fitx[i + 1]), int(ploty[i + 1]))
            cv2.line(color_warp, pt1, pt2, (255, 0, 0), 8)

    if right_fitx is not None:
        for i in range(0, len(ploty) - 1, 5):
            pt1 = (int(right_fitx[i]), int(ploty[i]))
            pt2 = (int(right_fitx[i + 1]), int(ploty[i + 1]))
            cv2.line(color_warp, pt1, pt2, (0, 0, 255), 8)

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
    result = cv2.addWeighted(original_img, 1, newwarp, 0.5, 0)

    return result


# ---- 车道线预处理（HSV + 梯度） ----

def preprocess_for_advanced(img):
    """结合 HSV 颜色阈值与 Sobel 梯度阈值，生成二值化车道线图。"""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Sobel 梯度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= CONFIG["sobel_thresh_min"]) &
             (scaled_sobel <= CONFIG["sobel_thresh_max"])] = 1

    # 白色车道线
    white_lower = np.array([0, CONFIG["white_thresh"], 0])
    white_upper = np.array([255, 255, 255])
    white_binary = np.zeros_like(gray)
    white_binary[
        (hls[:, :, 1] >= white_lower[1]) &
        (hls[:, :, 1] <= white_upper[1])
    ] = 1

    # 黄色车道线
    yellow_lower = np.array([CONFIG["yellow_h_low"], 0, CONFIG["yellow_s_low"]])
    yellow_upper = np.array([CONFIG["yellow_h_high"], 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_binary = np.zeros_like(gray)
    yellow_binary[
        (hsv[:, :, 0] >= yellow_lower[0]) & (hsv[:, :, 0] <= yellow_upper[0]) &
        (hsv[:, :, 1] >= yellow_lower[2]) & (hsv[:, :, 1] <= yellow_upper[2])
    ] = 1

    combined = np.zeros_like(gray)
    combined[(sxbinary == 1) | (white_binary == 1) | (yellow_binary == 1)] = 1
    return combined


# ---- 主流水线 ----

def run_advanced_pipeline(img_path=None, save_dir=None):
    """运行高级车道线检测流水线：透视变换 + 滑动窗口 + 二次多项式拟合。

    流程：
    1. 结合 HSV + Sobel 梯度提取车道线二值图
    2. 透视变换到鸟瞰图
    3. 直方图 + 滑动窗口搜索车道线像素
    4. 二次多项式拟合曲线
    5. 反透视变换叠加回原图
    """
    path = str(img_path or DEFAULT_IMAGE)
    img = cv2.imread(path)
    if img is None:
        print(f"错误：无法读取图片 {path}")
        return None

    height, width = img.shape[:2]
    M, Minv = compute_perspective_matrix(width, height)

    # 车道线二值化
    binary = preprocess_for_advanced(img)
    binary_warped = warp_to_birdseye(binary, M, width, height)

    # 滑动窗口搜索
    leftx, lefty, rightx, righty, sliding_window_img = extract_lane_pixels(binary_warped)

    # 多项式拟合
    left_fit, right_fit, left_fitx, right_fitx, ploty, poly_img = \
        fit_polynomial(binary_warped, leftx, lefty, rightx, righty)

    if left_fitx is None and right_fitx is None:
        print("警告：未能检测到车道线像素")
        return img

    # 反透视绘制
    result = draw_lane_on_original(img, binary_warped, Minv, left_fitx, right_fitx, ploty)

    if save_dir:
        save_dir = str(save_dir)
        cv2.imwrite(f"{save_dir}/step03_binary.jpg", (binary * 255).astype(np.uint8))
        cv2.imwrite(f"{save_dir}/step03_birdseye.jpg", (binary_warped * 255).astype(np.uint8))
        cv2.imwrite(f"{save_dir}/step03_sliding_window.jpg", sliding_window_img.astype(np.uint8))
        cv2.imwrite(f"{save_dir}/step03_poly_fit.jpg", poly_img.astype(np.uint8))
        cv2.imwrite(f"{save_dir}/step03_result.jpg", result)

    return result