"""步骤8：预处理方法对比与评估。

在同一张图上并排展示三种预处理方法的效果对比，并计算
定量评估指标（有效像素数、检测成功率、拟合 R²），
帮助选择最优预处理策略。

三种方法：
  1. 纯 HSV 颜色阈值（黄 + 白）
  2. 纯 Sobel 梯度阈值
  3. HSV + Sobel 联合（当前 advanced 模式默认）

输出：
  - 2×3 网格对比图（原图 + 3 种二值化 + 3 种检测结果）
  - 控制台打印评估指标
  - 可选保存到 docs/lane_detection/images
"""
import cv2
import numpy as np

from config import CONFIG
from lane_advanced import (
    compute_perspective_matrix, warp_to_birdseye,
    extract_lane_pixels, fit_polynomial, draw_lane_on_original,
    compute_lane_metrics,
)
from lane_warning import compute_warning_level


# ---------------------------------------------------------------------------
# 三种预处理方法
# ---------------------------------------------------------------------------

def preprocess_hsv_only(img):
    """纯 HSV 颜色阈值：黄色 + 白色车道线。"""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 白色
    white_binary = np.zeros_like(gray)
    white_binary[hls[:, :, 1] >= CONFIG["white_thresh"]] = 1

    # 黄色
    yellow_binary = np.zeros_like(gray)
    yellow_binary[
        (hsv[:, :, 0] >= CONFIG["yellow_h_low"]) &
        (hsv[:, :, 0] <= CONFIG["yellow_h_high"]) &
        (hsv[:, :, 1] >= CONFIG["yellow_s_low"])
    ] = 1

    combined = np.zeros_like(gray)
    combined[(white_binary == 1) | (yellow_binary == 1)] = 1
    return combined


def preprocess_sobel_only(img):
    """纯 Sobel 梯度阈值。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-6))
    sxbinary = np.zeros_like(gray)
    sxbinary[(scaled_sobel >= CONFIG["sobel_thresh_min"]) &
             (scaled_sobel <= CONFIG["sobel_thresh_max"])] = 1
    return sxbinary


def preprocess_combined(img):
    """HSV + Sobel 联合（当前 advanced 模式默认）。"""
    from lane_advanced import preprocess_for_advanced
    return preprocess_for_advanced(img)


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------

def compute_r2(fit, x, y):
    """计算多项式拟合的 R² 决定系数。

    Args:
        fit: np.polyfit 系数
        x, y: 实际像素坐标

    Returns:
        R² 值（0~1），None 表示无法计算
    """
    if fit is None or len(x) < 3:
        return None
    y_pred = np.polyval(fit, y)
    ss_res = np.sum((x - y_pred) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    if ss_tot < 1e-9:
        return 1.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def evaluate_method(img, preprocess_fn, method_name):
    """对单一预处理方法运行完整流水线并返回评估指标。

    Returns:
        dict: {
            "name": 方法名称,
            "binary": 二值化图像,
            "result": 检测结果图像,
            "pixels": 有效像素总数,
            "left_pixels": 左车道线像素数,
            "right_pixels": 右车道线像素数,
            "left_r2": 左线 R²,
            "right_r2": 右线 R²,
            "detected": 是否检测到双侧车道线,
            "metrics": 曲率与偏移指标,
            "warning": 预警级别,
        }
    """
    height, width = img.shape[:2]
    M, Minv = compute_perspective_matrix(width, height)

    binary = preprocess_fn(img)
    binary_warped = warp_to_birdseye(binary, M, width, height)

    leftx, lefty, rightx, righty, _ = extract_lane_pixels(binary_warped)
    left_fit, right_fit, left_fitx, right_fitx, ploty, _ = \
        fit_polynomial(binary_warped, leftx, lefty, rightx, righty)

    total_pixels = (len(leftx) + len(rightx))
    left_r2 = compute_r2(left_fit, leftx, lefty) if len(leftx) > 0 else None
    right_r2 = compute_r2(right_fit, rightx, righty) if len(rightx) > 0 else None
    detected = (left_fitx is not None and right_fitx is not None)

    result = None
    metrics = None
    warning = None
    if detected:
        metrics = compute_lane_metrics(left_fit, right_fit, left_fitx, right_fitx, width)
        warning = compute_warning_level(metrics)
        result = draw_lane_on_original(img, binary_warped, Minv,
                                       left_fitx, right_fitx, ploty,
                                       metrics=metrics, warning=warning)

    return {
        "name": method_name,
        "binary": binary,
        "result": result,
        "pixels": total_pixels,
        "left_pixels": len(leftx),
        "right_pixels": len(rightx),
        "left_r2": left_r2,
        "right_r2": right_r2,
        "detected": detected,
        "metrics": metrics,
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# 对比图生成
# ---------------------------------------------------------------------------

def build_comparison_image(img, evaluations, save_path=None):
    """生成 2×3 网格对比图。

    布局：
      第一行：原图 |      原图         |        原图
      第二行：HSV二值化 | Sobel二值化 | 联合二值化
      第三行：HSV检测结果 | Sobel检测结果 | 联合检测结果

    每列标题标注方法名称与评估指标。
    """
    height, width = img.shape[:2]
    h_pad, w_pad = 40, 20  # 内边距
    title_h = 50             # 列标题高度
    label_h = 20             # 指标行高度

    # 按比例缩放以适应屏幕
    scale = min(380 / height, 1.0)
    thumb_h = int(height * scale)
    thumb_w = int(width * scale)

    rows = 3
    cols = 3
    canvas_h = title_h + rows * (thumb_h + h_pad) + h_pad
    canvas_w = cols * (thumb_w + w_pad) + w_pad
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 40

    def place(thumb, row, col):
        """将缩略图放置在 canvas 的指定行列，返回放置区域的左上角坐标。"""
        if thumb is None:
            return (0, 0)
        if len(thumb.shape) == 2:
            thumb = cv2.cvtColor((thumb * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(thumb, (thumb_w, thumb_h))
        y = title_h + row * (thumb_h + h_pad) + h_pad
        x = col * (thumb_w + w_pad) + w_pad
        canvas[y:y + thumb_h, x:x + thumb_w] = resized
        return (x, y)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # 列标题
    names = ["HSV Color", "Sobel Gradient", "HSV + Sobel"]
    colors = [(0, 255, 255), (0, 255, 0), (255, 255, 255)]
    for col, (name, color) in enumerate(zip(names, colors)):
        x = col * (thumb_w + w_pad) + w_pad
        cv2.putText(canvas, name, (x, 30), font, 0.55, color, 2)

    # 第 1 行：原图 × 3
    thumb_img = cv2.resize(img, (thumb_w, thumb_h))
    for col in range(cols):
        place(thumb_img, 0, col)

    # 第 2 行：二值化图像
    for col, ev in enumerate(evaluations):
        place(ev["binary"].astype(np.uint8) * 255, 1, col)

    # 第 3 行：检测结果
    for col, ev in enumerate(evaluations):
        if ev["result"] is not None:
            place(ev["result"], 2, col)
        else:
            # 放置原图 + "DETECTION FAILED" 文字
            x, y = place(thumb_img, 2, col)
            cv2.putText(canvas, "FAILED", (x + 10, y + 30),
                        font, 0.7, (0, 0, 255), 2)

    # 在检测结果行下方叠加指标
    for col, ev in enumerate(evaluations):
        x = col * (thumb_w + w_pad) + w_pad + 5
        y_base = title_h + 2 * (thumb_h + h_pad) + h_pad + thumb_h + 15
        line1 = f"Px: {ev['pixels']} | R2: {ev['left_r2']:.2f}/{ev['right_r2']:.2f}" \
            if ev['left_r2'] is not None and ev['right_r2'] is not None \
            else f"Px: {ev['pixels']} | R2: --/--"
        status = "DETECTED" if ev['detected'] else "FAILED"
        status_color = (0, 255, 0) if ev['detected'] else (0, 0, 255)
        cv2.putText(canvas, line1, (x, y_base), font, 0.35, (200, 200, 200), 1)
        cv2.putText(canvas, status, (x, y_base + 18), font, 0.4, status_color, 1)

    if save_path:
        cv2.imwrite(save_path, canvas)
        print(f"对比图已保存: {save_path}")

    return canvas


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_compare_pipeline(image_path, save_dir=None):
    """运行预处理对比流水线。

    Args:
        image_path: 输入图像路径
        save_dir: 可选，保存对比图的目录

    Returns:
        comparison_image 或 None
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return None

    print(f"===== 预处理方法对比评估 =====")
    print(f"图像: {image_path}")
    print(f"尺寸: {img.shape[1]}x{img.shape[0]}")
    print()

    methods = [
        (preprocess_hsv_only, "HSV 颜色阈值"),
        (preprocess_sobel_only, "Sobel 梯度"),
        (preprocess_combined, "HSV + Sobel 联合"),
    ]

    evaluations = []
    for preprocess_fn, name in methods:
        ev = evaluate_method(img, preprocess_fn, name)
        evaluations.append(ev)
        r2_str = f"R2={ev['left_r2']:.3f}/{ev['right_r2']:.3f}" \
            if ev['left_r2'] is not None and ev['right_r2'] is not None \
            else "R2=--/--"
        status = "检测成功" if ev['detected'] else "检测失败"
        print(f"  {name:16s}  "
              f"像素: {ev['pixels']:5d} (L:{ev['left_pixels']:4d} R:{ev['right_pixels']:4d})  "
              f"{r2_str}  {status}")

    print()

    # 推荐
    print("----- 推荐 -----")
    scores = []
    for ev in evaluations:
        score = ev["pixels"] * 0.4
        if ev["left_r2"] is not None:
            score += ev["left_r2"] * 20
        if ev["right_r2"] is not None:
            score += ev["right_r2"] * 20
        if not ev["detected"]:
            score *= 0.3
        scores.append(score)
    best_idx = np.argmax(scores)
    print(f"最优方法: {evaluations[best_idx]['name']} (得分: {scores[best_idx]:.1f})")

    save_path = None
    if save_dir:
        save_path = f"{save_dir}/step08_compare.jpg"
    comparison = build_comparison_image(img, evaluations, save_path=save_path)
    return comparison