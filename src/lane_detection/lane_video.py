"""步骤4 & 7：视频车道线检测 + 帧间 EMA 平滑 + 快速搜索。

读取视频文件，逐帧执行步骤3的高级流水线，对连续帧的
多项式拟合系数做指数移动平均（EMA）平滑，减少相邻帧之间的抖动。

步骤7 新增快速搜索模式（--fast）：
- 首帧使用滑动窗口搜索
- 后续帧在上一帧曲线 ±margin 范围内搜索（带搜索），大幅减少计算量
- 检测丢失时自动回退到全窗口搜索
- 连续回退超限时暂停快速搜索，重新用滑动窗口校准

输出文件名为 `step04_<原视频名>_output.avi`。
"""
import time
import os
import cv2
import numpy as np

from config import CONFIG
from lane_advanced import (
    process_frame, draw_lane_on_original, compute_lane_metrics,
    extract_lane_pixels, extract_lane_pixels_fast, fit_polynomial,
    preprocess_for_advanced, warp_to_birdseye, compute_perspective_matrix,
)
from lane_advanced import process_frame, draw_lane_on_original, compute_lane_metrics
from lane_warning import compute_warning_level


def smooth_fit(new_fit, prev_fit, alpha):
    """对多项式系数做 EMA 平滑。

    Args:
        new_fit: 当前帧拟合系数 [A, B, C] 或 None
        prev_fit: 上一帧平滑后的系数 或 None
        alpha: 平滑系数，越小越平滑（0 < alpha <= 1）

    Returns:
        平滑后的系数 或 None
    """
    if new_fit is None:
        return prev_fit
    if prev_fit is None:
        return new_fit
    return alpha * np.array(new_fit) + (1 - alpha) * np.array(prev_fit)


def run_video_pipeline(video_path, save_dir=None, alpha=0.3, show=False,
                       use_fast=True):
    """运行视频车道线检测流水线。

    流程：
    1. 逐帧读取视频
    2. 首帧：完整滑动窗口搜索
    3. 后续帧（快速模式）：带搜索 → 失败回退滑动窗口
    4. EMA 平滑多项式系数，用平滑后的系数重新绘制
    5. 实时显示 / 保存输出视频

    Args:
        video_path: 输入视频路径
        save_dir: 输出目录
        alpha: EMA 平滑系数（0~1，越小越平滑，默认 0.3）
        show: 是否弹出实时预览窗口
        use_fast: 是否启用快速搜索（默认 True）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height}, {fps:.1f}fps, {total}帧")
    print(f"EMA 平滑系数: alpha={alpha}")
    print(f"快速搜索: {'启用 (margin=' + str(CONFIG['fast_search_margin']) + ')' if use_fast else '关闭 (滑动窗口)'}")

    writer = None
    out_path = None
    if save_dir:
        save_dir = str(save_dir)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = f"{save_dir}/step04_{base_name}_output.avi"
    if save_dir:
        save_dir = str(save_dir)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = f"{save_dir}/step04_{base_name}_output.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # 状态
    smooth_left = None
    smooth_right = None
    prev_left_fit = None
    prev_right_fit = None
    frame_idx = 0
    consecutive_fallback = 0
    max_consecutive_fallback = CONFIG["fast_max_consecutive_fallback"]
    fast_margin = CONFIG["fast_search_margin"]

    # 计时
    t_start = time.time()
    t_sw_count = 0          # 滑动窗口次数
    t_fast_count = 0        # 带搜索次数
    t_fallback_count = 0    # 回退次数

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 预处理 + 透视变换（两种模式共用）
        binary = preprocess_for_advanced(frame)
        M, Minv = compute_perspective_matrix(width, height)
        binary_warped = warp_to_birdseye(binary, M, width, height)

        left_fit = None
        right_fit = None
        left_fitx = None
        right_fitx = None

        # ---- 快速搜索模式 ----
        if use_fast and prev_left_fit is not None and prev_right_fit is not None \
                and consecutive_fallback < max_consecutive_fallback:
            fast_result = extract_lane_pixels_fast(
                binary_warped, prev_left_fit, prev_right_fit, margin=fast_margin)
            t_fast_count += 1

            if fast_result is not None:
                # 快速搜索成功
                leftx, lefty, rightx, righty, _ = fast_result
                left_fit, right_fit, left_fitx, right_fitx, ploty, _ = \
                    fit_polynomial(binary_warped, leftx, lefty, rightx, righty)
                consecutive_fallback = 0
            else:
                # 快速搜索失败，回退到滑动窗口
                t_fallback_count += 1
                consecutive_fallback += 1
                leftx, lefty, rightx, righty, _ = extract_lane_pixels(binary_warped)
                left_fit, right_fit, left_fitx, right_fitx, ploty, _ = \
                    fit_polynomial(binary_warped, leftx, lefty, rightx, righty)
                t_sw_count += 1
        else:
            # 首帧或连续回退太多次，使用滑动窗口
            leftx, lefty, rightx, righty, _ = extract_lane_pixels(binary_warped)
            left_fit, right_fit, left_fitx, right_fitx, ploty, _ = \
                fit_polynomial(binary_warped, leftx, lefty, rightx, righty)
            t_sw_count += 1

            if consecutive_fallback >= max_consecutive_fallback:
                consecutive_fallback = 0  # 重新校准，重置计数器

        # 保存当前帧拟合系数供下一帧使用
        prev_left_fit = left_fit
        prev_right_fit = right_fit

        # EMA 平滑
        smooth_left = smooth_fit(left_fit, smooth_left, alpha)
        smooth_right = smooth_fit(right_fit, smooth_right, alpha)

        # 用平滑后的系数重新计算坐标
        if smooth_left is not None and len(smooth_left) == 3:
            left_fitx = smooth_left[0] * ploty ** 2 + smooth_left[1] * ploty + smooth_left[2]
        else:
            left_fitx = None
        if smooth_right is not None and len(smooth_right) == 3:
            right_fitx = smooth_right[0] * ploty ** 2 + smooth_right[1] * ploty + smooth_right[2]
        else:
            right_fitx = None

        # 绘制
        if left_fitx is not None and right_fitx is not None:
            metrics = compute_lane_metrics(smooth_left, smooth_right,
                                           left_fitx, right_fitx, width)
            warning = compute_warning_level(metrics)
            display = draw_lane_on_original(frame, binary_warped, Minv,
                                            left_fitx, right_fitx, ploty,
            # 计算车道偏离预警级别
            warning = compute_warning_level(metrics)
            display = draw_lane_on_original(frame, binary_warped, Minv,
                                            left_fitx, right_fitx, ploty,
                                            metrics=metrics)
            display = draw_lane_on_original(frame, binary_warped, Minv,
                                            left_fitx, right_fitx, ploty,
                                            metrics=metrics, warning=warning)
        else:
            display = frame

        # 叠加帧号和搜索模式
        cv2.putText(display, f"Frame: {frame_idx}/{total}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"EMA alpha={alpha}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 快速搜索状态指示
        if use_fast:
            if consecutive_fallback > 0:
                sw_text = f"SW (fallback x{consecutive_fallback})"
                sw_color = (0, 200, 255)
            elif frame_idx <= 1:
                sw_text = "SW (init)"
                sw_color = (0, 200, 255)
            else:
                sw_text = "BAND (fast)"
                sw_color = (0, 255, 0)
            cv2.putText(display, sw_text, (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, sw_color, 1)

        if writer:
            writer.write(display)

        if show:
            cv2.imshow("Lane Detection Video (Step4)", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_idx % 30 == 0:
            pct = 100 * frame_idx // total
            print(f"  处理进度: {frame_idx}/{total} ({pct}%)")

    cap.release()
    if writer:
        writer.release()
        print(f"输出视频已保存至: {out_path}")
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"视频处理完成，共 {frame_idx} 帧")
    print(f"耗时: {elapsed:.1f}s, 平均: {elapsed / max(frame_idx, 1) * 1000:.1f}ms/帧")
    if use_fast:
        pct_fast = 100 * t_fast_count / max(frame_idx, 1)
        print(f"带搜索: {t_fast_count}次 ({pct_fast:.0f}%), "
              f"滑动窗口: {t_sw_count}次, 回退: {t_fallback_count}次")
    return True