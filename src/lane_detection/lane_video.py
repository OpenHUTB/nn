"""步骤4：视频车道线检测 + 帧间 EMA 平滑。

读取视频文件，逐帧执行步骤3的高级流水线，对连续帧的
多项式拟合系数做指数移动平均（EMA）平滑，减少相邻帧之间的抖动。
输出文件名为 `step04_<原视频名>_output.avi`，避免多个视频互相覆盖。
"""
import os
import cv2
import numpy as np

from lane_advanced import process_frame, draw_lane_on_original, compute_lane_metrics
"""
import os
import cv2
import numpy as np

from lane_advanced import process_frame, draw_lane_on_original, compute_lane_metrics
from lane_warning import compute_warning_level


def smooth_fit(new_fit, prev_fit, alpha):
    """对多项式系数做 EMA 平滑。

    Args:
        new_fit: 当前帧拟合系数 [c, b, a] 或 None
        prev_fit: 上一帧平滑后的系数 或 None
        alpha: 平滑系数，越小越平滑（0 < alpha <= 1）

    Returns:
        平滑后的系数 或 None
    """
    if new_fit is None:
        return prev_fit  # 当前帧检测失败，保持上一帧结果
    if prev_fit is None:
        return new_fit   # 没有历史帧，直接使用当前帧结果
    return alpha * np.array(new_fit) + (1 - alpha) * np.array(prev_fit)


def run_video_pipeline(video_path, save_dir=None, alpha=0.3, show=False):
    """运行视频车道线检测流水线。

    流程：
    1. 逐帧读取视频
    2. 调用 process_frame 执行高级检测
    3. EMA 平滑多项式系数，用平滑后的系数重新绘制
    4. 实时显示 / 保存输出视频

    Args:
        video_path: 输入视频路径
        save_dir: 输出目录（保存处理后的视频）
        alpha: EMA 平滑系数（0~1，越小越平滑，默认 0.3）
        show: 是否弹出实时预览窗口
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

    # 平滑状态
    smooth_left = None
    smooth_right = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 执行单帧检测
        result_img, inter = process_frame(frame)

        # EMA 平滑多项式系数
        smooth_left = smooth_fit(inter["left_fit"], smooth_left, alpha)
        smooth_right = smooth_fit(inter["right_fit"], smooth_right, alpha)

        # 用平滑后的系数重新计算车道线坐标
        ploty = inter["ploty"]
        binary_warped = inter["binary_warped"]
        Minv = inter["Minv"]

        left_fitx = None
        right_fitx = None
        if smooth_left is not None and len(smooth_left) == 3:
            left_fitx = smooth_left[0] * ploty ** 2 + smooth_left[1] * ploty + smooth_left[2]
        if smooth_right is not None and len(smooth_right) == 3:
            right_fitx = smooth_right[0] * ploty ** 2 + smooth_right[1] * ploty + smooth_right[2]

        # 用平滑后的曲线重新绘制
        if left_fitx is not None and right_fitx is not None:
            # 使用平滑后的系数计算曲率与偏移
            metrics = compute_lane_metrics(smooth_left, smooth_right,
                                           left_fitx, right_fitx, width)
            # 计算车道偏离预警级别
            warning = compute_warning_level(metrics)
            display = draw_lane_on_original(frame, binary_warped, Minv,
                                            left_fitx, right_fitx, ploty,
                                            metrics=metrics)
            display = draw_lane_on_original(frame, binary_warped, Minv,
                                            left_fitx, right_fitx, ploty,
                                            metrics=metrics, warning=warning)
        else:
            display = result_img

        # 叠加帧号信息
        cv2.putText(display, f"Frame: {frame_idx}/{total}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"EMA alpha={alpha}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if writer:
            writer.write(display)

        if show:
            cv2.imshow("Lane Detection Video (Step4)", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_idx % 30 == 0:
            print(f"  处理进度: {frame_idx}/{total} ({100 * frame_idx // total}%)")

    cap.release()
    if writer:
        writer.release()
        print(f"输出视频已保存至: {out_path}")
        print(f"输出视频已保存至: {save_dir}/step04_video_output.avi")
    if show:
        cv2.destroyAllWindows()

    print(f"视频处理完成，共 {frame_idx} 帧")
    return True