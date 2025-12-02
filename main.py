import sys
import os
sys.path.append('/home/dacun/nn/src')

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
from common.tools.lib.parser import parser

# 关闭TensorFlow所有冗余警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------------------------- 核心工具函数 --------------------------
def frames_to_tensor(frames):
    if len(frames) == 0:
        return np.array([])
    H = (frames.shape[1] * 2) // 3
    W = frames.shape[2]
    tensor = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.float32)
    tensor[:, 0] = frames[:, 0:H:2, 0::2]
    tensor[:, 1] = frames[:, 1:H:2, 0::2]
    tensor[:, 2] = frames[:, 0:H:2, 1::2]
    tensor[:, 3] = frames[:, 1:H:2, 1::2]
    tensor[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2, W//2))
    tensor[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2, W//2))
    return tensor / 128.0 - 1.0

def preprocess_frames(imgs):
    if not imgs:
        return np.array([])
    processed = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
    for i, img in enumerate(imgs):
        try:
            processed[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True, output_size=(512, 256))
        except:
            processed[i] = np.zeros((384, 512), dtype=np.uint8)
    return frames_to_tensor(processed)

# -------------------------- 主函数（无参数、全英文、车道线优化） --------------------------
def main():
    # 1. 初始化显示窗口（800x600，固定尺寸）
    win_name = "Lane Line Prediction (Blue=Left | Red=Right | Green=Path)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 800, 600)

    # 2. 读取视频（写死路径，无需传参）
    video_path = "/home/dacun/nn/sample.hevc"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        empty_frame = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.putText(empty_frame, "Cannot open video", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow(win_name, empty_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 读取前10帧（用于推理，保留原始帧和模型输入帧）
    raw_display_frames = []  # 用于显示的800x600帧
    model_input_imgs = []    # 用于模型的512x384 YUV帧
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        # 缩放为显示尺寸（800x600）
        display_frame = cv2.resize(frame, (800, 600))
        raw_display_frames.append(display_frame)
        # 转换为模型需要的YUV格式并缩放
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        model_frame = cv2.resize(yuv_frame, (512, 384), cv2.INTER_AREA)
        model_input_imgs.append(model_frame)
    cap.release()

    # 校验帧数是否足够
    if len(raw_display_frames) < 2:
        empty_frame = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.putText(empty_frame, "Insufficient video frames", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow(win_name, empty_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 3. 加载模型（显示英文提示，无乱码）
    load_frame = np.ones((600, 800, 3), dtype=np.uint8) * 255
    cv2.putText(load_frame, "Loading model...", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow(win_name, load_frame)
    cv2.waitKey(200)  # 刷新显示

    # 模型路径（写死，无需修改）
    model_path = "/home/dacun/桌面/openpilot-modeld-main/models/supercombo.h5"
    try:
        supercombo_model = load_model(model_path, compile=False)
    except Exception as e:
        empty_frame = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.putText(empty_frame, "Model load failed", (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow(win_name, empty_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 4. 预处理帧（显示英文提示）
    preprocess_frame = np.ones((600, 800, 3), dtype=np.uint8) * 255
    cv2.putText(preprocess_frame, "Preprocessing frames...", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow(win_name, preprocess_frame)
    cv2.waitKey(200)

    frame_tensors = preprocess_frames(model_input_imgs)
    if frame_tensors.size == 0:
        empty_frame = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.putText(empty_frame, "Preprocessing failed", (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow(win_name, empty_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 5. 模型状态初始化
    model_state = np.zeros((1, 512))
    model_desire = np.zeros((1, 8))

    # 6. 逐帧推理+绘制（核心优化：车道线右移+放大圆点）
    print("✅ Start inference and display (Press Q to exit)")
    for i in range(len(frame_tensors) - 1):
        # 确保帧存在，避免索引越界
        if i >= len(raw_display_frames):
            current_frame = np.ones((600, 800, 3), dtype=np.uint8) * 255
        else:
            current_frame = raw_display_frames[i].copy()  # 复制原始帧，避免修改

        try:
            # 模型推理（连续两帧作为输入）
            input_data = [np.vstack(frame_tensors[i:i+2])[None], model_desire, model_state]
            model_output = supercombo_model.predict(input_data, verbose=0)
            parsed_result = parser(model_output)
            model_state = model_output[-1]

            # -------------------------- 车道线绘制优化 --------------------------
            # 提取模型输出的车道线/路径x坐标
            left_lane_x = parsed_result["lll"][0]
            right_lane_x = parsed_result["rll"][0]
            path_x = parsed_result["path"][0]
            
            # 窗口尺寸
            win_h, win_w = 600, 800
            # y坐标映射（0-191 → 0-599）
            y_points = np.linspace(0, win_h - 1, 192).astype(int)
            # x坐标映射（0-512 → 0-799）+ 右移100像素（解决偏左问题）+ 放大圆点到8px
            left_x_mapped = (left_lane_x / 512 * win_w + 100).astype(int)
            right_x_mapped = (right_lane_x / 512 * win_w + 100).astype(int)
            path_x_mapped = (path_x / 512 * win_w + 100).astype(int)

            # 绘制左车道线（蓝色，8px实心圆）
            for x, y in zip(left_x_mapped, y_points):
                if 0 <= x < win_w and 0 <= y < win_h:
                    cv2.circle(current_frame, (x, y), 8, (255, 0, 0), -1)
            # 绘制右车道线（红色，8px实心圆）
            for x, y in zip(right_x_mapped, y_points):
                if 0 <= x < win_w and 0 <= y < win_h:
                    cv2.circle(current_frame, (x, y), 8, (0, 0, 255), -1)
            # 绘制预测路径（绿色，6px实心圆）
            for x, y in zip(path_x_mapped, y_points):
                if 0 <= x < win_w and 0 <= y < win_h:
                    cv2.circle(current_frame, (x, y), 6, (0, 255, 0), -1)

        except Exception as e:
            # 推理失败时仅打印错误，仍显示原始帧
            print(f"⚠️ Frame {i+1} inference error: {str(e)[:30]}")

        # 强制显示当前帧
        cv2.imshow(win_name, current_frame)
        # 按Q退出
        if cv2.waitKey(100) & 0xFF == ord('q'):
            print("🛑 Exit by user (Q pressed)")
            break

    # 7. 程序收尾
    cv2.destroyAllWindows()
    print("🎉 All frames processed successfully!")

if __name__ == "__main__":
    main()
