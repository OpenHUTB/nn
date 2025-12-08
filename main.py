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

# -------------------------- 主函数（v1.1优化版） --------------------------
def main():
    # 1. 初始化显示窗口（新增v1.1版本标识）
    win_name = "Lane Line Prediction v1.1 (Blue=Left | Red=Right | Green=Path)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 800, 600)

    # 2. 读取视频（固定路径）
    video_path = "/home/dacun/nn/sample.hevc"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        empty_frame = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.putText(empty_frame, "Cannot open video", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow(win_name, empty_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 读取前10帧（分离显示帧和模型输入帧）
    raw_display_frames = []
    model_input_imgs = []
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        # 缩放为显示尺寸
        display_frame = cv2.resize(frame, (800, 600))
        raw_display_frames.append(display_frame)
        # 转换为模型所需YUV格式
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

    # 3. 加载模型（新增成功提示）
    load_frame = np.ones((600, 800, 3), dtype=np.uint8) * 255
    cv2.putText(load_frame, "Loading model...", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow(win_name, load_frame)
    cv2.waitKey(200)

    model_path = "/home/dacun/桌面/openpilot-modeld-main/models/supercombo.h5"
    try:
        supercombo_model = load_model(model_path, compile=False)
        print("✅ Model loaded successfully (supercombo.h5)")  # 新增提示
    except Exception as e:
        empty_frame = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.putText(empty_frame, "Model load failed", (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow(win_name, empty_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 4. 预处理帧
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

    # 6. 逐帧推理+绘制（优化圆点大小：8→9/6→7）
    print("✅ Start inference and display (Press Q to exit)")
    for i in range(len(frame_tensors) - 1):
        current_frame = raw_display_frames[i].copy() if i < len(raw_display_frames) else np.ones((600, 800, 3), dtype=np.uint8) * 255
        try:
            input_data = [np.vstack(frame_tensors[i:i+2])[None], model_desire, model_state]
            model_output = supercombo_model.predict(input_data, verbose=0)
            parsed_result = parser(model_output)
            model_state = model_output[-1]

            # 车道线坐标映射+右移+圆点优化
            left_lane_x = parsed_result["lll"][0]
            right_lane_x = parsed_result["rll"][0]
            path_x = parsed_result["path"][0]
            win_h, win_w = 600, 800
            y_points = np.linspace(0, win_h - 1, 192).astype(int)
            left_x_mapped = (left_lane_x / 512 * win_w + 100).astype(int)
            right_x_mapped = (right_lane_x / 512 * win_w + 100).astype(int)
            path_x_mapped = (path_x / 512 * win_w + 100).astype(int)

            # 左车道（蓝，9px）、右车道（红，9px）、路径（绿，7px）
            for x, y in zip(left_x_mapped, y_points):
                if 0 <= x < win_w and 0 <= y < win_h:
                    cv2.circle(current_frame, (x, y), 9, (255, 0, 0), -1)
            for x, y in zip(right_x_mapped, y_points):
                if 0 <= x < win_w and 0 <= y < win_h:
                    cv2.circle(current_frame, (x, y), 9, (0, 0, 255), -1)
            for x, y in zip(path_x_mapped, y_points):
                if 0 <= x < win_w and 0 <= y < win_h:
                    cv2.circle(current_frame, (x, y), 7, (0, 255, 0), -1)
        except Exception as e:
            print(f"⚠️ Frame {i+1} inference error: {str(e)[:30]}")

        cv2.imshow(win_name, current_frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            print("🛑 Exit by user (Q pressed)")
            break

    # 7. 程序收尾
    cv2.destroyAllWindows()
    print("🎉 All frames processed successfully!")

if __name__ == "__main__":
    main()
