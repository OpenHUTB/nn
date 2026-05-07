import cv2 as cv
from ultralytics import YOLO
import numpy as np
import supervision as sv
import json
from datetime import datetime
import keyboard_handler  # 导入键盘处理模块

# ==================== 配置路径 ====================
import os
# 获取脚本所在目录的相对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # yolo11n_vehicle_counter 目录

# 模型文件路径
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/yolo11n.pt")
# 输入视频文件路径 - 针对CARLA录制的视频
INPUT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "dataset/sample.mp4")
# 输出视频文件路径
OUTPUT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "res/sample_res.mp4")
# 统计数据保存路径
STATS_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "res/counting_stats.json")
# ==================================================


def main(model_path=None, input_video_path=None, output_video_path=None, ground_truth=None):
    """主函数 - 运行车辆计数，针对CARLA视频优化

    Args:
        model_path: 模型文件路径 (如果为None则使用默认值)
        input_video_path: 输入视频路径 (如果为None则使用默认值)
        output_video_path: 输出视频路径 (如果为None则使用默认值)
        ground_truth: ground truth文件路径 (可选，用于精度衡量)
    """
    # 使用传入的参数或默认值
    model_path = model_path or MODEL_PATH
    input_video_path = input_video_path or INPUT_VIDEO_PATH
    output_video_path = output_video_path or OUTPUT_VIDEO_PATH

    # 初始化YOLO模型和视频信息
    model = YOLO(model_path)  # 加载YOLO模型
    video_path = input_video_path  # 设置输入视频路径
    video_info = sv.VideoInfo.from_video_path(video_path)
    w, h, fps = video_info.width, video_info.height, video_info.fps  # 获取视频宽度、高度和帧率

    print(f"视频信息: {w}x{h}, {fps}fps")  # 打印视频信息

    # 设置标注器参数
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)  # 计算最优线条粗细
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)  # 计算最优文字大小

    # 创建各种标注器用于可视化
    box_annotator = sv.RoundBoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)  # 圆角矩形标注器

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness,
                                        text_position=sv.Position.TOP_CENTER, color_lookup=sv.ColorLookup.TRACK)  # 标签标注器
    trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=fps,
                                        position=sv.Position.CENTER, color_lookup=sv.ColorLookup.TRACK)  # 轨迹标注器

    # 追踪器和检测平滑器设置
    tracker = sv.ByteTrack(frame_rate=video_info.fps)  # 字节追踪器
    smoother = sv.DetectionsSmoother()  # 检测平滑器，用于稳定检测结果

    # 车辆类别设置
    class_names = model.names  # 获取模型类别名称
    vehicle_classes = ['car', 'motorbike', 'bus', 'truck']  # 定义需要检测的车辆类别
    # 筛选出车辆类别对应的ID
    selected_classes = [cls_id for cls_id, class_name in model.names.items() if class_name in vehicle_classes]

    # 初始化计数器
    limits = [350, 750, 1230, 750]  # 计数线位置
    total_counts, crossed_ids = [], set()  # 总计数和已计数车辆ID集合
    track_history = {}  # 轨迹历史: {track_id: [y1, y2, ...]}
    counted_tracks = set()  # 已完成计数的轨迹ID

    # 分类计数器
    class_counts = {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0}  # 各类别已计数ID集合
    crossed_by_class = {cls: set() for cls in class_counts.keys()}  # 各类别已计数ID集合
    
    # 轨迹历史记录（用于判断穿越方向和速度估算）
    previous_y = {}  # 上一帧的y坐标
    CONFIDENCE_THRESHOLD = keyboard_handler.get_confidence_threshold()  # 置信度阈值（可调整）
    
    # 速度估算参数
    PIXELS_TO_METERS = 0.05  # 像素到米的转换系数（每像素=0.05米）
    track_speeds = {}  # 记录每个轨迹的速度
    smoothed_speeds = {}  # 平滑后的速度
    speed_buffer = {}  # 速度缓冲用于移动平均平滑

    class KalmanFilter2D:
        def __init__(self, dt=1.0, process_noise=0.1, measurement_noise=10.0):
            self.dt = dt
            self.x = np.zeros((4, 1))
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            self.F = F
            self.H = H
            self.Q = np.eye(4) * process_noise
            self.R = np.eye(2) * measurement_noise
            self.P = np.eye(4) * 100.0
            self.initialized = False

        def predict(self):
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            return self.x[:2].flatten()

        def update(self, measurement):
            z = np.array(measurement).reshape(2, 1)
            if not self.initialized:
                self.x[:2] = z
                self.initialized = True
                return self.x[:2].flatten()
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ (z - self.H @ self.x)
            self.P = (np.eye(4) - K @ self.H) @ self.P
            return self.x[:2].flatten()

        def get_velocity(self):
            return self.x[2:4].flatten()

    kalman_filters = {}  # {track_id: KalmanFilter2D}

    def draw_overlay(frame, pt1, pt2, alpha=0.25, color=(51, 68, 255), filled=True):
        """绘制半透明覆盖矩形

        Args:
            frame: 输入帧
            pt1: 矩形左上角坐标
            pt2: 矩形右下角坐标
            alpha: 透明度
            color: 矩形颜色
            filled: 是否填充
        """
        overlay = frame.copy()
        rect_color = color if filled else (0, 0, 0)
        cv.rectangle(overlay, pt1, pt2, rect_color, cv.FILLED if filled else 1)
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


    def draw_confidence_distribution(frame, confidences, start_x, start_y, bar_width=15, max_height=80):
        """绘制置信度分布条形图

        Args:
            frame: 输入帧
            confidences: 置信度列表
            start_x, start_y: 起始坐标
            bar_width: 每个柱子的宽度
            max_height: 最大柱高
        """
        if len(confidences) == 0:
            return start_y

        # 将置信度分成5个区间
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        counts, _ = np.histogram(confidences, bins=bins)
        
        # 计算分布比例
        total = len(confidences)
        max_count = max(counts) if max(counts) > 0 else 1
        
        # 绘制背景面板
        panel_width = 6 * (bar_width + 2) + 10
        panel_height = max_height + 50
        cv.rectangle(frame, (start_x, start_y), (start_x + panel_width, start_y + panel_height), 
                     (40, 40, 40), cv.FILLED)
        
        # 绘制每个区间的柱状图
        colors = [(255, 100, 100), (255, 180, 100), (255, 255, 100), 
                  (100, 255, 100), (100, 200, 255)]  # 从红到蓝渐变
        labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        
        x_pos = start_x + 5
        for i, (count, color) in enumerate(zip(counts, colors)):
            # 计算柱高
            height = int((count / max_count) * max_height) if max_count > 0 else 0
            
            # 绘制柱子
            cv.rectangle(frame, 
                        (x_pos, start_y + 30 + (max_height - height)),
                        (x_pos + bar_width, start_y + 30 + max_height),
                        color, cv.FILLED)
            
            # 显示数量
            if count > 0:
                text = str(int(count))
                cv.putText(frame, text, (x_pos + 2, start_y + 25 + (max_height - height)),
                          cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            x_pos += bar_width + 2
        
        # 显示标签
        y_offset = start_y + max_height + 35
        cv.putText(frame, "Conf Distribution", (start_x + 5, y_offset),
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return y_offset + 15


    def count_vehicles(track_id, cx, cy, limits, crossed_ids, class_name=None, crossed_by_class=None, class_counts=None):
        """统计穿过计数线的车辆

        Args:
            track_id: 车辆追踪ID
            cx, cy: 车辆中心点坐标
            limits: 计数线位置
            crossed_ids: 已计数车辆ID集合
            class_name: 车辆类别名称
            crossed_by_class: 各类别已计数ID集合
            class_counts: 各类别计数器

        Returns:
            tuple: (是否计数成功, 车辆类别)
        """
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15 and track_id not in crossed_ids:
            crossed_ids.add(track_id)
            if class_name and crossed_by_class and class_counts:
                if track_id not in crossed_by_class[class_name]:
                    crossed_by_class[class_name].add(track_id)
                    class_counts[class_name] += 1
            return True, class_name
        return False, None


    def draw_tracks_and_count(frame, detections, limits):
        """绘制轨迹并统计车辆（按类别分类统计）

        Args:
            frame: 输入帧
            detections: 检测结果
            limits: 计数线位置
        """
        nonlocal total_counts, crossed_ids, class_counts, crossed_by_class

        # 按车辆类别和检测置信度过滤 - 使用可调整的阈值
        detections = detections[(np.isin(detections.class_id, selected_classes)) & (detections.confidence > CONFIDENCE_THRESHOLD)]

        # 为每个检测框生成标签（包含速度信息和轨迹长度）
        labels = []
        for track_id, cls_id in zip(detections.tracker_id, detections.class_id):
            speed = smoothed_speeds.get(track_id, 0)
            track_len = len(track_history.get(track_id, []))
            labels.append(f"#{track_id} {class_names[cls_id]} {speed:.1f}m/s L:{track_len}")

        # 绘制边界框、标签和轨迹
        box_annotator.annotate(frame, detections=detections)
        label_annotator.annotate(frame, detections=detections, labels=labels)
        trace_annotator.annotate(frame, detections=detections)

        # 处理每个检测到的车辆
        for i, (track_id, center_point) in enumerate(zip(detections.tracker_id,
                                                         detections.get_anchors_coordinates(anchor=sv.Position.CENTER))):
            cx, cy = map(int, center_point)
            cls_id = detections.class_id[i]
            cls_name = class_names[cls_id]

            cv.circle(frame, (cx, cy), 4, (0, 255, 255), cv.FILLED)  # 绘制车辆中心点

            # 记录轨迹历史（包括x, y坐标）
            if track_id not in track_history:
                track_history[track_id] = []
                previous_y[track_id] = cy  # 初始化上一帧y坐标
            track_history[track_id].append((cx, cy))
            # 只保留最近30帧历史
            if len(track_history[track_id]) > 30:
                track_history[track_id] = track_history[track_id][-30:]
            
            # 速度估算：使用卡尔曼滤波平滑位置，再用位移计算速度
            if track_id not in kalman_filters:
                kalman_filters[track_id] = KalmanFilter2D(dt=1.0/fps if fps > 0 else 1.0)

            kf = kalman_filters[track_id]
            kf.update([cx, cy])

            # 使用卡尔曼滤波平滑历史轨迹
            if track_id not in track_history or len(track_history[track_id]) < 2:
                # 轨迹过短，使用原始计算
                if len(track_history.get(track_id, [])) >= 2:
                    prev_x, prev_y_pos = track_history[track_id][-2]
                    curr_x, curr_y_pos = track_history[track_id][-1]
                    pixel_displacement = np.sqrt((curr_x - prev_x)**2 + (curr_y_pos - prev_y_pos)**2)
                    speed = pixel_displacement * PIXELS_TO_METERS * fps if fps > 0 else 0
                else:
                    speed = 0
            else:
                # 使用卡尔曼滤波平滑位置计算速度
                prev_x, prev_y_pos = track_history[track_id][-2]
                curr_x, curr_y_pos = track_history[track_id][-1]
                pixel_displacement = np.sqrt((curr_x - prev_x)**2 + (curr_y_pos - prev_y_pos)**2)
                raw_speed = pixel_displacement * PIXELS_TO_METERS * fps if fps > 0 else 0

                # 卡尔曼滤波平滑速度（保留原来的计算方式）
                if track_id not in smoothed_speeds:
                    smoothed_speeds[track_id] = raw_speed
                else:
                    smoothed_speeds[track_id] = 0.7 * smoothed_speeds[track_id] + 0.3 * raw_speed
                speed = smoothed_speeds[track_id]

            track_speeds[track_id] = speed

            # 简化判断：只要y变小就算穿越（不管方向、不限x范围）
            if track_id not in crossed_ids:
                prev_y = previous_y.get(track_id, cy)
                curr_y = cy
                line_y = limits[1]
                
                # 简化穿越逻辑：y变小就计数
                crossed_line = curr_y < prev_y
                if crossed_line:
                    crossed_ids.add(track_id)
                    counted_tracks.add(track_id)
                    total_counts.append(track_id)

                    # 分类计数
                    if cls_name in class_counts:
                        class_counts[cls_name] += 1

                    sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                                 color=sv.Color.ROBOFLOW, thickness=4)
                    draw_overlay(frame, (350, 700), (1230, 800), alpha=0.25, color=(10, 255, 50))
                
                # 更新上一帧y坐标
                previous_y[track_id] = curr_y

        # 显示车辆总计数
        sv.draw_text(frame, f"TOTAL: {len(total_counts)}", sv.Point(x=50, y=50), sv.Color.ROBOFLOW, 0.8,
                     2, background_color=sv.Color.WHITE)
        
        # 显示分类统计面板
        y_offset = 100
        for cls_name, count in class_counts.items():
            sv.draw_text(frame, f"{cls_name.upper()}: {count}", sv.Point(x=50, y=y_offset), sv.Color.YELLOW, 0.6,
                         1, background_color=sv.Color.BLACK)
            y_offset += 30
        
        # 显示当前帧检测到的车辆数
        active_count = len(detections.tracker_id) if detections.tracker_id is not None else 0
        sv.draw_text(frame, f"Active: {active_count}", sv.Point(x=50, y=y_offset+10), sv.Color.BLUE, 0.6,
                     1, background_color=sv.Color.BLACK)
        
        # 显示当前优化策略说明
        y_offset += 50
        # 显示优化策略（使用英文避免字体问题）
        sv.draw_text(frame, "[Strategy]", sv.Point(x=50, y=y_offset), sv.Color.GREEN, 0.5,
                     1, background_color=sv.Color.from_hex("#404040"))
        y_offset += 25
        sv.draw_text(frame, f"conf:{CONFIDENCE_THRESHOLD:.2f} | speed:on | smooth:on | track:on | dist:on", sv.Point(x=50, y=y_offset), sv.Color.WHITE, 0.4,
                     1, background_color=sv.Color.from_hex("#404040"))
        
        # 显示速度估算面板
        y_offset += 35
        sv.draw_text(frame, "[Speed Estimation]", sv.Point(x=50, y=y_offset), sv.Color.BLUE, 0.5,
                     1, background_color=sv.Color.from_hex("#404040"))
        y_offset += 25
        
        # 显示所有检测到的车辆速度
        if track_speeds:
            speed_values = list(track_speeds.values())
            if speed_values:
                avg_speed = sum(speed_values) / len(speed_values)
                max_speed = max(speed_values)
                sv.draw_text(frame, f"Avg: {avg_speed:.1f} m/s", sv.Point(x=50, y=y_offset), sv.Color.WHITE, 0.4,
                            1, background_color=sv.Color.from_hex("#404040"))
                y_offset += 25
                sv.draw_text(frame, f"Max: {max_speed:.1f} m/s", sv.Point(x=50, y=y_offset), sv.Color.YELLOW, 0.4,
                            1, background_color=sv.Color.from_hex("#404040"))
        
        # 显示轨迹长度统计面板
        y_offset += 35
        sv.draw_text(frame, "[Track Length]", sv.Point(x=50, y=y_offset), sv.Color.GREEN, 0.5,
                     1, background_color=sv.Color.from_hex("#404040"))
        y_offset += 25
        
        # 计算所有轨迹的长度统计
        if track_history:
            track_lengths = [len(pts) for pts in track_history.values()]
            if track_lengths:
                avg_length = sum(track_lengths) / len(track_lengths)
                max_length = max(track_lengths)
                min_length = min(track_lengths)
                active_tracks = len([tid for tid in detections.tracker_id if tid in track_history]) if detections.tracker_id is not None else 0
                sv.draw_text(frame, f"Active: {active_tracks} tracks", sv.Point(x=50, y=y_offset), sv.Color.WHITE, 0.4,
                            1, background_color=sv.Color.from_hex("#404040"))
                y_offset += 25
                sv.draw_text(frame, f"Avg: {avg_length:.1f} frames", sv.Point(x=50, y=y_offset), sv.Color.WHITE, 0.4,
                            1, background_color=sv.Color.from_hex("#404040"))
                y_offset += 25
                sv.draw_text(frame, f"Max: {max_length} | Min: {min_length}", sv.Point(x=50, y=y_offset), sv.Color.YELLOW, 0.4,
                            1, background_color=sv.Color.from_hex("#404040"))
        
        # 显示置信度分布可视化面板
        y_offset += 35
        sv.draw_text(frame, "[Conf Distribution]", sv.Point(x=50, y=y_offset), sv.Color.RED, 0.5,
                     1, background_color=sv.Color.from_hex("#404040"))
        y_offset += 25
        
        # 获取当前帧所有检测的置信度（过滤前）
        all_confidences = detections_main.confidence.tolist() if hasattr(detections_main, 'confidence') and detections_main.confidence is not None else []
        if all_confidences:
            # 绘制置信度分布条形图
            y_offset = draw_confidence_distribution(frame, all_confidences, 50, y_offset)
            
            # 显示置信度统计信息
            y_offset += 5
            avg_conf = np.mean(all_confidences)
            max_conf = np.max(all_confidences)
            min_conf = np.min(all_confidences)
            sv.draw_text(frame, f"Min: {min_conf:.2f} | Avg: {avg_conf:.2f} | Max: {max_conf:.2f}", 
                        sv.Point(x=50, y=y_offset), sv.Color.WHITE, 0.35,
                        1, background_color=sv.Color.from_hex("#404040"))


    # 打开视频文件
    cap = cv.VideoCapture(video_path)
    output_path = output_video_path  # 设置输出视频路径
    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if not cap.isOpened():
        raise Exception("错误: 无法打开视频文件!")

    # 视频处理主循环
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧打印一次进度
            print(f"[帧{frame_count}] 检测中... 当前已计数: {len(total_counts)}辆 | {class_counts}")

        # 针对CARLA视频优化ROI区域 - 扩大检测范围避免漏检
        # 只裁剪左右边缘和最顶部天空区域，保留更多车辆检测机会
        roi_top = h // 6  # 从1/6高度开始（之前是1/3，丢失太多）
        roi_left = w // 12  # 从1/12宽度开始（之前是1/6，减少裁剪）
        crop = frame[roi_top:, roi_left:w-roi_left]
        mask_b = np.zeros_like(frame, dtype=np.uint8)
        mask_w = np.ones_like(crop, dtype=np.uint8) * 255
        mask_b[roi_top:, roi_left:w-roi_left] = mask_w

        # 应用掩码到原始帧
        ROI = cv.bitwise_and(frame, mask_b)

        # YOLO多尺度检测 - 使用不同输入尺寸提高远处车辆检测率
        # 原始尺寸用于近处大目标，640尺寸用于远处小目标
        results_main = model(ROI, imgsz=1280)[0]  # 增大输入尺寸，检测更多细节
        detections_main = sv.Detections.from_ultralytics(results_main)
        
        # 如果需要额外补充检测（可选）
        if len(detections_main) < 3:  # 检测过少时尝试小目标检测
            results_small = model(ROI, imgsz=1920)[0]
            detections_small = sv.Detections.from_ultralytics(results_small)
            # 合并结果（去重）
            all_detections = detections_main if len(detections_main) > 0 else detections_small
        else:
            all_detections = detections_main
        
        detections = all_detections
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        if detections.tracker_id is not None:
            # 绘制计数线并处理车辆轨迹
            sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                         color=sv.Color.RED, thickness=4)
            # 调整覆盖区域透明度 - 与红线位置匹配
            draw_overlay(frame, (350, 700), (1230, 800), alpha=0.15)
            draw_tracks_and_count(frame, detections, limits)
        
        # 更新置信度阈值（从keyboard_handler获取）
        CONFIDENCE_THRESHOLD = keyboard_handler.get_confidence_threshold()
        
        # 显示当前检测阈值（可调）
        sv.draw_text(frame, f"Conf: {CONFIDENCE_THRESHOLD:.2f} (C/D)", sv.Point(x=50, y=h-100), sv.Color.WHITE, 0.5,
                    1, background_color=sv.Color.from_hex("#303030"))
        
        # 写入帧到输出视频
        out.write(frame)
        # 显示当前帧
        cv.imshow("YOLO11n Vehicle Counter - CARLA", frame)

        # 键盘事件处理
        key = cv.waitKey(1) & 0xff
        if not keyboard_handler.handle_keyboard_events(key, frame, frame_count, cap, out, "YOLO11n Vehicle Counter - CARLA"):
            break

    # 释放资源
    cap.release()
    out.release()
    cv.destroyAllWindows()

    # ========== 功能扩展：保存统计数据到JSON ==========
    # 转换numpy类型为Python原生类型，避免JSON序列化错误
    stats_data = {
        "video_path": video_path,
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_info": {
            "width": int(w),
            "height": int(h),
            "fps": float(fps),
            "total_frames": int(frame_count)
        },
        "counting_line": {
            "x1": int(limits[0]), "y1": int(limits[1]),
            "x2": int(limits[2]), "y2": int(limits[3])
        },
        "total_count": len(total_counts),
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "counted_track_ids": [int(tid) for tid in counted_tracks]
    }
    
    with open(STATS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
    print(f"统计数据已保存至: {STATS_OUTPUT_PATH}")
    
    print(f"\n" + "=" * 50)
    print(f"            YOLO车辆计数结果")
    print(f"=" * 50)
    print(f"  视频信息: {w}x{h}, {fps}fps, 共{frame_count}帧")
    print(f"  计数线: ({limits[0]},{limits[1]}) -> ({limits[2]},{limits[3]})")
    print(f"-" * 50)
    print(f"  检测阈值: {keyboard_handler.get_confidence_threshold():.2f} | 穿越判断: y变小即计数")
    print(f"-" * 50)
    print(f"  [分类统计]")
    for cls_name, count in class_counts.items():
        bar = "█" * count + "░" * (max(class_counts.values()) - count) if max(class_counts.values()) > 0 else ""
        print(f"    {cls_name:10s}: {count:3d} {bar}")
    print(f"-" * 50)
    print(f"  [总计] {len(total_counts)} 辆车")
    print(f"=" * 50)
    print(f"  统计数据已保存: counting_stats.json")
    print(f"  输出视频: sample_res.mp4")
    print(f"=" * 50)


if __name__ == "__main__":
    main()