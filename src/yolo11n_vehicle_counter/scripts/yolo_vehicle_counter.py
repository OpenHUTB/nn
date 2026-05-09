import cv2 as cv
from ultralytics import YOLO
import numpy as np
import supervision as sv
import os
import json
import time
from keyboard_handler import handle_keyboard_events

# ==================== 配置路径 ====================
# 模型文件路径
MODEL_PATH = "../models/yolo11n.pt"
# 输入视频文件路径
INPUT_VIDEO_PATH = "../dataset/sample.mp4"
# 输出视频文件路径
OUTPUT_VIDEO_PATH = "../res/sample_res.mp4"
# ==================================================

# 检测区域初始值
current_region = [400, 300, 1250, 500]  # [left, top, right, bottom]
region_padding = 50  # 区域扩展的 padding 值
region_adjust_interval = 10  # 每10帧调整一次区域


def auto_adjust_detection_region(frame, detections, current_region, padding=50):
    """自动调整检测区域大小
    
    Args:
        frame: 当前视频帧
        detections: 检测结果
        current_region: 当前区域 [left, top, right, bottom]
        padding: 区域扩展的像素数
        
    Returns:
        list: 调整后的区域 [left, top, right, bottom]
    """
    if len(detections) == 0:
        return current_region
    
    # 获取所有检测到的车辆边界框
    boxes = detections.xyxy
    if len(boxes) == 0:
        return current_region
    
    # 计算所有车辆的边界
    min_x = min(boxes[:, 0])
    max_x = max(boxes[:, 2])
    min_y = min(boxes[:, 1])
    max_y = max(boxes[:, 3])
    
    # 扩展区域边界
    frame_height, frame_width = frame.shape[:2]
    new_left = max(0, int(min_x - padding))
    new_top = max(0, int(min_y - padding))
    new_right = min(frame_width, int(max_x + padding))
    new_bottom = min(frame_height, int(max_y + padding))
    
    # 确保区域不会太小
    min_region_width = 200
    min_region_height = 100
    if new_right - new_left < min_region_width:
        center_x = (new_left + new_right) // 2
        new_left = max(0, center_x - min_region_width // 2)
        new_right = min(frame_width, center_x + min_region_width // 2)
    
    if new_bottom - new_top < min_region_height:
        center_y = (new_top + new_bottom) // 2
        new_top = max(0, center_y - min_region_height // 2)
        new_bottom = min(frame_height, center_y + min_region_height // 2)
    
    return [new_left, new_top, new_right, new_bottom]


def main(model_path=None, input_video_path=None, output_video_path=None, ground_truth_file=None):
    """主函数 - 运行车辆计数

    Args:
        model_path: 模型文件路径 (如果为None则使用默认值)
        input_video_path: 输入视频路径 (如果为None则使用默认值)
        output_video_path: 输出视频路径 (如果为None则使用默认值)
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
    # 基础计数区域 [left, top, right, bottom]
    counting_region = [400, 350, 1250, 450]  # 初始计数区域
    total_counts = []  # 总计数
    type_counts = {'car': [], 'motorbike': [], 'bus': [], 'truck': []}  # 分车型统计
    # 车辆状态跟踪：{track_id: {'in_region': bool, 'entry_time': timestamp}}
    vehicle_states = {}
    # 区域调整参数
    region_adjust_interval = 3  # 每3帧调整一次区域（提高响应速度）
    region_padding = 100  # 区域扩展的padding值（增大扩展范围）
    smooth_factor = 0.6  # 平滑因子（增大响应速度）
    
    # 精度优化参数
    confidence_threshold = 0.6  # 置信度阈值（平衡精确率和召回率）
    min_detection_frames = 3  # 最小检测帧数（连续检测4帧才认为有效）
    iou_threshold = 0.25  # IOU阈值（适中的NMS严格度）
    min_track_length = 4  # 最小轨迹长度（轨迹至少6个点才计数）
    max_speed_threshold = 150  # 最大速度阈值（像素/帧），过滤不合理的跳跃
    
    # 速度计算参数
    speed_history = {}  # 存储车辆速度历史 {track_id: [speed1, speed2, ...]}
    max_speed_history = 10  # 每个车辆保留的最大速度历史记录
    pixel_to_meter_ratio = 0.02  # 像素到米的转换比例（根据实际场景调整，更符合真实道路场景）
    speed_units = 'km/h'  # 速度单位：'km/h' 或 'm/s'
    speed_display_threshold = 10  # 最小显示速度阈值（避免显示低速噪声）
    speed_smoothing_factor = 0.7  # 速度平滑因子，使速度变化更自然

    # 精度衡量配置
    ground_truth_file = "../dataset/ground_truth/ground_truth.txt"  # 可以设置为包含真实车辆总数的文件路径
    # 支持两种格式：
    # 1. 简单数字格式：文件中只有一个数字，表示整个视频的真实车辆总数
    # 2. 详细格式：每行一个数字，表示不同片段的车辆数量（用于分段验证）

    ground_truth_total = None  # 真实车辆总数

    # 尝试加载ground truth数据
    if ground_truth_file and os.path.exists(ground_truth_file):
        try:
            with open(ground_truth_file, 'r') as f:
                content = f.read().strip()

            # 尝试解析为单个数字（整个视频的总车辆数）
            try:
                ground_truth_total = int(content)
                print(f"✅ 已加载ground truth总数: {ground_truth_total} 辆车")
            except ValueError:
                # 如果不是单个数字，尝试解析为多行数据
                lines = content.split('\n')
                numbers = []
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            numbers.append(int(line))
                        except ValueError:
                            continue

                if numbers:
                    # 使用平均值作为总车辆数
                    ground_truth_total = int(np.mean(numbers))
                    print(f"✅ 已加载ground truth数据: {len(numbers)} 个样本，平均 {ground_truth_total} 辆车")
                else:
                    ground_truth_total = None
                    print(f"❌ 无法解析ground truth数据，将使用实时计算模式")
        except Exception as e:
            ground_truth_total = None
            print(f"❌ 无法加载ground truth数据: {e}，将使用实时计算模式")
    else:
        ground_truth_total = None
        print(f"ℹ️  未设置ground truth文件，将使用实时计算模式")


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


    def calculate_vehicle_speed(track_id, cx, cy, vehicle_states, speed_history, fps):
        """计算车辆速度

        Args:
            track_id: 车辆追踪ID
            cx, cy: 车辆中心点坐标
            vehicle_states: 车辆状态字典
            speed_history: 速度历史字典
            fps: 视频帧率

        Returns:
            float: 车辆速度（像素/秒）
        """
        if track_id not in vehicle_states or len(vehicle_states[track_id]['positions']) < 2:
            return 0.0
        
        # 获取最近的两个位置
        positions = vehicle_states[track_id]['positions']
        prev_pos = positions[-2]
        curr_pos = (cx, cy)
        
        # 计算距离（像素）
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        distance_pixels = np.sqrt(dx**2 + dy**2)
        
        # 计算时间差（秒）
        time_diff = 1.0 / fps
        
        # 计算速度（像素/秒）
        speed_pixels_per_second = distance_pixels / time_diff
        
        # 存储速度历史
        if track_id not in speed_history:
            speed_history[track_id] = []
        
        # 应用速度平滑
        if speed_history[track_id]:
            # 使用指数移动平均进行平滑
            last_speed = speed_history[track_id][-1]
            smoothed_speed = last_speed * speed_smoothing_factor + speed_pixels_per_second * (1 - speed_smoothing_factor)
            speed_history[track_id].append(smoothed_speed)
        else:
            speed_history[track_id].append(speed_pixels_per_second)
        
        if len(speed_history[track_id]) > max_speed_history:
            speed_history[track_id].pop(0)
        
        # 返回平均速度
        return np.mean(speed_history[track_id]) if speed_history[track_id] else 0.0

    def convert_pixel_speed_to_real(speed_pixels_per_second, pixel_to_meter_ratio, units='km/h'):
        """将像素速度转换为真实速度

        Args:
            speed_pixels_per_second: 像素/秒
            pixel_to_meter_ratio: 像素到米的转换比例
            units: 速度单位 ('km/h' 或 'm/s')

        Returns:
            float: 真实速度
        """
        # 转换为米/秒
        speed_mps = speed_pixels_per_second * pixel_to_meter_ratio
        
        # 转换为指定单位
        if units == 'km/h':
            return speed_mps * 3.6  # 米/秒 -> 公里/小时
        return speed_mps  # 米/秒

    def count_vehicles_by_region(track_id, cx, cy, region, vehicle_states, total_counts, type_counts, cls_id):
        """基于区域统计车辆 - 改进版：使用轨迹方向判断 + 稳定性检查

        Args:
            track_id: 车辆追踪ID
            cx, cy: 车辆中心点坐标
            region: 计数区域 [left, top, right, bottom]
            vehicle_states: 车辆状态字典
            total_counts: 总计数列表

        Returns:
            bool: 是否计数成功
        """
        left, top, right, bottom = region
        region_center_y = (top + bottom) / 2
        is_in_region = left < cx < right and top < cy < bottom
        
        # 初始化车辆状态
        if track_id not in vehicle_states:
            vehicle_states[track_id] = {
                'in_region': is_in_region,
                'entry_time': frame_count,
                'positions': [(cx, cy)],  # 记录位置历史
                'counted': False,
                'detection_count': 1,  # 连续检测计数
                'last_seen': frame_count  # 上次检测到的帧
            }
            return False
        
        # 更新检测状态
        vehicle_states[track_id]['detection_count'] += 1
        vehicle_states[track_id]['last_seen'] = frame_count
        
        # 更新位置历史
        vehicle_states[track_id]['positions'].append((cx, cy))
        if len(vehicle_states[track_id]['positions']) > 10:  # 只保留最近10个位置
            vehicle_states[track_id]['positions'].pop(0)
        
        # 如果已经计数过，不再计数
        if vehicle_states[track_id]['counted']:
            return False
        
        # 稳定性检查：确保车辆被连续检测到足够帧数
        if vehicle_states[track_id]['detection_count'] < min_detection_frames:
            return False
        
        # 轨迹长度检查：确保轨迹足够长
        positions = vehicle_states[track_id]['positions']
        if len(positions) < min_track_length:
            return False
        
        # 速度一致性检查：过滤不合理的跳跃
        if len(positions) >= 2:
            speeds = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                speed = np.sqrt(dx**2 + dy**2)
                speeds.append(speed)
            
            if speeds:
                avg_speed = np.mean(speeds)
                max_speed = np.max(speeds)
                # 如果平均速度或最大速度超过阈值，认为是不合理轨迹
                if avg_speed > max_speed_threshold or max_speed > max_speed_threshold * 2:
                    return False
        
        # 检测车辆是否穿过区域中心线（从下到上或从上到下）
        if len(positions) >= 3:
            # 计算垂直方向移动
            y_positions = [p[1] for p in positions]
            
            # 检测是否穿过区域中心线
            prev_y = y_positions[-2]
            curr_y = y_positions[-1]
            
            # 从下到上穿过中心线，或从上到下穿过中心线
            if (prev_y > region_center_y and curr_y <= region_center_y) or \
               (prev_y < region_center_y and curr_y >= region_center_y):
                # 确保车辆确实在区域内或附近
                if is_in_region or vehicle_states[track_id]['in_region']:
                    total_counts.append(track_id)
                    vehicle_states[track_id]['counted'] = True
                    type_counts[class_names[cls_id]].append(track_id)
                    return True
        
        # 更新区域状态
        vehicle_states[track_id]['in_region'] = is_in_region
        if is_in_region and not vehicle_states[track_id]['in_region']:
            vehicle_states[track_id]['entry_time'] = frame_count
        
        return False

    def filter_overlapping_detections(detections, iou_threshold=0.3):
        """使用NMS过滤重叠的检测框
        
        Args:
            detections: 检测结果
            iou_threshold: IOU阈值
            
        Returns:
            Detections: 过滤后的检测结果
        """
        if len(detections) == 0:
            return detections
        
        # 获取边界框和置信度
        boxes = detections.xyxy
        scores = detections.confidence
        
        # 使用OpenCV的NMS
        indices = cv.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=confidence_threshold,
            nms_threshold=iou_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return detections[indices]
        return detections

    def adjust_counting_region(frame, detections, current_region, padding=50):
        """动态调整计数区域

        Args:
            frame: 当前视频帧
            detections: 检测结果
            current_region: 当前计数区域 [left, top, right, bottom]
            padding: 区域扩展的像素数

        Returns:
            list: 调整后的区域 [left, top, right, bottom]
        """
        if len(detections) == 0:
            return current_region
        
        # 获取所有检测到的车辆边界框
        boxes = detections.xyxy
        if len(boxes) == 0:
            return current_region
        
        # 计算所有车辆的边界
        min_x = min(boxes[:, 0])
        max_x = max(boxes[:, 2])
        min_y = min(boxes[:, 1])
        max_y = max(boxes[:, 3])
        
        # 扩展区域边界
        frame_height, frame_width = frame.shape[:2]
        new_left = max(0, int(min_x - padding))
        new_top = max(0, int(min_y - padding))
        new_right = min(frame_width, int(max_x + padding))
        new_bottom = min(frame_height, int(max_y + padding))
        
        # 确保区域不会太小
        min_region_width = 200
        min_region_height = 100
        if new_right - new_left < min_region_width:
            center_x = (new_left + new_right) // 2
            new_left = max(0, center_x - min_region_width // 2)
            new_right = min(frame_width, center_x + min_region_width // 2)
        
        if new_bottom - new_top < min_region_height:
            center_y = (new_top + new_bottom) // 2
            new_top = max(0, center_y - min_region_height // 2)
            new_bottom = min(frame_height, center_y + min_region_height // 2)
        
        # 平滑过渡到新区域（使用外部定义的平滑因子）
        left, top, right, bottom = current_region
        new_left = int(left * (1 - smooth_factor) + new_left * smooth_factor)
        new_top = int(top * (1 - smooth_factor) + new_top * smooth_factor)
        new_right = int(right * (1 - smooth_factor) + new_right * smooth_factor)
        new_bottom = int(bottom * (1 - smooth_factor) + new_bottom * smooth_factor)
        
        return [new_left, new_top, new_right, new_bottom]


    def calculate_accuracy_metrics(ground_truth_total, detected_count):
        """计算精度指标

        Args:
            ground_truth_total: 真实的车辆总数
            detected_count: 检测到的车辆数

        Returns:
            dict: 精度指标（如果ground_truth_total为None则返回None）
        """
        if ground_truth_total is None:
            return None

        # 计算精度（基于计数误差）
        error_rate = abs(detected_count - ground_truth_total) / max(ground_truth_total, 1)
        accuracy = 1.0 - error_rate
        return {'accuracy': accuracy, 'precision': accuracy, 'recall': accuracy, 'f1_score': accuracy}

    def draw_tracks_and_count(frame, detections, total_counts, type_counts, region, vehicle_states, speed_history, w=1920):
        """绘制轨迹并统计车辆

        Args:
            frame: 输入帧
            detections: 检测结果
            total_counts: 总计数列表
            region: 计数区域
            vehicle_states: 车辆状态字典
            speed_history: 速度历史字典
            w: 视频宽度
        """
        # 按车辆类别和检测置信度过滤（使用更高的阈值）
        detections = detections[(np.isin(detections.class_id, selected_classes)) & (detections.confidence > confidence_threshold)]
        
        # 过滤重叠检测（NMS）
        if len(detections) > 0:
            detections = filter_overlapping_detections(detections, iou_threshold)

        # 为每个检测框生成标签（包含速度信息）
        labels = []
        for track_id, cls_id in zip(detections.tracker_id, detections.class_id):
            # 计算车辆速度
            center_point = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)[list(detections.tracker_id).index(track_id)]
            cx, cy = map(int, center_point)
            
            # 计算速度
            pixel_speed = calculate_vehicle_speed(track_id, cx, cy, vehicle_states, speed_history, fps)
            real_speed = convert_pixel_speed_to_real(pixel_speed, pixel_to_meter_ratio, speed_units)
            
            # 生成标签
            if real_speed >= speed_display_threshold:
                label = f"#{track_id} {class_names[cls_id]} {real_speed:.1f} {speed_units}"
            else:
                label = f"#{track_id} {class_names[cls_id]}"
            labels.append(label)

        # 绘制边界框、标签和轨迹
        box_annotator.annotate(frame, detections=detections)
        label_annotator.annotate(frame, detections=detections, labels=labels)
        trace_annotator.annotate(frame, detections=detections)

        # 计算精度指标
        accuracy_metrics = calculate_accuracy_metrics(ground_truth_total, len(total_counts))

        # 处理每个检测到的车辆
        tracker_ids = detections.tracker_id
        class_ids = detections.class_id
        center_points = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)

        for i in range(len(tracker_ids)):
            track_id = tracker_ids[i]
            center_point = center_points[i]
            cx, cy = map(int, center_point)
            cls_id = class_ids[i]

            cv.circle(frame, (cx, cy), 4, (0, 255, 255), cv.FILLED)  # 绘制车辆中心点

            if count_vehicles_by_region(track_id, cx, cy, region, vehicle_states, total_counts, type_counts, cls_id):
                # 计数成功时高亮显示计数区域
                draw_overlay(frame, (region[0], region[1]), (region[2], region[3]), alpha=0.25, color=(10, 255, 50))

        # 绘制计数区域
        cv.rectangle(frame, (region[0], region[1]), (region[2], region[3]), (0, 255, 0), 2)
        cv.putText(frame, "Counting Region", (region[0], region[1]-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示车辆计数和精度信息
        sv.draw_text(frame, f"COUNTS: {len(total_counts)}", sv.Point(x=110, y=30), sv.Color.ROBOFLOW, 1.25,
                     2, background_color=sv.Color.WHITE)

        # 显示分车型统计
        type_text = f"Car:{len(type_counts['car'])} Motorbike:{len(type_counts['motorbike'])} Bus:{len(type_counts['bus'])} Truck:{len(type_counts['truck'])}"
        sv.draw_text(frame, type_text, sv.Point(x=110, y=60), sv.Color.YELLOW, 0.7,
                     1, background_color=sv.Color.BLACK)

        # 显示精度指标
        if accuracy_metrics:
            accuracy_text = f"ACC: {accuracy_metrics['accuracy']:.2%} | F1: {accuracy_metrics['f1_score']:.2f}"
            sv.draw_text(frame, accuracy_text, sv.Point(x=w-400, y=30), sv.Color.GREEN, 1.0,
                         1, background_color=sv.Color.WHITE)
        
        # 显示速度统计信息
        if speed_history:
            all_speeds = []
            for speeds in speed_history.values():
                if speeds:
                    all_speeds.extend(speeds)
            
            if all_speeds:
                avg_speed_pixel = np.mean(all_speeds)
                avg_speed_real = convert_pixel_speed_to_real(avg_speed_pixel, pixel_to_meter_ratio, speed_units)
                max_speed_pixel = np.max(all_speeds)
                max_speed_real = convert_pixel_speed_to_real(max_speed_pixel, pixel_to_meter_ratio, speed_units)
                
                speed_text = f"AVG: {avg_speed_real:.1f} {speed_units} | MAX: {max_speed_real:.1f} {speed_units}"
                sv.draw_text(frame, speed_text, sv.Point(x=w//2 - 150, y=30), sv.Color.BLUE, 1.0,
                             1, background_color=sv.Color.WHITE)



    # 打开视频文件
    cap = cv.VideoCapture(video_path)
    output_path = output_video_path  # 设置输出视频路径
    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if not cap.isOpened():
        raise Exception("错误: 无法打开视频文件!")

    # 视频处理主循环
    frame_count = 0
    detection_accuracies = []  # 存储每帧的检测精度
    total_detections = 0  # 总检测数
    correct_detections = 0  # 正确检测数
    
    # FPS计算相关变量
    start_time = time.time()
    fps_history = []  # 存储FPS历史
    frame_time_history = []  # 存储帧时间历史
    fps_update_interval = 10  # 每10帧更新一次FPS

    while cap.isOpened():
        # 记录帧开始时间
        frame_start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 定义追踪感兴趣区域(ROI)
        crop = frame[225:, 220:]
        mask_b = np.zeros_like(frame, dtype=np.uint8)
        mask_w = np.ones_like(crop, dtype=np.uint8) * 255
        mask_b[225:, 220:] = mask_w

        # 应用掩码到原始帧
        ROI = cv.bitwise_and(frame, mask_b)

        # YOLO检测和追踪
        results = model(ROI)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        # 计算帧级精度（基于置信度）
        if len(detections) > 0:
            avg_confidence = np.mean(detections.confidence) if hasattr(detections, 'confidence') and len(detections.confidence) > 0 else 0.5
            detection_accuracies.append(avg_confidence)
            total_detections += len(detections)
            correct_detections += int(len(detections) * avg_confidence)

        # 每N帧自动调整计数区域
        if frame_count % region_adjust_interval == 0:
            counting_region = adjust_counting_region(frame, detections, counting_region, region_padding)

        if detections.tracker_id is not None:
            # 处理车辆轨迹和计数
            draw_tracks_and_count(frame, detections, total_counts, type_counts, counting_region, vehicle_states, speed_history, w)

        # 计算帧时间和FPS
        frame_end_time = time.time()
        frame_time = (frame_end_time - frame_start_time) * 1000  # 转换为毫秒
        current_fps = 1000 / frame_time if frame_time > 0 else 0
        
        # 存储历史数据
        fps_history.append(current_fps)
        frame_time_history.append(frame_time)
        
        # 只保留最近30帧的数据
        if len(fps_history) > 30:
            fps_history.pop(0)
        if len(frame_time_history) > 30:
            frame_time_history.pop(0)
        
        # 计算平均FPS和帧时间
        avg_fps = np.mean(fps_history) if fps_history else 0
        avg_frame_time = np.mean(frame_time_history) if frame_time_history else 0
        
        # 计算状态色标
        if avg_fps > 60:
            fps_color = (0, 255, 0)  # 绿色
        elif avg_fps > 30:
            fps_color = (0, 255, 255)  # 黄色
        else:
            fps_color = (0, 0, 255)  # 红色
        
        # 显示FPS、帧时间和状态色标
        fps_text = f"FPS: {avg_fps:.1f}"
        frame_time_text = f"Frame Time: {avg_frame_time:.1f}ms"
        
        # 绘制FPS信息面板
        cv.rectangle(frame, (w - 200, 60), (w - 10, 120), (255, 255, 255), cv.FILLED)
        cv.rectangle(frame, (w - 200, 60), (w - 10, 120), fps_color, 2)
        cv.putText(frame, fps_text, (w - 190, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv.putText(frame, frame_time_text, (w - 190, 115), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 写入帧到输出视频
        out.write(frame)
        # 显示当前帧
        cv.imshow("Camera", frame)

        # 键盘事件处理
        key = cv.waitKey(1) & 0xff
        if not handle_keyboard_events(key, frame, frame_count, cap, out, "Camera"):
            break

    # 计算整体精度
    overall_accuracy = 0.0
    if detection_accuracies:
        overall_accuracy = np.mean(detection_accuracies)

    detection_precision = 0.0
    if total_detections > 0:
        detection_precision = correct_detections / total_detections

    # 释放资源
    cap.release()
    out.release()
    cv.destroyAllWindows()

    # 输出精度报告
    print("\n" + "="*60)
    print("📊 精度衡量报告")
    print("="*60)
    print(f"📈 处理总帧数: {frame_count}")
    print(f"🚗 总计数车辆: {len(total_counts)}")
    print(f"   - Car: {len(type_counts['car'])} | Motorbike: {len(type_counts['motorbike'])} | Bus: {len(type_counts['bus'])} | Truck: {len(type_counts['truck'])}")
    print(f"📊 平均检测精度: {overall_accuracy:.2%}")
    print(f"🎯 检测精确率: {detection_precision:.2%}")
    print(f"📝 总检测框数: {total_detections}")
    print(f"✅ 正确检测框数: {correct_detections}")

    # 计算并显示计数稳定性（基于计数变化的平滑度）
    if len(total_counts) > 1:
        count_stability = 1.0 / (1.0 + np.std([len(total_counts)] * frame_count))  # 简化计算
        print(f"⚖️  计数稳定性: {count_stability:.2%}")

    print("="*60)


if __name__ == "__main__":
    main()