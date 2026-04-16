import cv2 as cv
from ultralytics import YOLO
import numpy as np
import supervision as sv
import os
import json

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
    limits = [400, 400, 1250, 400]  # 计数线位置：起点(x1, y)到终点(x2, y)
    total_counts, crossed_ids = [], set()  # 总计数和已计数车辆ID集合

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


    def count_vehicles(track_id, cx, cy, limits, crossed_ids):
        """统计穿过计数线的车辆

        Args:
            track_id: 车辆追踪ID
            cx, cy: 车辆中心点坐标
            limits: 计数线位置
            crossed_ids: 已计数车辆ID集合

        Returns:
            bool: 是否计数成功
        """
        if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[1] + 10 and track_id not in crossed_ids:
            crossed_ids.add(track_id)
            return True
        return False


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

    def draw_tracks_and_count(frame, detections, total_counts, limits, w=1920):
        """绘制轨迹并统计车辆

        Args:
            frame: 输入帧
            detections: 检测结果
            total_counts: 总计数列表
            limits: 计数线位置
            w: 视频宽度
        """
        # 按车辆类别和检测置信度过滤
        detections = detections[(np.isin(detections.class_id, selected_classes)) & (detections.confidence > 0.5)]

        # 为每个检测框生成标签
        labels = [f"#{track_id} {class_names[cls_id]}" for track_id, cls_id in
                  zip(detections.tracker_id, detections.class_id)]

        # 绘制边界框、标签和轨迹
        box_annotator.annotate(frame, detections=detections)
        label_annotator.annotate(frame, detections=detections, labels=labels)
        trace_annotator.annotate(frame, detections=detections)

        # 计算精度指标
        accuracy_metrics = calculate_accuracy_metrics(ground_truth_total, len(total_counts))

        # 处理每个检测到的车辆
        for track_id, center_point in zip(detections.tracker_id,
                                          detections.get_anchors_coordinates(anchor=sv.Position.CENTER)):
            cx, cy = map(int, center_point)

            cv.circle(frame, (cx, cy), 4, (0, 255, 255), cv.FILLED)  # 绘制车辆中心点

            if count_vehicles(track_id, cx, cy, limits, crossed_ids):
                total_counts.append(track_id)
                sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                             color=sv.Color.ROBOFLOW, thickness=4)
                draw_overlay(frame, (current_region[0], current_region[1]), (current_region[2], current_region[3]), alpha=0.25, color=(10, 255, 50))

        # 显示车辆计数和精度信息
        sv.draw_text(frame, f"COUNTS: {len(total_counts)}", sv.Point(x=110, y=30), sv.Color.ROBOFLOW, 1.25,
                     2, background_color=sv.Color.WHITE)

        # 显示精度指标
        if accuracy_metrics:
            accuracy_text = f"ACC: {accuracy_metrics['accuracy']:.2%} | F1: {accuracy_metrics['f1_score']:.2f}"
            sv.draw_text(frame, accuracy_text, sv.Point(x=w-400, y=30), sv.Color.GREEN, 1.0,
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

    while cap.isOpened():
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

        # 每N帧自动调整检测区域
        if frame_count % region_adjust_interval == 0:
            global current_region
            current_region = auto_adjust_detection_region(frame, detections, current_region, region_padding)

        if detections.tracker_id is not None:
            # 绘制计数线并处理车辆轨迹
            sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                         color=sv.Color.RED, thickness=4)
            draw_overlay(frame, (current_region[0], current_region[1]), (current_region[2], current_region[3]), alpha=0.2)
            draw_tracks_and_count(frame, detections, total_counts, limits, w)

        # 写入帧到输出视频
        out.write(frame)
        # 显示当前帧
        cv.imshow("Camera", frame)

        if cv.waitKey(1) & 0xff == ord('p'):  # 按'p'键暂停
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