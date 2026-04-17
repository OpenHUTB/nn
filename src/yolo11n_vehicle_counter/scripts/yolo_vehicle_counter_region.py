#!/usr/bin/env python3
"""
YOLO11n 区域车辆计数系统
=====================================

该脚本使用YOLO11n模型和supervision库，通过多边形区域来统计车辆数量。
支持多车道区域计数，分别统计上行和下行方向的车辆。

使用方法:
    python yolo_vehicle_counter_region.py
"""

import cv2 as cv
from ultralytics import YOLO
import numpy as np
import supervision as sv

# ==================== 配置路径 ====================
# 模型文件路径
MODEL_PATH = "../models/yolo11n.pt"
# 输入视频文件路径
INPUT_VIDEO_PATH = "../dataset/sample_region.mp4"
# 输出视频文件路径
OUTPUT_VIDEO_PATH = "../res/sample_region_res.mp4"
# ==================================================


def main(model_path=None, input_video_path=None, output_video_path=None):
    """主函数 - 运行区域车辆计数

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
    w, h, fps = video_info.width, video_info.height, video_info.fps

    print(f"📊 视频信息: {w}x{h}, {fps}fps")

    # 设置标注器参数
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)  # 计算最优线条粗细
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)  # 计算最优文字大小

    # 追踪器和检测平滑器设置
    tracker = sv.ByteTrack(frame_rate=video_info.fps)  # 字节追踪器
    smoother = sv.DetectionsSmoother()  # 检测平滑器，用于稳定检测结果

    # 车辆类别设置
    class_names = model.names  # 获取模型类别名称
    vehicle_classes = ['car', 'motorbike', 'bus', 'truck']  # 定义需要检测的车辆类别
    # 筛选出车辆类别对应的ID
    selected_classes = [cls_id for cls_id, class_name in model.names.items() if class_name in vehicle_classes]

    # 初始化计数器
    total_counts, crossed_ids = [], set()  # 总计数和已计数车辆ID集合
    counts_up, ids_up = [], set()  # 上行计数
    counts_down, ids_down = [], set()  # 下行计数

    # 定义计数区域（多边形区域）
    # 区域1（下行区域）
    arr1 = np.array([[761, 642], [1073, 642], [1070, 732], [968, 776], [872, 1038], [97, 1049]], dtype=np.int32)
    # 区域2（上行区域）
    arr2 = np.array([[1105, 639], [1402, 645], [1920, 959], [1920, 1080], [930, 1073], [991, 811], [1105, 755]],
                    dtype=np.int32)

    # 根据视频分辨率缩放区域坐标
    scale_factor = min(w / 1920, h / 1080)  # 计算缩放因子
    zone_points = [np.floor(arr * scale_factor).astype(np.int32) for arr in [arr1, arr2]]
    zones = [sv.PolygonZone(points) for points in zone_points]
    colors = sv.ColorPalette.from_hex(['#ef260e', '#07f921'])  # 区域颜色（红色和绿色）


    def draw_overlay(frame, points, color, alpha=0.25):
        """绘制半透明覆盖多边形

        Args:
            frame: 输入帧
            points: 多边形顶点
            color: 颜色
            alpha: 透明度
        """
        overlay = frame.copy()
        cv.fillPoly(overlay, [points], color)
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


    def count_vehicle_in_zone(track_id, cx, cy, zone_idx):
        """统计进入区域的车辆

        Args:
            track_id: 车辆追踪ID
            cx, cy: 车辆坐标
            zone_idx: 区域索引
        """
        if cv.pointPolygonTest(zone_points[zone_idx], (cx, cy), False) >= 0:
            if track_id not in crossed_ids:
                total_counts.append(track_id)
                crossed_ids.add(track_id)
            if zone_idx == 0 and track_id not in ids_down:
                counts_down.append(track_id)
                ids_down.add(track_id)
            elif zone_idx == 1 and track_id not in ids_up:
                counts_up.append(track_id)
                ids_up.add(track_id)


    def annotate_frame(frame, detections):
        """标注帧中的车辆和区域

        Args:
            frame: 输入帧
            detections: 检测结果
        """
        # 按车辆类别过滤
        detections = detections[(np.isin(detections.class_id, selected_classes)) & (detections.confidence > 0.5)]

        # 绘制区域覆盖
        for points, color in zip(zone_points, [(88, 117, 234), (11, 244, 113)]):
            draw_overlay(frame, points, color=color, alpha=0.25)

        # 创建标注器
        box_annotator = sv.RoundBoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)  # 圆角矩形标注器
        label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness,
                                            text_position=sv.Position.TOP_CENTER, color_lookup=sv.ColorLookup.TRACK)  # 标签标注器

        # 处理每个区域
        for idx, zone in enumerate(zones):
            zone_annotator = sv.PolygonZoneAnnotator(zone, thickness=4, color=colors.by_idx(idx), text_scale=2,
                                                     text_thickness=2)
            mask = zone.trigger(detections)
            filtered_detections = detections[mask]

            # 绘制边界框和标签
            box_annotator.annotate(frame, filtered_detections)
            label_annotator.annotate(
                frame, detections=filtered_detections,
                labels=[f"{class_names[class_id]} #{trk_id}" for class_id, trk_id in
                        zip(filtered_detections.class_id,
                            filtered_detections.tracker_id)]
            )
            zone_annotator.annotate(frame)

        # 统计车辆（基于底部中心坐标）
        for track_id, bottom_center in zip(detections.tracker_id,
                                           detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)):
            cx, cy = map(int, bottom_center)
            cv.circle(frame, (cx, cy), 4, (0, 255, 255), cv.FILLED)  # 绘制车辆中心点
            count_vehicle_in_zone(track_id, cx, cy, 0)
            count_vehicle_in_zone(track_id, cx, cy, 1)

        # 显示计数器
        counter_labels = [f"COUNTS: {len(total_counts)}", f"UP: {len(counts_up)}", f"DOWN: {len(counts_down)}"]
        count_colors = [(0, 0, 0), (6, 104, 2), (0, 0, 255)]
        cv.rectangle(frame, (0, 0), (300, 150), (255, 255, 255), cv.FILLED)  # 白色背景
        for i, (label, color) in enumerate(zip(counter_labels, count_colors)):
            cv.putText(frame, label, (20, 50 + i * 40), cv.FONT_HERSHEY_SIMPLEX, 1.25, color, 3)


    # 打开视频文件
    cap = cv.VideoCapture(video_path)
    out = cv.VideoWriter(OUTPUT_VIDEO_PATH, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

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
            print(f"处理进度: 第 {frame_count} 帧, 已计数: {len(total_counts)} 辆车")

        # YOLO检测和追踪
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        if detections.tracker_id is not None:
            annotate_frame(frame, detections)

        # 写入帧到输出视频
        out.write(frame)
        # 显示当前帧
        cv.imshow("YOLO11n Region Vehicle Counter", frame)

        if cv.waitKey(1) & 0xff == ord('p'):  # 按'p'键暂停
            print("用户暂停，按任意键继续...")
            cv.waitKey(0)

    # 释放资源
    cap.release()
    out.release()
    cv.destroyAllWindows()

    print(f"处理完成！总计数: {len(total_counts)} 辆车 (上行: {len(counts_up)}, 下行: {len(counts_down)})")


if __name__ == "__main__":
    main()
