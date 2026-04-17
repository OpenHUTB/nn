import cv2 as cv
from ultralytics import YOLO
import numpy as np
import supervision as sv

# ==================== 配置路径 ====================
# 模型文件路径
MODEL_PATH = "../models/yolo11n.pt"
# 输入视频文件路径（上下行车辆计数）
INPUT_VIDEO_PATH = "../dataset/sample_updown.mp4"
# 输出视频文件路径（上下行计数版本，使用improved后缀）
OUTPUT_VIDEO_PATH = "../res/sample_updown_res_improved.mp4"
# ==================================================


def main(model_path=None, input_video_path=None, output_video_path=None):
    """主函数 - 运行上下行车辆计数

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
    trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=fps * 2,
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
    limits = [0, 300, 1280, 300]  # 计数线位置：起点(x1, y)到终点(x2, y)
    partition_limit = 550  # 分区限制，用于区分上下行
    total_counts, crossed_ids = [], set()  # 总计数和已计数车辆ID集合

    total_counts_up, crossed_ids_up = [], set()  # 上行车辆计数
    total_counts_down, crossed_ids_down = [], set()  # 下行车辆计数


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


    def count_vehicles_up(track_id, cx, cy, limits, crossed_ids_up):
        """统计上行车辆（左侧区域）

        Args:
            track_id: 车辆追踪ID
            cx, cy: 车辆中心点坐标
            limits: 计数线位置
            crossed_ids_up: 已计数上行车辆ID集合

        Returns:
            bool: 是否计数成功
        """
        if limits[0] < cx < partition_limit and limits[1] - 10 < cy < limits[1] + 10 and track_id not in crossed_ids_up:
            crossed_ids_up.add(track_id)
            return True
        return False


    def count_vehicles_down(track_id, cx, cy, limits, crossed_ids_down):
        """统计下行车辆（右侧区域）

        Args:
            track_id: 车辆追踪ID
            cx, cy: 车辆中心点坐标
            limits: 计数线位置
            crossed_ids_down: 已计数下行车辆ID集合

        Returns:
            bool: 是否计数成功
        """
        if partition_limit < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15 and track_id not in crossed_ids_down:
            crossed_ids_down.add(track_id)
            return True
        return False


    def draw_tracks_and_count(frame, detections, total_counts, limits):
        """绘制轨迹并统计车辆

        Args:
            frame: 输入帧
            detections: 检测结果
            total_counts: 总计数列表
            limits: 计数线位置
        """
        # 按车辆类别过滤
        detections = detections[np.isin(detections.class_id, selected_classes)]
        # 为每个检测框生成标签
        labels = [f"#{track_id} {class_names[cls_id]}" for track_id, cls_id in
                  zip(detections.tracker_id, detections.class_id)]

        # 绘制标签、边界框和轨迹
        label_annotator.annotate(frame, detections=detections, labels=labels)
        box_annotator.annotate(frame, detections=detections)
        trace_annotator.annotate(frame, detections=detections)

        # 处理每个检测到的车辆
        for track_id, center_point in zip(detections.tracker_id,
                                          detections.get_anchors_coordinates(anchor=sv.Position.CENTER)):
            cx, cy = map(int, center_point)
            cv.circle(frame, (cx, cy), 4, (0, 255, 255), cv.FILLED)  # 绘制车辆中心点

            # 统计总计数
            if count_vehicles(track_id, cx, cy, limits, crossed_ids):
                total_counts.append(track_id)
                sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                             color=sv.Color.ROBOFLOW, thickness=4)
                draw_overlay(frame, (0, 200), (1287, 400), alpha=0.25, color=(10, 255, 50))

            # 统计上行车辆（左侧区域）
            if count_vehicles_up(track_id, cx, cy, limits, crossed_ids_up):
                total_counts_up.append(track_id)
            # 统计下行车辆（右侧区域）
            if count_vehicles_down(track_id, cx, cy, limits, crossed_ids_down):
                total_counts_down.append(track_id)

        # 显示计数结果
        sv.draw_text(frame, f"COUNTS: {len(total_counts)}", sv.Point(x=120, y=30), sv.Color.ROBOFLOW, 1.25,
                     2, background_color=sv.Color.WHITE)
        sv.draw_text(frame, f"UP: {len(total_counts_up)}", sv.Point(x=560, y=280), sv.Color.WHITE, 1,
                     2)
        sv.draw_text(frame, f"DOWN: {len(total_counts_down)}", sv.Point(x=560, y=320), sv.Color.WHITE, 1,
                     2)


    # 打开视频文件
    cap = cv.VideoCapture(video_path)
    output_path = output_video_path  # 设置输出视频路径
    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if not cap.isOpened():
        raise Exception("错误: 无法打开视频文件!")

    # 视频处理主循环
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 定义追踪感兴趣区域(ROI)
        crop = frame[150:, :]
        mask_b = np.zeros_like(frame, dtype=np.uint8)
        mask_w = np.ones_like(frame[150:, :], dtype=np.uint8) * 255
        mask_b[150:, :] = mask_w

        # 应用掩码到原始帧
        ROI = cv.bitwise_and(frame, mask_b)

        # YOLO检测和追踪
        results = model(ROI)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        if detections.tracker_id is not None:
            # 绘制计数线并处理车辆轨迹
            sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                         color=sv.Color.RED, thickness=4)
            draw_overlay(frame, (0, 200), (1287, 400), alpha=0.2)
            draw_tracks_and_count(frame, detections, total_counts, limits)

        # 写入帧到输出视频
        out.write(frame)
        # 显示当前帧
        cv.imshow("Camera", frame)

        if cv.waitKey(1) & 0xff == ord('p'):  # 按'p'键暂停
            break

    # 释放资源
    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
