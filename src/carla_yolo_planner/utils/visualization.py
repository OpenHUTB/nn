import cv2
import numpy as np
from config import config


# ============ 1. 交互式置信度阈值滑动条 ============
def create_confidence_trackbar(window_name="CARLA Object Detection"):
    """
    创建置信度阈值滑动条
    """
    cv2.createTrackbar("Confidence", window_name, 50, 100, lambda x: None)


def get_confidence_threshold(window_name="CARLA Object Detection"):
    """
    获取滑动条当前的置信度阈值
    返回值范围 0.0 ~ 1.0
    """
    conf_value = cv2.getTrackbarPos("Confidence", window_name)
    return conf_value / 100.0


def show_confidence_value(image, conf_thres):
    """
    在 FPS 显示下方绘制当前置信度阈值
    """
    conf_text = f"Conf: {conf_thres:.2f}"
    cv2.putText(image, conf_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image


# ============ 2. 目标跟踪器 ============
class Tracker:
    """基于 IoU 的目标跟踪器"""
    def __init__(self, max_disappeared=5):
        self.next_id = 0
        self.objects = {}
        self.max_disappeared = max_disappeared
        self.max_history = 30

    def _compute_iou(self, box1, box2):
        """计算两个框的 IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        iou = inter_area / float(box1_area + box2_area - inter_area + 1e-6)
        return iou

    def update(self, detections):
        """更新跟踪器"""
        if len(detections) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]['disappeared'] += 1
                if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                    del self.objects[obj_id]
            return self.objects

        if len(self.objects) == 0:
            for det in detections:
                x, y, w, h = det[:4]
                obj_id = self.next_id
                self.next_id += 1
                cx, cy = x + w // 2, y + h // 2
                self.objects[obj_id] = {
                    'box': (x, y, w, h),
                    'center': (cx, cy),
                    'history': [(cx, cy)],
                    'disappeared': 0
                }
        else:
            used_ids = set()
            for det in detections:
                x, y, w, h = det[:4]
                det_box = (x, y, w, h)
                best_iou = 0.3
                best_id = None
                
                for obj_id, obj_data in self.objects.items():
                    if obj_id in used_ids:
                        continue
                    iou = self._compute_iou(det_box, obj_data['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_id = obj_id
                
                if best_id is not None:
                    used_ids.add(best_id)
                    cx, cy = x + w // 2, y + h // 2
                    history = self.objects[best_id]['history']
                    history.append((cx, cy))
                    if len(history) > self.max_history:
                        history.pop(0)
                    self.objects[best_id] = {
                        'box': det_box,
                        'center': (cx, cy),
                        'history': history,
                        'disappeared': 0
                    }
                else:
                    obj_id = self.next_id
                    self.next_id += 1
                    cx, cy = x + w // 2, y + h // 2
                    self.objects[obj_id] = {
                        'box': det_box,
                        'center': (cx, cy),
                        'history': [(cx, cy)],
                        'disappeared': 0
                    }

            for obj_id in list(self.objects.keys()):
                if obj_id not in used_ids:
                    self.objects[obj_id]['disappeared'] += 1
                    if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                        del self.objects[obj_id]

        return self.objects


def get_color_by_id(obj_id):
    """根据 ID 生成固定颜色"""
    np.random.seed(obj_id)
    return tuple(np.random.randint(50, 255, 3).tolist())


def draw_trajectories(image, tracker):
    """绘制所有跟踪目标的轨迹线"""
    for obj_id, obj_data in tracker.objects.items():
        history = obj_data['history']
        if len(history) < 2:
            continue
        
        color = get_color_by_id(obj_id)
        points = np.array(history, np.int32)
        
        for i in range(len(points) - 1):
            cv2.line(image, tuple(points[i]), tuple(points[i + 1]), color, 2)
    
    return image


def draw_tracking_ids(image, tracker):
    """绘制跟踪 ID"""
    for obj_id, obj_data in tracker.objects.items():
        x, y, w, h = obj_data['box']
        color = get_color_by_id(obj_id)
        cv2.putText(image, f"ID:{obj_id}", (x, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


# ============ 3. 安全区域雷达热力图 ============
def draw_risk_heatmap(image, detections, classes):
    """
    在图像底部绘制碰撞风险热力图
    """
    h, w = image.shape[:2]
    num_regions = 10
    region_width = w // num_regions
    bar_height_max = 60
    bar_base_y = h - 20

    region_risks = [0.0] * num_regions

    for det in detections:
        x, y, box_w, box_h, class_id, conf = det
        label = str(classes[class_id]).lower()
        
        dynamic_weights = {'car': 1.0, 'person': 1.5, 'truck': 1.2, 'bus': 1.2, 
                         'bicycle': 1.3, 'motorbike': 1.3}
        weight = dynamic_weights.get(label, 1.0)
        relative_distance = 1.0 - (y / h)
        risk = conf * weight * (1.0 - relative_distance * 0.5)
        
        center_x = x + box_w // 2
        region_idx = min(center_x // region_width, num_regions - 1)
        region_risks[region_idx] = max(region_risks[region_idx], risk)

    for i, risk in enumerate(region_risks):
        bar_height = int(risk * bar_height_max)
        x1 = i * region_width
        x2 = x1 + region_width - 2
        
        if risk < 0.3:
            color = (0, 255, 0)
        elif risk < 0.6:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        
        cv2.rectangle(image, (x1, bar_base_y - bar_height), (x2, bar_base_y), color, -1)
        cv2.rectangle(image, (x1, bar_base_y - bar_height), (x2, bar_base_y), (255, 255, 255), 1)

    legend_text = "Collision Risk Heatmap"
    cv2.putText(image, legend_text, (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    max_risk_idx = region_risks.index(max(region_risks)) if any(region_risks) else -1
    if max_risk_idx >= 0:
        warning_text = f"Max Risk: Region {max_risk_idx + 1}"
        cv2.putText(image, warning_text, (w - 180, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return image


# ============ 4. 检测目标计数面板 ============
def draw_object_count(image, detections, classes):
    """
    在右上角绘制检测目标计数面板
    """
    h, w = image.shape[:2]
    
    count_dict = {}
    for (x, y, box_w, box_h, class_id, conf) in detections:
        label = str(classes[class_id])
        count_dict[label] = count_dict.get(label, 0) + 1
    
    if not count_dict:
        return image
    
    sorted_counts = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    
    panel_width = 180
    panel_height = 30 + len(sorted_counts) * 25
    panel_x = w - panel_width - 10
    panel_y = 10
    
    overlay = image.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 1)
    
    cv2.putText(image, "Detection Count", (panel_x + 10, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    y_offset = panel_y + 45
    for label, count in sorted_counts:
        text = f"{label}: {count}"
        cv2.putText(image, text, (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_offset += 22
    
    return image


# ============ 5. 检测耗时百分比条形图 ============
def draw_detection_cost_bar(image, detect_time, total_time, fps):
    """
    在图像底部绘制检测耗时百分比条形图
    """
    h, w = image.shape[:2]
    
    perc = detect_time / total_time if total_time > 0 else 0
    bar_width = int(perc * w)
    bar_height = 20
    bar_y = h - 25
    
    cv2.rectangle(image, (0, bar_y), (w, bar_y + bar_height), (50, 50, 50), -1)
    cv2.rectangle(image, (0, bar_y), (bar_width, bar_y + bar_height), (0, 100, 255), -1)
    cv2.rectangle(image, (0, bar_y), (w, bar_y + bar_height), (255, 255, 255), 1)
    
    detect_text = f"Detection: {perc*100:.0f}%"
    cv2.putText(image, detect_text, (10, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if perc > 0.8 and fps < 20:
        warning_x = w - 40
        cv2.putText(image, "!", (warning_x, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return image


# ============ 6. AEB 决策动画 ============
aeb_active = False
aeb_trigger_frame = 0
aeb_animation_duration = 30


def trigger_aeb_animation(frame_number):
    """触发 AEB 动画"""
    global aeb_active, aeb_trigger_frame
    aeb_active = True
    aeb_trigger_frame = frame_number


def draw_aeb_warning(image, current_frame, vehicle_center=None):
    """
    绘制 AEB 警告动画效果
    """
    global aeb_active, aeb_trigger_frame, aeb_animation_duration
    
    if not aeb_active:
        return image
    
    frames_since_trigger = current_frame - aeb_trigger_frame
    
    if frames_since_trigger > aeb_animation_duration:
        aeb_active = False
        return image
    
    blink_alpha = 0.2 if (frames_since_trigger // 3) % 2 == 0 else 0.1
    
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), -1)
    cv2.addWeighted(overlay, blink_alpha, image, 1 - blink_alpha, 0, image)
    
    text = "AEB ACTIVE"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] - text_size[1]) // 2
    
    cv2.rectangle(image, (text_x - 20, text_y - 40), (text_x + text_size[0] + 20, text_y + 20), (0, 0, 255), -1)
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    if vehicle_center is not None:
        cx, cy = vehicle_center
        max_radius = 100
        current_radius = int((frames_since_trigger / aeb_animation_duration) * max_radius)
        
        for i in range(3):
            ring_radius = current_radius - i * 20
            if ring_radius > 0:
                cv2.circle(image, (cx, cy), ring_radius, (0, 0, 255), 2)
    
    return image


# ============ 7. 深度估计伪彩色叠加层 ============
def draw_depth_overlay(image, depth_frame=None):
    """
    添加深度估计伪彩色叠加层
    """
    h, w = image.shape[:2]
    
    if depth_frame is not None:
        depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        alpha = 0.4
        overlay = cv2.addWeighted(image, 1 - alpha, depth_colormap, alpha, 0)
    else:
        overlay = image.copy()
        for y in range(h):
            depth_value = int((y / h) * 255)
            depth_color = cv2.applyColorMap(np.array([[depth_value]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
            cv2.line(overlay, (0, y), (w, y), tuple(map(int, depth_color)), 1)
        overlay = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    
    legend_h = 15
    legend_w = 150
    legend_x = 10
    legend_y = h - legend_h - 50
    
    for i in range(legend_w):
        value = int(255 * i / legend_w)
        color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
        cv2.line(overlay, (legend_x + i, legend_y), (legend_x + i, legend_y + legend_h), tuple(map(int, color)), 1)
    
    cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h), (255, 255, 255), 1)
    cv2.putText(overlay, "Near", (legend_x, legend_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(overlay, "Far", (legend_x + legend_w - 25, legend_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return overlay


# ============ 基础可视化函数 ============
def draw_safe_zone(image):
    """
    绘制驾驶安全走廊范围 (辅助调试)
    """
    h, w = image.shape[:2]
    center_x = w // 2

    # 计算安全区域宽度的一半
    half_width = int((w * config.SAFE_ZONE_RATIO) / 2)

    # 左边界和右边界
    left_x = center_x - half_width
    right_x = center_x + half_width

    # 颜色 (BGR): 蓝色
    color = (255, 0, 0)
    thickness = 2

    # 画两条竖线
    cv2.line(image, (left_x, 0), (left_x, h), color, thickness)
    cv2.line(image, (right_x, 0), (right_x, h), color, thickness)

    # 在上方标注文字
    cv2.putText(image, "Driving Corridor", (left_x + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    return image


# ============ 绘制检测结果（主函数） ============
def draw_results(image, results, classes, fps=None, tracker=None, detect_time=None, total_time=None, show_depth=False, depth_frame=None):
    """
    绘制检测结果 - 集成所有可视化功能
    
    参数:
        image: 输入图像
        results: 检测结果列表
        classes: 类别名称列表
        fps: 帧率（可选）
        tracker: 目标跟踪器（可选）
        detect_time: 检测耗时（可选）
        total_time: 总处理耗时（可选）
        show_depth: 是否显示深度叠加层（可选）
        depth_frame: 深度图（可选）
    """
    # 类别颜色映射
    class_colors = {
        'car': (0, 255, 0),        # 绿色
        'person': (0, 0, 255),      # 红色
        'traffic light': (255, 255, 0),  # 黄色
        'stop sign': (255, 0, 255),  # 洋红色
        'truck': (0, 255, 255),     # 青色
        'bus': (128, 0, 128),       # 紫色
        'bicycle': (255, 165, 0),   # 橙色
        'motorbike': (128, 128, 0)  # 橄榄绿
    }
    default_color = (0, 255, 0)
    
    # 创建半透明填充图层
    overlay = image.copy()
    
    # 绘制检测框和标签
    for (x, y, w, h, class_id, conf) in results:
        label = str(classes[class_id])
        color = class_colors.get(label.lower(), default_color)
        
        # 半透明填充
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        
        # 边框
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # 标签
        text = f"{label} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y), color, -1)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 混合半透明图层
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
    
    # 绘制轨迹线和跟踪 ID
    if tracker is not None:
        image = draw_trajectories(image, tracker)
        image = draw_tracking_ids(image, tracker)
    
    # 绘制 FPS 信息
    if fps is not None:
        image = show_fps(image, fps)
        # 显示置信度阈值
        image = show_confidence_value(image, config.conf_thres)
    
    # 绘制检测目标计数面板
    image = draw_object_count(image, results, classes)
    
    # 绘制碰撞风险热力图
    image = draw_risk_heatmap(image, results, classes)
    
    # 绘制检测耗时条形图
    if detect_time is not None and total_time is not None and fps is not None:
        image = draw_detection_cost_bar(image, detect_time, total_time, fps)
    
    # 绘制深度叠加层
    if show_depth:
        image = draw_depth_overlay(image, depth_frame)
    
    return image


def show_fps(image, fps):
    """
    在图像左上角绘制实时帧率（FPS）
    """
    fps_text = f"FPS: {fps:.1f}"
    
    if fps < 15:
        fps_color = (0, 0, 255)
    elif fps < 25:
        fps_color = (0, 255, 255)
    else:
        fps_color = (0, 255, 0)
    
    (text_w, text_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(image, (5, 5), (text_w + 15, text_h + 15), (0, 0, 0), -1)
    cv2.putText(image, fps_text, (10, text_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
    
    return image