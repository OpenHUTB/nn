from typing import List, Optional, Tuple
from config import config


class SimplePlanner:
    """
    基础轨迹规划器
    目前实现功能：基于视觉感知的自动紧急制动 (AEB)
    """

    def __init__(self):
        # 从全局配置中加载参数
        self.img_width = max(1, int(config.CAMERA_WIDTH))
        self.img_height = max(1, int(config.CAMERA_HEIGHT))

        # 驾驶走廊宽度比例 (0.0 - 1.0)
        self.center_zone_ratio = self._clamp_ratio(config.SAFE_ZONE_RATIO)

        # 碰撞预警面积阈值 (0.0 - 1.0)
        self.collision_area_threshold = self._clamp_ratio(config.COLLISION_AREA_THRES)

    @staticmethod
    def _clamp_ratio(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _normalize_detection(detection) -> Optional[Tuple[float, float, float, float]]:
        if detection is None or len(detection) < 4:
            return None

        x, y, w, h = detection[:4]
        try:
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
        except (TypeError, ValueError):
            return None
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h

    def plan(self, detections: List[list]) -> Tuple[bool, str]:
        """
        根据检测结果规划车辆行为

        :param detections: 检测结果列表 [[x, y, w, h, class_id, conf], ...]
        :return: (is_brake, warning_message)
                 is_brake: bool, 是否需要紧急制动
                 warning_message: str, 警告原因
        """
        brake = False
        warning_msg = ""

        img_area = self.img_width * self.img_height
        img_center_x = self.img_width / 2

        # 计算安全区域的半宽 (像素)
        safe_zone_half_width = (self.img_width * self.center_zone_ratio) / 2

        for detection in detections or []:
            normalized_detection = self._normalize_detection(detection)
            if normalized_detection is None:
                continue

            x, y, w, h = normalized_detection
            # 1. 计算物体中心点
            box_center_x = x + (w / 2)

            # 2. 判断物体是否在车辆正前方的“驾驶走廊”内
            dist_to_center = abs(box_center_x - img_center_x)
            is_in_path = dist_to_center < safe_zone_half_width

            if is_in_path:
                # 3. 基于面积估算距离 (视觉测距的简易替代方案)
                box_area = w * h
                area_ratio = box_area / img_area

                # 如果物体够大，说明离得很近了
                if area_ratio > self.collision_area_threshold:
                    brake = True
                    warning_msg = f"Obstacle Ahead! Area: {area_ratio:.2%}"
                    # 只要发现一个危险障碍物，立即决策刹车，跳出循环
                    break

        return brake, warning_msg
