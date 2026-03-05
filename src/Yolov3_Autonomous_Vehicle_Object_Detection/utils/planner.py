import numpy as np
from config import config


class SimplePlanner:
    """
    基础轨迹规划器
    目前实现功能：基于视觉感知的自动紧急制动 (AEB)
    """

    def __init__(self):
        self.img_width = config.CAMERA_WIDTH
        self.img_height = config.CAMERA_HEIGHT

        # 定义危险区域参数
        # 如果物体中心在画面水平中间 40% 的区域内，视为在车道上
        self.center_zone_ratio = 0.4

        # 如果物体面积超过画面的 10%，视为距离过近，需要刹车
        self.collision_area_threshold = 0.10

    def plan(self, detections):
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
        safe_zone_half_width = (self.img_width * self.center_zone_ratio) / 2

        for (x, y, w, h, class_id, conf) in detections:
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