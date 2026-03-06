# -*- coding: utf-8 -*-
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class DecisionMaker:
    """
    决策器：根据感知到的目标信息，决定车辆的行驶状态
    """

    def __init__(self):
        self.emergency_brake = False

    def process_detections(self, detections, frame_height):
        """
        分析 YOLO 检测结果，判断是否存在碰撞风险
        """
        self.emergency_brake = False

        for det in detections:
            # 只关注潜在的交通障碍物
            if det['class'] in config.OBSTACLE_CLASSES:
                x, y, w, h = det['box']

                # 简单的测距逻辑：利用框的高度占比
                # 如果障碍物在画面中心附近且足够大，视为危险
                box_height_ratio = h / frame_height

                if box_height_ratio > config.DANGER_THRESHOLD_HEIGHT:
                    self.emergency_brake = True
                    print(f"警告：检测到前方有 {det['class']}，执行紧急避障！")
                    break

        return self.emergency_brake