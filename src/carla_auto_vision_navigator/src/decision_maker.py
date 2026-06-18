<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Project: CARLA AutoVision Navigator
Module: Decision - Behavioral Logic
Version: v1.0.0
Description: 行为决策模块。根据视觉感知层返回的目标信息进行风险评估，实现紧急自动制动（AEB）等避障决策。
Author: wangadsa
License: MIT License
"""
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
=======
import logging
from config import UNSTRUCTURED_SCENES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDecisionMaker:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.anomaly_levels = {
            "normal": 0,
            "pothole": 1,  # 低风险：减速即可
            "fallen_tree": 2,  # 中风险：绕行
            "construction": 2,
            "broken_road": 2,
            "stray_vehicle": 3,  # 高风险：紧急停车
            "pedestrian_violation": 3,
            "unknown": 0
        }

    def make_decision(self, anomaly_result):
        """根据异常检测结果生成决策"""
        anomaly_type = anomaly_result["anomaly_type"]
        confidence = anomaly_result["confidence"]
        level = self.anomaly_levels.get(anomaly_type, 0)

        if confidence < 0.5:
            decision = "continue"
            action = {"speed": 30, "steer": 0.0, "brake": 0.0}
        else:
            if level == 1:
                decision = "slow_down"
                action = {"speed": 10, "steer": 0.0, "brake": 0.2}
            elif level == 2:
                decision = "detour"
                action = {"speed": 15, "steer": 0.5, "brake": 0.1}  # 向右绕行
            elif level == 3:
                decision = "emergency_stop"
                action = {"speed": 0, "steer": 0.0, "brake": 1.0}
            else:
                decision = "continue"
                action = {"speed": 30, "steer": 0.0, "brake": 0.0}

        logger.info(f"异常类型：{anomaly_type} | 置信度：{confidence:.2f} | 决策：{decision}")
        return decision, action
>>>>>>> upstream/main
