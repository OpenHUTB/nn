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