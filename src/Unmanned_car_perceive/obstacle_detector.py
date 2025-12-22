# obstacle_detector.py
import numpy as np


class ObstacleDetector:
    """简洁版障碍物检测器"""

    def __init__(self):
        self.obstacles = []
        self.warning_level = 0  # 0=无, 1=注意, 2=警告, 3=危险
        self.warning_message = ""  # 初始化时设置空字符串

    def detect(self, point_cloud):
        """检测障碍物 - 简化版本"""
        self.obstacles = []

        if point_cloud is None or len(point_cloud) < 10:
            self.warning_level = 0
            self.warning_message = "前方区域清晰"
            return self.obstacles

        try:
            # 1. 简化的检测逻辑：统计前方10米内的点
            forward_points = point_cloud[
                (point_cloud[:, 0] > 0) &  # 前方
                (point_cloud[:, 0] < 10) &  # 10米内
                (np.abs(point_cloud[:, 1]) < 3)  # 左右3米内
                ]

            if len(forward_points) > 50:  # 足够多的点才认为是障碍物
                # 计算平均距离
                avg_distance = np.mean(forward_points[:, 0])

                # 找到最近的障碍物
                min_distance = np.min(forward_points[:, 0])

                self.obstacles.append({
                    'distance': avg_distance,
                    'min_distance': min_distance,
                    'point_count': len(forward_points)
                })

                # 设置警告级别和消息
                if min_distance < 2.0:
                    self.warning_level = 3
                    self.warning_message = f"危险！前方{min_distance:.1f}米处有障碍物"
                elif min_distance < 4.0:
                    self.warning_level = 2
                    self.warning_message = f"警告：前方{min_distance:.1f}米处有障碍物"
                elif min_distance < 7.0:
                    self.warning_level = 1
                    self.warning_message = f"注意：前方{min_distance:.1f}米处有障碍物"
                else:
                    self.warning_level = 0
                    self.warning_message = "前方区域清晰"
            else:
                self.warning_level = 0
                self.warning_message = "前方区域清晰"

        except Exception as e:
            print(f"障碍物检测出错: {e}")
            self.warning_level = 0
            self.warning_message = "检测系统异常"

        return self.obstacles

    def get_warning_color(self):
        """获取警告颜色"""
        if self.warning_level == 0:
            return (0, 255, 0)  # 绿色
        elif self.warning_level == 1:
            return (255, 255, 0)  # 黄色
        elif self.warning_level == 2:
            return (255, 165, 0)  # 橙色
        else:  # level 3
            return (255, 0, 0)  # 红色