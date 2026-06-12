import numpy as np
from typing import List, Tuple, Optional
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import config


class PurePursuitPlanner:
    """
    纯追踪算法（Pure Pursuit）轨迹跟踪控制器
    
    核心思想：在车辆前方设定一个"前视点"(lookahead point)，
    车辆朝向该点行驶，从而实现沿预设路径自动行驶。
    
    算法步骤：
    1. 根据当前速度和设定的前视距离，找到路径上的前视点
    2. 将前视点转换到车辆坐标系
    3. 计算弧长（转向角）alpha
    4. 根据转向角和自行车模型计算前轮转角
    """
    
    def __init__(
        self,
        wheelbase: float = 2.9,        # 车辆轴距 (m)，约等于车辆前后轮距离
        lookahead_distance: float = 5.0, # 前视距离 (m)，越大越平滑但响应越慢
        lookahead_gain: float = 0.5,    # 前视距离系数：ld = lookahead_gain * speed + lookahead_distance
        min_lookahead: float = 3.0,      # 最小前视距离 (m)
        max_lookahead: float = 15.0,     # 最大前视距离 (m)
        steering_limit: float = 1.0,    # 最大转向角限幅 (rad，约57度)
        speed_threshold: float = 0.5     # 速度阈值，低于此速度不转向
    ):
        self.wheelbase = wheelbase
        self.lookahead_distance = lookahead_distance
        self.lookahead_gain = lookahead_gain
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.steering_limit = steering_limit
        self.speed_threshold = speed_threshold
        
        # 预设轨迹路径（默认：直线前进）
        self.global_path = self._create_default_path()
        
    def _create_default_path(self) -> np.ndarray:
        """
        创建默认直线路径
        格式：[[x, y], [x, y], ...]，单位：米
        """
        path = []
        for i in range(100):  # 生成100个路径点
            x = i * 1.0  # 每1米一个点
            y = 0.0      # y=0表示车道中心
            path.append([x, y])
        return np.array(path)
    
    def set_path(self, path: np.ndarray):
        """
        设置全局路径
        
        Args:
            path: numpy数组，shape=(N, 2)，每行是[x, y]坐标
        """
        if len(path) < 2:
            raise ValueError("路径至少需要2个点")
        self.global_path = path
    
    def set_curve_path(self, length: float = 100, curve_radius: float = 50):
        """
        设置弯道路径（用于测试）
        
        Args:
            length: 路径总长度 (m)
            curve_radius: 弯道曲率半径 (m)，越小越弯
        """
        path = []
        num_points = int(length)
        for i in range(num_points):
            x = float(i)
            # 左转弯道：y = (x^2) / (2R)
            y = (x ** 2) / (2 * curve_radius)
            path.append([x, y])
        self.global_path = np.array(path)
    
    def _get_lookahead_distance(self, speed: float) -> float:
        """
        根据速度动态计算前视距离
        
        Args:
            speed: 当前车速 (m/s)
            
        Returns:
            前视距离 (m)
        """
        ld = self.lookahead_gain * speed + self.lookahead_distance
        return np.clip(ld, self.min_lookahead, self.max_lookahead)
    
    def _find_lookahead_point(
        self, 
        current_pos: Tuple[float, float],
        closest_idx: int
    ) -> Tuple[np.ndarray, int]:
        """
        在路径上找到距离为前视距离的目标点
        
        Args:
            current_pos: 当前位置 (x, y)
            closest_idx: 路径上离当前位置最近的点的索引
            
        Returns:
            lookahead_point: 前视点坐标 [x, y]
            lookahead_idx: 前视点在路径中的索引
        """
        current_pos = np.array(current_pos)
        lookahead_dist = self._get_lookahead_distance(0)  # 先用默认距离找点
        
        # 从最近点开始往后找
        for i in range(closest_idx, len(self.global_path)):
            dist = np.linalg.norm(self.global_path[i] - current_pos)
            if dist >= lookahead_dist:
                return self.global_path[i], i
        
        # 如果没找到，返回路径终点
        return self.global_path[-1], len(self.global_path) - 1
    
    def _find_closest_point(self, current_pos: Tuple[float, float]) -> int:
        """
        找到路径上离当前位置最近的点
        
        Args:
            current_pos: 当前位置 (x, y)
            
        Returns:
            最近点的索引
        """
        current_pos = np.array(current_pos)
        distances = np.linalg.norm(self.global_path - current_pos, axis=1)
        return int(np.argmin(distances))
    
    def compute_steering(
        self, 
        current_pos: Tuple[float, float],
        current_yaw: float,
        speed: float
    ) -> float:
        """
        计算转向角控制量（Pure Pursuit核心算法）
        
        Args:
            current_pos: 当前位置 (x, y) in 世界坐标系
            current_yaw: 当前偏航角 (rad)，0=朝向X正方向，逆时针为正
            speed: 当前车速 (m/s)
            
        Returns:
            steering: 前轮转角控制量 (rad)，正=左转，负=右转
        """
        if speed < self.speed_threshold:
            return 0.0
        
        # 1. 找到路径上最近的点
        closest_idx = self._find_closest_point(current_pos)
        
        # 2. 计算前视距离（根据当前速度）
        lookahead_dist = self._get_lookahead_distance(speed)
        
        # 3. 找到前视点
        lookahead_point = None
        lookahead_idx = closest_idx
        
        for i in range(closest_idx, len(self.global_path)):
            dist = np.linalg.norm(self.global_path[i] - np.array(current_pos))
            if dist >= lookahead_dist:
                lookahead_point = self.global_path[i]
                lookahead_idx = i
                break
        
        if lookahead_point is None:
            lookahead_point = self.global_path[-1]
            lookahead_idx = len(self.global_path) - 1
        
        # 4. 将前视点转换到车辆坐标系
        # 先平移，再旋转（绕原点旋转-current_yaw）
        dx = lookahead_point[0] - current_pos[0]
        dy = lookahead_point[1] - current_pos[1]
        
        # 旋转角度 = -(current_yaw)，因为要从世界坐标系转到车身坐标系
        cos_yaw = np.cos(-current_yaw)
        sin_yaw = np.sin(-current_yaw)
        
        # 车辆坐标系下的坐标
        # x_axis = 车身正前方，y_axis = 车身正左方
        lookahead_x = dx * cos_yaw - dy * sin_yaw  # 纵向距离（前方为正）
        lookahead_y = dx * sin_yaw + dy * cos_yaw  # 横向距离（左为正，右为负）
        
        # 5. 计算弧长 alpha（前视点相对于车辆的夹角）
        # 使用向量的角度计算
        alpha = np.arctan2(lookahead_y, lookahead_x)
        
        # 6. 计算前轮转角（Pure Pursuit公式）
        # curvature = 2 * sin(alpha) / ld
        # steering_angle = arctan(wheelbase * curvature)
        ld = np.sqrt(lookahead_x**2 + lookahead_y**2)  # 实际前视距离
        
        if ld < 0.1:  # 防止除零
            return 0.0
        
        curvature = 2 * np.sin(alpha) / ld
        steering = np.arctan(self.wheelbase * curvature)
        
        # 7. 限幅
        steering = np.clip(steering, -self.steering_limit, self.steering_limit)
        
        # 存储计算结果供调试
        self._last_debug = {
            'lookahead_point': lookahead_point,
            'lookahead_idx': lookahead_idx,
            'lookahead_dist': ld,
            'alpha': alpha,
            'steering': steering,
            'closest_idx': closest_idx
        }
        
        return steering
    
    def compute_control(
        self,
        current_pos: Tuple[float, float],
        current_yaw: float,
        speed: float,
        target_speed: float = None,
        max_throttle: float = 0.5
    ) -> Tuple[float, float]:
        """
        计算完整的控制量（转向 + 速度）
        
        Args:
            current_pos: 当前位置 (x, y)
            current_yaw: 当前偏航角 (rad)
            speed: 当前车速 (m/s)
            target_speed: 目标车速 (m/s)，None=保持当前速度
            max_throttle: 最大油门 (0-1)
            
        Returns:
            (throttle, steering): 油门控制量，转向角控制量
        """
        # 计算转向角
        steering = self.compute_steering(current_pos, current_yaw, speed)
        
        # 计算油门/刹车
        throttle = 0.0
        if target_speed is not None:
            speed_error = target_speed - speed
            if speed_error > 0.1:
                # 加速
                throttle = min(max_throttle, speed_error / 5.0)
            elif speed_error < -0.5:
                # 刹车
                throttle = -min(0.5, abs(speed_error) / 5.0)
        
        return throttle, steering
    
    def is_path_completed(self, current_pos: Tuple[float, float]) -> bool:
        """
        检查是否到达路径终点
        
        Args:
            current_pos: 当前位置
            
        Returns:
            True if 到达终点
        """
        dist_to_end = np.linalg.norm(np.array(current_pos) - self.global_path[-1])
        return dist_to_end < 2.0  # 2米内算到达
    
    def get_progress(self, current_pos: Tuple[float, float]) -> float:
        """
        获取路径完成进度
        
        Args:
            current_pos: 当前位置
            
        Returns:
            进度 (0.0 - 1.0)
        """
        closest_idx = self._find_closest_point(current_pos)
        return closest_idx / (len(self.global_path) - 1)
    
    def get_debug_info(self) -> dict:
        """获取调试信息"""
        return self._last_debug if hasattr(self, '_last_debug') else {}


class SimplePlanner:
    """
    基础轨迹规划器
    目前实现功能：基于视觉感知的自动紧急制动 (AEB)
    """

    def __init__(self):
        # 从全局配置中加载参数
        self.img_width = config.CAMERA_WIDTH
        self.img_height = config.CAMERA_HEIGHT

        # 驾驶走廊宽度比例 (0.0 - 1.0)
        self.center_zone_ratio = config.SAFE_ZONE_RATIO

        # 碰撞预警面积阈值 (0.0 - 1.0)
        self.collision_area_threshold = config.COLLISION_AREA_THRES

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