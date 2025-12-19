import math
import logging
from typing import List, Tuple

# 配置日志，方便追踪运行状态
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("UAV_Autonomous_Navigation")


class UAVAutonomousNavigation:
    """无人机自主导航核心类：整合航点导航、避障、路径重规划"""

    def __init__(self):
        # 无人机核心状态
        self.current_position: Tuple[float, float] = (0.0, 0.0)  # (纬度, 经度)
        self.current_heading: float = 0.0  # 当前航向（度，0=正北，90=正东）
        self.safety_distance: float = 10.0  # 避障安全距离（米）
        self.arrival_threshold: float = 5.0  # 到达航点判定阈值（米）

        # 导航相关参数
        self.waypoints: List[Tuple[float, float]] = []  # 航点列表
        self.target_waypoint_idx: int = 0  # 当前目标航点索引
        self.is_obstacle_detected: bool = False  # 是否检测到障碍物
        self.obstacle_distance: float = float("inf")  # 障碍物距离（初始为无穷大）

    def set_waypoints(self, waypoints: List[Tuple[float, float]]) -> None:
        """
        设置航点列表，增加输入验证
        :param waypoints: 航点列表，每个元素为(纬度, 经度)元组
        """
        try:
            # 验证航点格式
            if not isinstance(waypoints, list):
                raise TypeError("航点必须为列表格式")
            for idx, wp in enumerate(waypoints):
                if not isinstance(wp, tuple) or len(wp) != 2:
                    raise ValueError(f"第{idx + 1}个航点格式错误，需为(纬度, 经度)元组")
                if not (isinstance(wp[0], (int, float)) and isinstance(wp[1], (int, float))):
                    raise ValueError(f"第{idx + 1}个航点经纬度必须为数字")

            self.waypoints = waypoints
            self.target_waypoint_idx = 0
            logger.info(f"成功设置{len(waypoints)}个航点，初始目标航点：1")

        except (TypeError, ValueError) as e:
            logger.error(f"设置航点失败：{str(e)}")
            raise

    def calculate_haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        高精度计算两个经纬度点的地面距离（米）
        采用Haversine公式，适配无人机低空导航场景
        """
        try:
            # 地球平均半径（米）
            EARTH_RADIUS = 6371008.8
            # 角度转弧度
            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)

            # 经纬度差值
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad

            # Haversine核心计算
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = EARTH_RADIUS * c

            logger.debug(f"距离计算：({lat1:.6f},{lon1:.6f}) → ({lat2:.6f},{lon2:.6f}) = {distance:.2f}米")
            return distance

        except Exception as e:
            logger.error(f"距离计算失败：{str(e)}")
            raise

    def update_uav_state(self, new_lat: float, new_lon: float, new_heading: float = None) -> None:
        """
        更新无人机位置和航向
        :param new_lat: 新纬度
        :param new_lon: 新经度
        :param new_heading: 新航向（可选，不传入则保持原航向）
        """
        try:
            if not (isinstance(new_lat, (int, float)) and isinstance(new_lon, (int, float))):
                raise ValueError("经纬度必须为数字")

            self.current_position = (new_lat, new_lon)
            if new_heading is not None:
                if not isinstance(new_heading, (int, float)) or new_heading < 0 or new_heading > 360:
                    raise ValueError("航向需为0-360之间的数字")
                self.current_heading = new_heading

            logger.info(f"无人机状态更新：位置({new_lat:.6f},{new_lon:.6f})，航向{self.current_heading:.1f}°")

        except ValueError as e:
            logger.error(f"更新无人机状态失败：{str(e)}")
            raise

    def update_obstacle_data(self, distance: float) -> None:
        """
        更新障碍物检测数据（模拟激光雷达/超声波传感器输入）
        :param distance: 前方障碍物距离（米），None表示未检测到
        """
        if distance is None:
            self.is_obstacle_detected = False
            self.obstacle_distance = float("inf")
            logger.info("未检测到障碍物")
            return

        try:
            if not isinstance(distance, (int, float)) or distance < 0:
                raise ValueError("障碍物距离需为非负数字")

            self.obstacle_distance = distance
            self.is_obstacle_detected = distance < self.safety_distance

            if self.is_obstacle_detected:
                logger.warning(f"检测到前方障碍物！距离：{distance:.2f}米（安全阈值：{self.safety_distance}米）")
            else:
                logger.info(f"前方障碍物距离：{distance:.2f}米（安全）")

        except ValueError as e:
            logger.error(f"更新障碍物数据失败：{str(e)}")
            raise

    def replan_path(self) -> List[Tuple[float, float]]:
        """
        路径重规划（简化版）：遇到障碍时生成绕行航点
        :return: 重规划后的航点列表
        """
        logger.info("触发路径重规划：生成绕行航点")

        # 获取当前位置和目标航点
        current_lat, current_lon = self.current_position
        target_lat, target_lon = self.waypoints[self.target_waypoint_idx]

        # 计算绕行偏移量（向左偏移0.0001度，约11米，适配经纬度坐标系）
        offset_lat = (target_lat - current_lat) * 0.1  # 纬度偏移
        offset_lon = (target_lon - current_lon) * 0.1  # 经度偏移

        # 生成绕行航点（当前位置 → 绕行点 → 原目标航点）
        detour_point = (
            current_lat + offset_lon,  # 左偏移经度
            current_lon - offset_lat  # 左偏移纬度
        )

        # 重规划航点列表：保留已完成航点 + 绕行点 + 剩余航点
        replanned_waypoints = (
                self.waypoints[:self.target_waypoint_idx] +
                [detour_point] +
                self.waypoints[self.target_waypoint_idx:]
        )

        logger.info(f"路径重规划完成：新增绕行航点{detour_point}")
        return replanned_waypoints

    def navigate(self) -> None:
        """
        核心导航逻辑：
        1. 检查航点是否为空
        2. 检测障碍物，必要时重规划路径
        3. 飞向目标航点，到达后切换下一个
        """
        # 检查航点
        if not self.waypoints:
            logger.error("导航终止：未设置航点")
            return

        # 检查是否完成所有航点
        if self.target_waypoint_idx >= len(self.waypoints):
            logger.info("导航完成：所有航点已到达")
            return

        # 障碍物检测与路径重规划
        if self.is_obstacle_detected:
            self.waypoints = self.replan_path()

        # 获取目标航点
        target_lat, target_lon = self.waypoints[self.target_waypoint_idx]
        current_lat, current_lon = self.current_position

        # 计算到目标航点的距离
        distance_to_target = self.calculate_haversine_distance(
            current_lat, current_lon, target_lat, target_lon
        )

        # 判断是否到达目标航点
        if distance_to_target < self.arrival_threshold:
            logger.info(f"到达航点{self.target_waypoint_idx + 1}：({target_lat:.6f},{target_lon:.6f})")
            self.target_waypoint_idx += 1
            # 重置障碍物状态
            self.is_obstacle_detected = False
            self.obstacle_distance = float("inf")
        else:
            # 生成飞行指令（实际场景替换为无人机SDK API）
            logger.info(
                f"飞向航点{self.target_waypoint_idx + 1}："
                f"当前距离{distance_to_target:.2f}米，航向{self.current_heading:.1f}°"
            )


# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 初始化导航实例
    uav_nav = UAVAutonomousNavigation()

    # 1. 设置航点（示例：3个真实地理坐标）
    waypoints_list = [
        (39.908823, 116.397470),  # 航点1：天安门
        (39.997563, 116.337624),  # 航点2：颐和园
        (40.006401, 116.397029)  # 航点3：圆明园
    ]
    uav_nav.set_waypoints(waypoints_list)

    # 2. 模拟无人机初始状态
    uav_nav.update_uav_state(39.908820, 116.397468, 270.0)  # 初始位置靠近航点1，航向270°（正西）

    # 3. 模拟导航过程1：无障碍物，飞向航点1
    logger.info("\n===== 导航阶段1：无障碍物 =====")
    uav_nav.update_obstacle_data(20.0)  # 障碍物距离20米（安全）
    uav_nav.navigate()

    # 4. 模拟到达航点1，更新位置
    uav_nav.update_uav_state(39.908823, 116.397470)
    uav_nav.navigate()

    # 5. 模拟导航阶段2：检测到障碍物，触发路径重规划
    logger.info("\n===== 导航阶段2：检测到障碍物 =====")
    uav_nav.update_obstacle_data(8.0)  # 障碍物距离8米（触发避障）
    uav_nav.navigate()

    # 6. 模拟绕开障碍后继续导航
    logger.info("\n===== 导航阶段3：绕开障碍后 =====")
    uav_nav.update_obstacle_data(None)  # 无障碍物
    uav_nav.update_uav_state(39.950000, 116.360000)  # 模拟飞到绕行点
    uav_nav.navigate()