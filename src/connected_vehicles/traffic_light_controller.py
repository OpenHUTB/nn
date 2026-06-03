import carla
import math
import logging
from utils import calculate_vehicle_speed_kmh
from config import TRAFFIC_LIGHT_DETECT_DISTANCE, TRAFFIC_LIGHT_FILTER

logger = logging.getLogger(__name__)

class TrafficLightController:
    def __init__(self):
        self.traffic_lights_cache = None  # 缓存红绿灯列表，减少重复获取

    def _update_traffic_lights_cache(self, world: carla.World) -> None:
        """更新红绿灯缓存（避免每次遍历都重新获取）"""
        if self.traffic_lights_cache is None or len(self.traffic_lights_cache) == 0:
            self.traffic_lights_cache = world.get_actors().filter(TRAFFIC_LIGHT_FILTER)
            logger.debug(f"红绿灯缓存更新：共{len(self.traffic_lights_cache)}个红绿灯")

    def get_vehicle_traffic_light(self, world: carla.World, vehicle: carla.Vehicle) -> carla.TrafficLight:
        """
        获取车辆当前接近的红灯状态红绿灯（优化：缓存+距离过滤）
        :param world: CARLA World对象
        :param vehicle: 车辆Actor
        :return: 红绿灯Actor（无则返回None）
        """
        self._update_traffic_lights_cache(world)
        if not self.traffic_lights_cache:
            return None

        vehicle_loc = vehicle.get_transform().location
        min_distance = float('inf')
        target_light = None

        for light in self.traffic_lights_cache:
            if light.get_state() != carla.TrafficLightState.Red:
                continue

            # 计算距离
            light_loc = light.get_transform().location
            distance = math.sqrt(
                (vehicle_loc.x - light_loc.x)**2 +
                (vehicle_loc.y - light_loc.y)**2 +
                (vehicle_loc.z - light_loc.z)**2
            )

            # 距离过滤
            if distance < TRAFFIC_LIGHT_DETECT_DISTANCE and distance < min_distance:
                min_distance = distance
                target_light = light

        return target_light

    @staticmethod
    def get_traffic_light_state(light: carla.TrafficLight) -> str:
        """获取红绿灯状态（字符串格式）"""
        if not light:
            return "off"
        state = light.get_state()
        state_map = {
            carla.TrafficLightState.Red: "red",
            carla.TrafficLightState.Yellow: "yellow",
            carla.TrafficLightState.Green: "green"
        }
        return state_map.get(state, "off")

    def check_red_light_violation(self, world: carla.World, vehicle: carla.Vehicle) -> bool:
        """检测车辆是否闯红灯"""
        light = self.get_vehicle_traffic_light(world, vehicle)
        if light and self.get_traffic_light_state(light) == "red":
            speed = calculate_vehicle_speed_kmh(vehicle)
            if speed > 0:
                logger.warning(f"闯红灯检测：车速 {speed} km/h | 位置 {vehicle.get_transform().location}")
                return True
        return False

# 全局实例
tl_controller = TrafficLightController()
get_vehicle_traffic_light = tl_controller.get_vehicle_traffic_light
get_traffic_light_state = tl_controller.get_traffic_light_state
check_red_light_violation = tl_controller.check_red_light_violation