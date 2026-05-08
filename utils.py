import carla
import math

# 全局变量：存储碰撞时的车速
GLOBAL_COLLISION_SPEED = 0.0

def calculate_vehicle_speed_kmh(vehicle: carla.Vehicle) -> float:
    """
    计算车辆当前速度（km/h）
    :param vehicle: CARLA车辆Actor
    :return: 车速（km/h）
    """
    velocity = vehicle.get_velocity()
    speed_m_s = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    speed_km_h = speed_m_s * 3.6  # 米/秒 → 千米/小时
    return round(speed_km_h, 1)

def set_global_collision_speed(speed: float):
    """设置全局碰撞车速"""
    global GLOBAL_COLLISION_SPEED
    GLOBAL_COLLISION_SPEED = speed

def get_global_collision_speed() -> float:
    """获取全局碰撞车速"""
    return GLOBAL_COLLISION_SPEED

def reset_collision_status():
    """重置碰撞状态"""
    global GLOBAL_COLLISION_SPEED
    GLOBAL_COLLISION_SPEED = 0.0