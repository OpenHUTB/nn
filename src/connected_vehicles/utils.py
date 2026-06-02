import carla
import math

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

def reset_collision_status():
    """重置碰撞状态（保留空函数，避免调用报错）"""
    pass