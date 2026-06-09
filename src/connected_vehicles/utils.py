import carla
import math
import logging
import threading
from config import LOG_LEVEL, LOG_FORMAT

# 初始化日志
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# 线程锁：保证多线程下数据操作安全
status_lock = threading.Lock()

def calculate_vehicle_speed_kmh(vehicle: carla.Vehicle) -> float:
    """
    计算车辆当前速度（km/h），保留1位小数
    :param vehicle: CARLA车辆Actor
    :return: 车速（km/h）
    :raises AttributeError: 车辆Actor无效时抛出
    """
    try:
        velocity = vehicle.get_velocity()
        speed_m_s = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_km_h = speed_m_s * 3.6
        return round(speed_km_h, 1)
    except AttributeError as e:
        logger.error(f"计算车速失败：车辆Actor无效 - {e}")
        return 0.0

def debounce_check(key_pressed: bool, trigger_flag: list) -> bool:
    """
    通用防抖检查函数（解决按键重复触发问题）
    :param key_pressed: 当前按键是否按下
    :param trigger_flag: 防抖标记（用list实现可变对象传递）
    :return: 是否触发有效操作
    """
    if key_pressed and not trigger_flag[0]:
        trigger_flag[0] = True
        return True
    elif not key_pressed and trigger_flag[0]:
        trigger_flag[0] = False
    return False

def safe_update_dict(target_dict: dict, key: str, value) -> None:
    """
    线程安全的字典更新（加锁）
    :param target_dict: 目标字典
    :param key: 要更新的键
    :param value: 要更新的值
    """
    with status_lock:
        if key in target_dict:
            target_dict[key] = value
            logger.debug(f"字典更新：{key} = {value}")