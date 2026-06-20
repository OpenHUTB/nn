import carla
import random
import logging
from datetime import datetime
from utils import calculate_vehicle_speed_kmh, safe_update_dict
from nn.src.connected_vehicles.src.config import (
    VITALS_BASE_HEART_RATE, VITALS_BASE_BLOOD_PRESSURE, VITALS_BASE_FATIGUE,
    VITALS_SPEED_FACTOR, VITALS_TIME_FACTOR, VITALS_WEATHER_FACTORS,
    VITALS_COLLISION_FACTOR
)

logger = logging.getLogger(__name__)


class DriverVitalsMonitor:
    def __init__(self):
        # 基础体征参数
        self.base_heart_rate = VITALS_BASE_HEART_RATE  # 基础心率(次/分钟)
        self.base_blood_pressure = VITALS_BASE_BLOOD_PRESSURE  # 基础血压(收缩压/舒张压)
        self.base_fatigue = VITALS_BASE_FATIGUE  # 基础疲惫度(0-100)

        # 动态监测数据
        self.driving_start_time = datetime.now()
        self.current_heart_rate = self.base_heart_rate
        self.current_blood_pressure = self.base_blood_pressure
        self.current_fatigue = self.base_fatigue
        self.last_collision_speed = 0.0  # 上次碰撞车速
        self.vitals_data = {
            "heart_rate": self.base_heart_rate,
            "blood_pressure": f"{self.base_blood_pressure[0]}/{self.base_blood_pressure[1]}",
            "fatigue": self.base_fatigue,
            "fatigue_level": "normal"
        }

    def _calculate_driving_duration_min(self) -> float:
        """计算累计驾驶时长(分钟)"""
        duration = datetime.now() - self.driving_start_time
        return duration.total_seconds() / 60

    def _get_weather_factor(self, weather_type: str) -> float:
        """获取天气对体征的影响因子"""
        return VITALS_WEATHER_FACTORS.get(weather_type, 1.0)

    def _update_fatigue_level(self) -> None:
        """更新疲惫度等级"""
        if self.current_fatigue < 30:
            level = "normal"
        elif 30 <= self.current_fatigue < 60:
            level = "tired"
        elif 60 <= self.current_fatigue < 80:
            level = "fatigued"
        else:
            level = "extreme_fatigue"
        self.vitals_data["fatigue_level"] = level

    def update_vitals(self, vehicle: carla.Vehicle, weather_type: str, collision_occurred: bool = False) -> None:
        """
        更新驾驶员体征数据（结合车速、驾驶时长、天气、碰撞）
        :param vehicle: 车辆Actor
        :param weather_type: 当前天气类型
        :param collision_occurred: 是否发生碰撞
        """
        # 1. 获取基础参数
        current_speed = calculate_vehicle_speed_kmh(vehicle)
        driving_duration = self._calculate_driving_duration_min()
        weather_factor = self._get_weather_factor(weather_type)

        # 2. 更新碰撞车速（若有碰撞）
        if collision_occurred:
            self.last_collision_speed = current_speed
            logger.warning(f"碰撞影响：车速{current_speed}km/h，体征波动加剧")

        # 3. 计算心率（基础值 + 车速影响 + 时长影响 + 天气影响 + 碰撞影响）
        speed_heart_impact = current_speed * VITALS_SPEED_FACTOR["heart_rate"]
        time_heart_impact = driving_duration * VITALS_TIME_FACTOR["heart_rate"]
        collision_heart_impact = self.last_collision_speed * VITALS_COLLISION_FACTOR["heart_rate"]
        self.current_heart_rate = min(
            self.base_heart_rate + speed_heart_impact + time_heart_impact * weather_factor + collision_heart_impact,
            180  # 心率上限
        )
        # 加入微小随机波动模拟真实体征
        self.current_heart_rate = round(self.current_heart_rate + random.uniform(-2, 2), 1)

        # 4. 计算血压（基础值 + 车速影响 + 天气影响 + 碰撞影响）
        sys_base, dia_base = self.base_blood_pressure
        speed_bp_impact = current_speed * VITALS_SPEED_FACTOR["blood_pressure"]
        collision_bp_impact = self.last_collision_speed * VITALS_COLLISION_FACTOR["blood_pressure"]
        new_sys = min(
            sys_base + speed_bp_impact * weather_factor + collision_bp_impact,
            180  # 收缩压上限
        )
        new_dia = min(
            dia_base + (speed_bp_impact * 0.5) * weather_factor + (collision_bp_impact * 0.5),
            120  # 舒张压上限
        )
        self.current_blood_pressure = (round(new_sys), round(new_dia))

        # 时间流速 ×10 倍
        time_fatigue = driving_duration * VITALS_TIME_FACTOR["fatigue"] * 10
        weather_fatigue = time_fatigue * weather_factor
        collision_fatigue = self.last_collision_speed * 0.3 if collision_occurred else 0

        # 计算
        total_fatigue = self.base_fatigue + weather_fatigue + collision_fatigue
        total_fatigue = min(total_fatigue, 100)

        # 强制递增，绝不卡住
        self.current_fatigue = max(self.current_fatigue, round(total_fatigue, 1))
        # ==================================================================================

        # 6. 更新体征数据字典
        safe_update_dict(self.vitals_data, "heart_rate", self.current_heart_rate)
        safe_update_dict(self.vitals_data, "blood_pressure",
                         f"{self.current_blood_pressure[0]}/{self.current_blood_pressure[1]}")
        safe_update_dict(self.vitals_data, "fatigue", self.current_fatigue)
        self._update_fatigue_level()

        # 7. 日志记录异常体征
        if self.current_heart_rate > 120:
            logger.warning(
                f"心率异常：{self.current_heart_rate}次/分钟（驾驶时长：{driving_duration:.1f}分钟，车速：{current_speed}km/h）")
        if self.current_fatigue > 80:
            logger.warning(f"极度疲惫：疲惫度{self.current_fatigue}%，建议立即停车休息")

    def get_vitals_data(self) -> dict:
        """获取当前体征数据"""
        return self.vitals_data.copy()

    def reset_vitals(self) -> None:
        """重置体征数据（车辆重置时调用）"""
        self.driving_start_time = datetime.now()
        self.current_heart_rate = self.base_heart_rate
        self.current_blood_pressure = self.base_blood_pressure
        self.current_fatigue = self.base_fatigue
        self.last_collision_speed = 0.0
        self.vitals_data = {
            "heart_rate": self.base_heart_rate,
            "blood_pressure": f"{self.base_blood_pressure[0]}/{self.base_blood_pressure[1]}",
            "fatigue": self.base_fatigue,
            "fatigue_level": "normal"
        }
        logger.info("驾驶员体征数据已重置")


# 全局实例
vitals_monitor = DriverVitalsMonitor()
update_driver_vitals = vitals_monitor.update_vitals
get_driver_vitals = vitals_monitor.get_vitals_data
reset_driver_vitals = vitals_monitor.reset_vitals