import carla
import logging
from config import DEFAULT_WEATHER

logger = logging.getLogger(__name__)

class EnvironmentController:
    def __init__(self):
        self.weather_type = DEFAULT_WEATHER
        self.visibility = 100.0
        self.current_hour = 12

    def set_weather(self, world: carla.World, weather_type: str = DEFAULT_WEATHER) -> None:
        """
        设置CARLA仿真环境的天气，并更新环境状态
        :param world: CARLA World对象
        :param weather_type: 天气类型，可选值：clear/rain/fog/night
        """
        weather = carla.WeatherParameters()

        # 重置基础参数
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.fog_density = 0.0
        weather.fog_distance = 0.0
        weather.sun_altitude_angle = 30.0
        self.visibility = 100.0

        # 配置天气参数
        if weather_type == "rain":
            weather.precipitation = 80.0
            weather.precipitation_deposits = 50.0
            self.visibility = 60.0
            self.current_hour = 14
        elif weather_type == "fog":
            weather.fog_density = 70.0
            weather.fog_distance = 10.0
            self.visibility = 30.0
            self.current_hour = 8
        elif weather_type == "night":
            weather.sun_altitude_angle = -30.0
            weather.moon_altitude_angle = 30.0
            weather.streetlights = True
            self.visibility = 80.0
            self.current_hour = 22
        elif weather_type == "clear":
            self.visibility = 100.0
            self.current_hour = 12
        else:
            logger.warning(f"未知天气类型：{weather_type}，默认使用晴天")
            weather_type = DEFAULT_WEATHER

        # 应用天气设置
        world.set_weather(weather)
        self.weather_type = weather_type
        logger.info(f"天气已切换为：{weather_type} | 能见度：{self.visibility}% | 模拟时间：{self.current_hour}:00")

    def get_current_environment_state(self) -> dict:
        """获取当前环境状态"""
        return {
            "weather_type": self.weather_type,
            "visibility": self.visibility,
            "current_hour": self.current_hour
        }

# 全局实例
env_controller = EnvironmentController()
set_weather = env_controller.set_weather
get_current_environment_state = env_controller.get_current_environment_state