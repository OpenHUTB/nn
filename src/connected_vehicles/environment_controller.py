import carla

# 全局环境状态
global_weather_type = "clear"  # 天气类型：clear/rain/fog/night
global_visibility = 100.0      # 能见度（0-100）
global_time_hour = 12          # 模拟当前小时（0-23）

def set_weather(world: carla.World, weather_type: str = "clear"):
    """
    设置CARLA仿真环境的天气，并更新全局环境状态
    :param world: CARLA World对象
    :param weather_type: 天气类型，可选值：clear/rain/fog/night
    """
    global global_weather_type, global_visibility, global_time_hour
    weather = carla.WeatherParameters()

    # 重置基础天气参数
    weather.precipitation = 0.0
    weather.precipitation_deposits = 0.0
    weather.fog_density = 0.0
    weather.fog_distance = 0.0
    weather.sun_altitude_angle = 30.0  # 太阳高度角（正数：白天，负数：夜晚）
    global_visibility = 100.0

    # 根据天气类型配置参数
    if weather_type == "rain":
        weather.precipitation = 80.0
        weather.precipitation_deposits = 50.0
        global_visibility = 60.0
        global_time_hour = 14  # 午后降雨
    elif weather_type == "fog":
        weather.fog_density = 70.0
        weather.fog_distance = 10.0
        global_visibility = 30.0
        global_time_hour = 8  # 清晨雾天
    elif weather_type == "night":
        weather.sun_altitude_angle = -30.0
        weather.moon_altitude_angle = 30.0
        weather.streetlights = True
        global_visibility = 80.0
        global_time_hour = 22  # 夜间10点
    elif weather_type == "clear":
        global_visibility = 100.0
        global_time_hour = 12  # 正午
    else:
        print(f"⚠️ 未知天气类型：{weather_type}，默认使用晴天")
        weather_type = "clear"

    # 应用天气设置
    world.set_weather(weather)
    global_weather_type = weather_type
    print(f"\n🌤️  天气已切换为：{weather_type} | 能见度：{global_visibility}% | 模拟时间：{global_time_hour}:00")

def get_current_environment_state() -> dict:
    """获取当前环境状态"""
    return {
        "weather_type": global_weather_type,
        "visibility": global_visibility,
        "current_hour": global_time_hour
    }