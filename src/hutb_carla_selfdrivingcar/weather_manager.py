import carla

class WeatherManager:
    def __init__(self, world):
        self.world = world
        
    def set_weather(self, weather_type):
        """设置天气类型"""
        weather_options = {
            'sunny': carla.WeatherParameters.ClearNoon,
            'cloudy': carla.WeatherParameters.CloudyNoon,
            'rainy': carla.WeatherParameters.WetNoon,
            'stormy': carla.WeatherParameters.HardRainNoon,
            'night': carla.WeatherParameters.ClearSunset,
            'rainy_night': carla.WeatherParameters.WetSunset,
            'soft_rain': carla.WeatherParameters.SoftRainNoon
        }
        
        if weather_type in weather_options:
            self.world.set_weather(weather_options[weather_type])
            print(f"[Weather] 天气已切换为: {weather_type}")
        else:
            print(f"[Weather] 未知天气类型: {weather_type}")
            print(f"可用天气: {', '.join(weather_options.keys())}")
            
    def get_current_weather(self):
        """获取当前天气"""
        weather = self.world.get_weather()
        return weather
            
    def cycle_weather(self):
        """循环切换天气"""
        weather_list = ['sunny', 'cloudy', 'rainy', 'stormy', 'soft_rain', 'night', 'rainy_night']
        
        # 获取当前天气的大致类型
        current = self.world.get_weather()
        
        # 根据亮度判断当前天气
        if current.sun_altitude_angle > 45:
            if current.precipitation < 0.1:
                if current.fog_density < 0.1:
                    current_idx = 0  # sunny
                else:
                    current_idx = 4  # foggy
            else:
                if current.precipitation < 0.5:
                    current_idx = 2  # rainy
                else:
                    current_idx = 3  # stormy
        else:
            if current.precipitation < 0.1:
                current_idx = 5  # night
            else:
                current_idx = 6  # rainy_night
        
        # 切换到下一个天气
        next_idx = (current_idx + 1) % len(weather_list)
        self.set_weather(weather_list[next_idx])
        return weather_list[next_idx]