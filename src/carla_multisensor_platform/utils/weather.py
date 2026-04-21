import math
from colorama import Fore, Back, Style, Cursor

def clamp(value, min_val=0.0, max_val=100.0):
    return max(min_val, min(value, max_val))

class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self.time = 0.0

    def tick(self, delta_second):
        self.time += delta_second * 0.0008        
        self.time %= 2.0 * math.pi
        self.azimuth += 0.025 * delta_second
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self.time)) - 20

    def info(self):
        return self.azimuth, self.altitude


class Storm(object):
    def __init__(self, precipitation):
        self.time = precipitation if precipitation > 0.0 else -50.0
        self.increase = True
        self.cloud = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0
    
    def tick(self, delta_second):
        delta = (0.13 if self.increase else -0.13) * delta_second
        self.time = clamp(self.time + delta, -250.0, 100.0)
        self.cloud = clamp(self.time + 40.0, 0.0, 90.0)
        self.rain = clamp(self.time, 0.0, 80.0)
        delay = -10.0 if self.increase else 90.0
        self.puddles = clamp(self.time + delay, 0.0, 85.0)
        self.wetness = clamp(self.time * 0.5, 0.0, 100.0)
        self.wind = 5.0 if self.cloud <= 20 else 90 if self.cloud >= 70 else 40
        self.fog = clamp(self.time - 10, 0.0, 30.0)
        if self.time == -250.0:
            self.increase = True
        elif self.time == 100.0:
            self.increase = False

    def info(self):
        return self.cloud, self.rain, self.wind


class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        self.sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self.storm = Storm(weather.precipitation)

    def tick(self, delta_second):
        self.sun.tick(delta_second)
        self.storm.tick(delta_second)
        self.weather.cloudiness = self.storm.cloud
        self.weather.precipitation = self.storm.rain
        self.weather.precipitation_deposits = self.storm.puddles
        self.weather.wind_intensity = self.storm.wind
        self.weather.fog_density = self.storm.fog
        self.weather.sun_azimuth_angle = self.sun.azimuth
        self.weather.sun_altitude_angle = self.sun.altitude

    def weather_info(self):
        azimuth_info = f"{Fore.CYAN}{Style.NORMAL}Azimuth: {self.weather.sun_azimuth_angle:3.2f}{Style.RESET_ALL}"
        altitude_info = f"{Fore.CYAN}{Style.NORMAL}Sun altitude angle: {self.weather.sun_altitude_angle:3.2f}{Style.RESET_ALL}"
        cloud_info = f"{Fore.CYAN}{Style.NORMAL}Cloud: {self.weather.cloudiness:3.2f}{Style.RESET_ALL}"
        rain_info = f"{Fore.CYAN}{Style.NORMAL}Rain: {self.weather.precipitation:3.2f}{Style.RESET_ALL}"
        wind_info = f"{Fore.CYAN}{Style.NORMAL}Wind: {self.weather.wind_intensity:3.2f}{Style.RESET_ALL}"

        return f"{azimuth_info} | {altitude_info} | {cloud_info} | {rain_info} | {wind_info}\n"