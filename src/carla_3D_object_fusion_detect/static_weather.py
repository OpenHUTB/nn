import carla
import time

# 连接 CARLA 服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# 设置强雨天天气
weather = carla.WeatherParameters(
    cloudiness=90.0,
    precipitation=90.0,
    precipitation_deposits=90.0,
    wind_intensity=20.0,
    wetness=90.0,
    rain_intensity=100.0 
)
world.set_weather(weather)

print("✅ 雨天天气已设置成功！")
time.sleep(2)
