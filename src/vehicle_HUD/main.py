import carla
import cv2
import numpy as np
import random

# 连接Carla
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()

# 设置同步模式
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# 生成车辆
vehicle_bp = bp_lib.find('vehicle.tesla.model3')
spawn_points = world.get_map().get_spawn_points()
vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
vehicle.set_autopilot(True)

# 创建相机
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')
trans = carla.Transform(carla.Location(x=2.5, z=1.5))
camera = world.spawn_actor(camera_bp, trans, attach_to=vehicle)

images = {}
def callback(data):
    array = np.frombuffer(data.raw_data, dtype=np.uint8)
    images['front'] = array.reshape((480, 640, 4))[:, :, :3]
camera.listen(callback)

# 功能变量
spectator = world.get_spectator()
view_mode = 'top'
speed_limit = 60
speed_warning = True

# 导航
nav_enabled = False
destination = None

def init_nav():
    global nav_enabled, destination
    spawn_points = world.get_map().get_spawn_points()
    current_loc = vehicle.get_location()
    candidates = [sp.location for sp in spawn_points if sp.location.distance(current_loc) > 50]
    if candidates:
        destination = random.choice(candidates)
        nav_enabled = True

# 天气
weather_types = ['clear', 'cloudy', 'rain', 'fog', 'snow']
weather_idx = 0
current_hour = 12

def set_weather():
    w = carla.WeatherParameters()
    if weather_types[weather_idx] == 'clear':
        w.cloudiness, w.precipitation, w.fog_density = 0, 0, 0
    elif weather_types[weather_idx] == 'cloudy':
        w.cloudiness, w.precipitation, w.fog_density = 80, 0, 20
    elif weather_types[weather_idx] == 'rain':
        w.cloudiness, w.precipitation, w.fog_density = 100, 80, 30
    elif weather_types[weather_idx] == 'fog':
        w.cloudiness, w.precipitation, w.fog_density = 90, 20, 60
    elif weather_types[weather_idx] == 'snow':
        w.cloudiness, w.precipitation, w.fog_density = 100, 100, 40
    w.sun_altitude_angle = (current_hour - 6) * 15
    world.set_weather(w)

# 自动泊车
parking = False
park_stage = 0

def do_parking():
    global parking, park_stage
    if not parking: return
    
    ctrl = carla.VehicleControl()
    if park_stage == 0:
        ctrl.brake = 1.0
        park_stage = 1
    elif park_stage == 1:
        ctrl.throttle, ctrl.steer = 0.4, -1.0
        if vehicle.get_velocity().length() > 2: park_stage = 2
    elif park_stage == 2:
        ctrl.brake = 1.0
        park_stage = 3
    elif park_stage == 3:
        ctrl.throttle, ctrl.steer, ctrl.reverse = 0.4, 1.0, True
        park_stage = 4
    elif park_stage == 4:
        ctrl.throttle, ctrl.reverse = 0.3, True
        park_stage = 5
    elif park_stage == 5:
        ctrl.brake = 1.0
        parking = False
        park_stage = 0
        vehicle.set_autopilot(True)
        return
    vehicle.apply_control(ctrl)

# 主循环
cv2.namedWindow("HUD", cv2.WINDOW_NORMAL)
set_weather()

try:
    while True:
        world.tick()
        
        # 更新视角
        vt = vehicle.get_transform()
        if view_mode == 'top':
            spectator.set_transform(carla.Transform(vt.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
        elif view_mode == 'follow':
            spectator.set_transform(carla.Transform(vt.location + carla.Location(x=-5, z=3), carla.Rotation(pitch=-15, yaw=vt.rotation.yaw)))
        elif view_mode == 'chase':
            spectator.set_transform(carla.Transform(vt.location + carla.Location(x=-5, z=2), carla.Rotation(pitch=-10, yaw=vt.rotation.yaw)))
        elif view_mode == 'side':
            spectator.set_transform(carla.Transform(vt.location + carla.Location(y=5, z=3), carla.Rotation(pitch=-15, yaw=vt.rotation.yaw+90)))
        elif view_mode == 'close':
            spectator.set_transform(carla.Transform(vt.location + carla.Location(x=-5, z=4), carla.Rotation(pitch=-25, yaw=vt.rotation.yaw)))
        
        do_parking()
        
        if 'front' in images:
            frame = images['front'].copy()
            speed = np.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2) * 3.6
            speed_color = (0,0,255) if speed_warning and speed > speed_limit else (0,255,0)
            
            cv2.putText(frame, f"Speed: {speed:.1f} km/h", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, speed_color, 2)
            cv2.putText(frame, f"Limit: {speed_limit} km/h", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(frame, f"View: {view_mode}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Weather: {weather_types[weather_idx]}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Time: {current_hour:02d}:00", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Nav: {'ON' if nav_enabled else 'OFF'}", (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"Parking: {'ACTIVE' if parking else 'READY'}", (10,210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if parking else (255,255,255), 2)
            
            cv2.imshow("HUD", frame)
        
        key = cv2.waitKey(1)
        if key == 27: break
        elif key in [ord('t'), ord('T')]: view_mode = 'top'
        elif key in [ord('f'), ord('F')]: view_mode = 'follow'
        elif key in [ord('c'), ord('C')]: view_mode = 'chase'
        elif key in [ord('s'), ord('S')]: view_mode = 'side'
        elif key in [ord('x'), ord('X')]: view_mode = 'close'
        elif key in [ord('+'), ord('=')]: speed_limit += 10
        elif key in [ord('-'), ord('_')]: speed_limit = max(10, speed_limit-10)
        elif key in [ord('w'), ord('W')]: speed_warning = not speed_warning
        elif key in [ord('n'), ord('N')]: init_nav()
        elif key in [ord('v'), ord('V')]: weather_idx = (weather_idx+1)%5; set_weather()
        elif key in [ord('u'), ord('U')]: current_hour = (current_hour+3)%24; set_weather()
        elif key in [ord('p'), ord('P')]: parking = True; vehicle.set_autopilot(False)

finally:
    settings.synchronous_mode = False
    world.apply_settings(settings)
    camera.destroy()
    vehicle.destroy()
    cv2.destroyAllWindows()
