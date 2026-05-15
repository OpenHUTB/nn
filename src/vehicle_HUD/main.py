import carla
import random
import cv2
import numpy as np

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
    except Exception as e:
        print(f"连接 Carla 失败: {e}")
        return

    # 同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    # 生成车辆（通用蓝图）
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter('vehicle.*')
    if not vehicle_bps:
        print("未找到车辆蓝图")
        return
    vehicle_bp = random.choice(vehicle_bps)

    spawn_points = world.get_map().get_spawn_points()
    vehicle = None
    for sp in random.sample(spawn_points, len(spawn_points)):
        try:
            vehicle = world.spawn_actor(vehicle_bp, sp)
            break
        except Exception as e:
            print(f"生成失败 {sp}: {e}")
            continue

    if not vehicle:
        print("无法生成车辆，请检查地图是否有可用生成点。")
        return

    vehicle.set_autopilot(True)
    spectator = world.get_spectator()

    # 路径规划导航系统（全局变量，在函数内部定义）
    nav_enabled = False
    destination = None
    next_waypoint = None
    navigation_route = []
    
    # 初始化导航（自动启动）
    def init_navigation():
        nonlocal destination, next_waypoint, navigation_route, nav_enabled
        map = world.get_map()
        spawn_points = map.get_spawn_points()
        if len(spawn_points) >= 2:
            # 随机选择一个目的地（不是当前位置）
            current_loc = vehicle.get_location()
            candidates = [sp for sp in spawn_points if sp.location.distance(current_loc) > 50]
            if candidates:
                destination_point = random.choice(candidates)
                destination = destination_point.location
                navigation_route = [destination]
                next_waypoint = destination
                nav_enabled = True
                print(f"导航已启动，目的地距离: {current_loc.distance(destination):.1f} 米")
            else:
                print("无法找到合适的目的地")
        else:
            print("地图上可用的生成点不足")
    
    # 启动时自动初始化导航
    init_navigation()

    # 视角模式
    camera_mode = 'top'  # 'top', 'follow', 'chase', 'side'
    
    # 90° 俯视跟随
    def update_top_view():
        trans = vehicle.get_transform()
        loc = trans.location + carla.Location(z=20)
        rot = carla.Rotation(pitch=-90, yaw=trans.rotation.yaw)
        spectator.set_transform(carla.Transform(loc, rot))
    
    # 第三人称跟随视角
    def update_follow_view():
        trans = vehicle.get_transform()
        loc = trans.location + carla.Location(x=-5, z=3)
        rot = carla.Rotation(pitch=-15, yaw=trans.rotation.yaw)
        spectator.set_transform(carla.Transform(loc, rot))
    
    # 追逐视角（车尾正后方）
    def update_chase_view():
        trans = vehicle.get_transform()
        loc = trans.location + carla.Location(x=-8, z=2)
        rot = carla.Rotation(pitch=0, yaw=trans.rotation.yaw)
        spectator.set_transform(carla.Transform(loc, rot))
    
    # 侧视视角
    def update_side_view():
        trans = vehicle.get_transform()
        loc = trans.location + carla.Location(y=-6, z=3)
        rot = carla.Rotation(pitch=-10, yaw=trans.rotation.yaw + 90)
        spectator.set_transform(carla.Transform(loc, rot))
    
    # 更新 spectator 视角
    def update_spectator():
        if camera_mode == 'top':
            update_top_view()
        elif camera_mode == 'follow':
            update_follow_view()
        elif camera_mode == 'chase':
            update_chase_view()
        elif camera_mode == 'side':
            update_side_view()

    # 相机传感器
    cam_w, cam_h = 640, 480
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(cam_w))
    camera_bp.set_attribute('image_size_y', str(cam_h))

    cameras = []
    frames = {}

    def callback(data, name):
        frame_id = data.frame
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = array.reshape((cam_h, cam_w, 4))[:, :, :3]
        if frame_id not in frames:
            frames[frame_id] = {}
        frames[frame_id][name] = array

    cam_configs = [
        {"name": "front",  "x": 1.8, "y":  0.0, "z": 1.8, "pitch": 0, "yaw":   0},
        {"name": "back",   "x":-2.0, "y":  0.0, "z": 1.8, "pitch": 0, "yaw": 180},
        {"name": "left",   "x": 0.0, "y": -1.0, "z": 1.8, "pitch": 0, "yaw": -90},
        {"name": "right",  "x": 0.0, "y":  1.0, "z": 1.8, "pitch": 0, "yaw":  90},
    ]

    for cfg in cam_configs:
        trans = carla.Transform(
            carla.Location(x=cfg['x'], y=cfg['y'], z=cfg['z']),
            carla.Rotation(pitch=cfg['pitch'], yaw=cfg['yaw'])
        )
        try:
            cam = world.spawn_actor(camera_bp, trans, attach_to=vehicle)
            cam.listen(lambda data, name=cfg['name']: callback(data, name))
            cameras.append(cam)
        except Exception as e:
            print(f"生成相机 {cfg['name']} 失败: {e}")

    cv2.namedWindow("Camera Monitor", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # 当前显示的视角，默认显示所有视角
    current_view = 'all'  # 'all', 'front', 'back', 'left', 'right'
    
    # HUD 显示开关
    hud_enabled = True
    
    # 天气/时间系统
    weather_types = ['Clear', 'Cloudy', 'Rain', 'Fog', 'Snow']
    current_weather_index = 0  # 默认晴天
    current_hour = 12  # 默认中午 12 点
    
    # 设置初始天气和时间
    def set_weather_and_time():
        nonlocal current_weather_index, current_hour
        
        # 计算太阳高度角（根据时间）
        sun_altitude = 90 - abs(current_hour - 12) * 15
        
        # 创建天气参数对象
        weather = carla.WeatherParameters()
        
        # 根据天气类型设置参数
        if weather_types[current_weather_index] == 'Clear':
            weather.cloudiness = 0.0
            weather.precipitation = 0.0
            weather.precipitation_deposits = 0.0
            weather.fog_density = 0.0
            weather.wetness = 0.0
        elif weather_types[current_weather_index] == 'Cloudy':
            weather.cloudiness = 80.0
            weather.precipitation = 0.0
            weather.precipitation_deposits = 0.0
            weather.fog_density = 0.0
            weather.wetness = 0.0
        elif weather_types[current_weather_index] == 'Rain':
            weather.cloudiness = 60.0
            weather.precipitation = 40.0
            weather.precipitation_deposits = 40.0
            weather.fog_density = 10.0
            weather.wetness = 60.0
        elif weather_types[current_weather_index] == 'Fog':
            weather.cloudiness = 90.0
            weather.precipitation = 0.0
            weather.precipitation_deposits = 0.0
            weather.fog_density = 85.0
            weather.wetness = 10.0
        elif weather_types[current_weather_index] == 'Snow':
            weather.cloudiness = 70.0
            weather.precipitation = 80.0
            weather.precipitation_deposits = 80.0
            weather.fog_density = 20.0
            weather.wetness = 70.0
        
        # 根据时间调整太阳角度
        weather.sun_altitude_angle = sun_altitude
        weather.sun_azimuth_angle = 0
        
        world.set_weather(weather)
    
    # 获取当前天气名称
    def get_weather_name():
        return weather_types[current_weather_index]
    
    # 设置初始天气和时间
    set_weather_and_time()
    
    # 速度限制警告系统
    speed_limit = 60  # 默认限速 60 km/h
    speed_warning_enabled = True
    
    # 更新导航信息
    def update_navigation():
        nonlocal next_waypoint, nav_enabled, destination
        if not nav_enabled or destination is None:
            return None
        
        current_loc = vehicle.get_location()
        distance_to_dest = current_loc.distance(destination)
        
        # 到达目的地检测（距离小于5米）
        if distance_to_dest < 5:
            print("已到达目的地！")
            init_navigation()  # 重新设置新目的地
            return None
        
        # 计算方向（Carla坐标系：X向东，Y向北）
        dx = destination.x - current_loc.x  # 东方向
        dy = destination.y - current_loc.y  # 北方向
        
        # 获取车辆朝向（yaw：0=北，90=东，180=南，270=西）
        vehicle_yaw = vehicle.get_transform().rotation.yaw
        
        # 计算目标方向角度（0=东，90=北，180=西，270=南）
        target_angle = np.degrees(np.arctan2(dy, dx))  # arctan2(y, x) -> 从X轴正方向逆时针角度
        
        # 将目标角度转换为Carla坐标系（0=北，90=东）
        target_angle_carla = 90 - target_angle
        
        # 计算角度差
        angle_diff = target_angle_carla - vehicle_yaw
        # 归一化到 -180 到 180
        angle_diff = (angle_diff + 180) % 360 - 180
        
        return {
            'distance': round(distance_to_dest, 1),
            'direction': get_direction_text(angle_diff),
            'angle_diff': round(angle_diff, 1)
        }
    
    # 获取方向文本
    def get_direction_text(angle_diff):
        if angle_diff < -45:
            return "LEFT"
        elif angle_diff > 45:
            return "RIGHT"
        else:
            return "STRAIGHT"
    
    # 获取车辆状态数据
    def get_vehicle_data():
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # m/s -> km/h
        
        control = vehicle.get_control()
        throttle = control.throttle
        steer = control.steer
        brake = control.brake
        reverse = control.reverse
        
        # 确定档位
        if reverse:
            gear = "R"
        elif throttle > 0.1:
            gear = "D"
        elif brake > 0.1:
            gear = "B"
        else:
            gear = "N"
        
        # 刹车状态
        brake_status = "ON" if brake > 0.1 else "OFF"
        
        return {
            'speed': round(speed, 1),
            'throttle': round(throttle * 100, 0),
            'steer': round(steer * 100, 0),
            'gear': gear,
            'brake_status': brake_status
        }
    
    # 在图像上绘制 HUD
    def draw_hud(image):
        if not hud_enabled:
            return image
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (0, 255, 0)  # 绿色
        bg_color = (0, 0, 0)      # 黑色背景
        
        data = get_vehicle_data()
        
        # 速度警告检测
        speed_warning = False
        speed_warning_color = text_color
        if speed_warning_enabled and data['speed'] > speed_limit:
            speed_warning = True
            speed_warning_color = (0, 0, 255)  # 红色警告
        
        # 获取当前时间（模拟）
        current_time_str = f"{current_hour:02d}:00"
        
        hud_lines = [
            f"Speed: {data['speed']} km/h",
            f"Limit: {speed_limit} km/h",
            f"Throttle: {data['throttle']}%",
            f"Steer: {data['steer']}%",
            f"Gear: {data['gear']}",
            f"Brake: {data['brake_status']}",
            f"Weather: {get_weather_name()}",
            f"Time: {current_time_str}"
        ]
        
        # 添加导航信息
        nav_data = update_navigation()
        if nav_data:
            nav_lines = [
                f"[NAV] Distance: {nav_data['distance']} m",
                f"[NAV] Direction: {nav_data['direction']}",
                f"[NAV] Angle: {nav_data['angle_diff']}°"
            ]
            hud_lines.extend(nav_lines)
        elif nav_enabled:
            hud_lines.append("[NAV] Calculating...")
        
        start_x = 10
        start_y = 30
        line_height = 25
        
        for i, line in enumerate(hud_lines):
            y = start_y + i * line_height
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            cv2.rectangle(image, (start_x - 5, y - text_height - 5), 
                         (start_x + text_width + 5, y + 5), bg_color, -1)
            # 速度超过限制时使用红色警告
            current_color = speed_warning_color if i == 0 else text_color
            # 导航信息使用蓝色
            if "[NAV]" in line:
                current_color = (255, 0, 0)  # 红色导航
            cv2.putText(image, line, (start_x, y), font, font_scale, current_color, font_thickness)
        
        # 绘制导航箭头指示
        if nav_data and hud_enabled:
            draw_navigation_arrow(image, nav_data['direction'])
        
        return image
    
    # 绘制导航箭头
    def draw_navigation_arrow(image, direction):
        height, width = image.shape[:2]
        arrow_center_x = width // 2
        arrow_center_y = height - 40
        
        # 箭头大小
        arrow_size = 15
        
        # 清除箭头区域背景
        cv2.rectangle(image, (arrow_center_x - 35, arrow_center_y - 25), 
                     (arrow_center_x + 35, arrow_center_y + 20), (0, 0, 0), -1)
        
        # 绘制箭头底座（指向屏幕前方）
        base_size = 8
        cv2.line(image, 
                 (arrow_center_x - base_size, arrow_center_y + 10), 
                 (arrow_center_x + base_size, arrow_center_y + 10), 
                 (255, 255, 255), 2)
        
        # 根据方向绘制箭头（相对于车辆前方）
        if direction == "LEFT":
            # 向左箭头（从底座向左上方）
            pts = np.array([[arrow_center_x - base_size, arrow_center_y + 10],
                           [arrow_center_x - base_size - arrow_size, arrow_center_y + 10 - arrow_size],
                           [arrow_center_x - base_size + 5, arrow_center_y + 10]], np.int32)
            cv2.polylines(image, [pts], False, (0, 255, 0), 2)
            cv2.putText(image, "LEFT", (arrow_center_x - 60, arrow_center_y + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif direction == "RIGHT":
            # 向右箭头（从底座向右上方）
            pts = np.array([[arrow_center_x + base_size, arrow_center_y + 10],
                           [arrow_center_x + base_size + arrow_size, arrow_center_y + 10 - arrow_size],
                           [arrow_center_x + base_size - 5, arrow_center_y + 10]], np.int32)
            cv2.polylines(image, [pts], False, (0, 255, 0), 2)
            cv2.putText(image, "RIGHT", (arrow_center_x + 15, arrow_center_y + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # 直行箭头（向上）
            pts = np.array([[arrow_center_x, arrow_center_y + 10],
                           [arrow_center_x, arrow_center_y + 10 - arrow_size],
                           [arrow_center_x - 5, arrow_center_y + 10]], np.int32)
            cv2.polylines(image, [pts], False, (0, 255, 0), 2)
            pts = np.array([[arrow_center_x, arrow_center_y + 10],
                           [arrow_center_x, arrow_center_y + 10 - arrow_size],
                           [arrow_center_x + 5, arrow_center_y + 10]], np.int32)
            cv2.polylines(image, [pts], False, (0, 255, 0), 2)
            cv2.putText(image, "STRAIGHT", (arrow_center_x - 35, arrow_center_y + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    try:
        while True:
            world.tick()
            current_frame = world.get_snapshot().frame
            update_spectator()

            # 查找最新完整帧
            latest = None
            for fid in sorted(frames.keys(), reverse=True):
                if len(frames[fid]) == 4:
                    latest = fid
                    break

            if latest is None:
                key = cv2.waitKey(1)
                if key == 27:
                    break
                continue

            # 取数据
            fdata = frames[latest]

            # 根据当前视角显示
            if current_view == 'all':
                # 2x2 拼接
                top    = np.hstack((fdata['front'], fdata['back']))
                bottom = np.hstack((fdata['left'], fdata['right']))
                display_frame = np.vstack((top, bottom))
            else:
                # 显示单个视角
                display_frame = fdata[current_view]

            # 绘制 HUD
            display_frame = draw_hud(display_frame)

            cv2.imshow("Camera Monitor", display_frame)
            
            # 按键检测
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('1'):
                current_view = 'front'
                print("切换到前视角")
            elif key == ord('2'):
                current_view = 'back'
                print("切换到后视角")
            elif key == ord('3'):
                current_view = 'left'
                print("切换到左视角")
            elif key == ord('4'):
                current_view = 'right'
                print("切换到右视角")
            elif key == ord('0'):
                current_view = 'all'
                print("切换到四视角模式")
            elif key == ord('h') or key == ord('H'):
                hud_enabled = not hud_enabled
                print(f"HUD 显示 {'开启' if hud_enabled else '关闭'}")
            elif key == ord('+') or key == ord('='):
                speed_limit += 10
                print(f"限速提高到: {speed_limit} km/h")
            elif key == ord('-') or key == ord('_'):
                speed_limit = max(10, speed_limit - 10)
                print(f"限速降低到: {speed_limit} km/h")
            elif key == ord('w') or key == ord('W'):
                speed_warning_enabled = not speed_warning_enabled
                print(f"速度警告 {'开启' if speed_warning_enabled else '关闭'}")
            elif key == ord('t') or key == ord('T'):
                camera_mode = 'top'
                print("切换到俯视视角")
            elif key == ord('f') or key == ord('F'):
                camera_mode = 'follow'
                print("切换到第三人称跟随视角")
            elif key == ord('c') or key == ord('C'):
                camera_mode = 'chase'
                print("切换到追逐视角")
            elif key == ord('s') or key == ord('S'):
                camera_mode = 'side'
                print("切换到侧视视角")
            elif key == ord('n') or key == ord('N'):
                init_navigation()
            elif key == ord('v') or key == ord('V'):
                # 切换天气
                current_weather_index = (current_weather_index + 1) % len(weather_types)
                set_weather_and_time()
                print(f"天气切换到：{get_weather_name()}")
            elif key == ord('u') or key == ord('U'):
                # 切换时间（小时）
                current_hour = (current_hour + 3) % 24
                set_weather_and_time()
                print(f"时间切换到：{current_hour:02d}:00")

            # 清理旧帧
            max_to_keep = 5
            for fid in list(frames.keys()):
                if fid < latest - max_to_keep:
                    del frames[fid]

    finally:
        print("清理资源...")
        for cam in cameras:
            cam.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()