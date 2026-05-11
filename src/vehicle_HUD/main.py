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

    # 90° 俯视跟随（完全硬跟随）
    def update_top_view():
        trans = vehicle.get_transform()
        loc = trans.location + carla.Location(z=20)
        rot = carla.Rotation(pitch=-90, yaw=trans.rotation.yaw)
        spectator.set_transform(carla.Transform(loc, rot))

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
    
    # 速度限制警告系统
    speed_limit = 60  # 默认限速 60 km/h
    speed_warning_enabled = True
    
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
        
        hud_lines = [
            f"Speed: {data['speed']} km/h",
            f"Limit: {speed_limit} km/h",
            f"Throttle: {data['throttle']}%",
            f"Steer: {data['steer']}%",
            f"Gear: {data['gear']}",
            f"Brake: {data['brake_status']}"
        ]
        
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
            cv2.putText(image, line, (start_x, y), font, font_scale, current_color, font_thickness)
        
        return image

    try:
        while True:
            world.tick()
            current_frame = world.get_snapshot().frame
            update_top_view()

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