import carla
import cv2
import numpy as np
from core.sensors import Sensors
from core.npc_manager import NpcManager
from core.recorder import Recorder
from core.player import Player
from core.blackbox import BlackBox
from core.map_drawer import MapDrawer
from core.ui_dashboard import VirtualDashboard
from core.traffic_light_monitor import TrafficLightMonitor

def main():
    # ====================== 1. 连接CARLA模拟器 ======================
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    world = client.get_world()
    tm = client.get_trafficmanager()

    # 开启同步模式，保证仿真流畅稳定
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)

    # 获取车辆蓝图和出生点
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # ====================== 2. 生成自车（特斯拉Model3） ======================
    vehicle = None
    for spawn in np.random.permutation(spawn_points):
        try:
            vehicle = world.spawn_actor(bp_lib.filter('vehicle.*model3*')[0], spawn)
            break
        except:
            continue
    if not vehicle:
        return

    # 开启自动驾驶
    vehicle.set_autopilot(True)
    spectator = world.get_spectator()

    # 俯视视角跟随车辆
    def update_view():
        t = vehicle.get_transform()
        spectator.set_transform(carla.Transform(
            t.location + carla.Location(z=20),
            carla.Rotation(pitch=-90, yaw=t.rotation.yaw)
        ))

    # ====================== 3. 天气系统：白天/夜间/雨天/雾天 ======================
    weather = carla.WeatherParameters.ClearNoon
    is_night = False

    def set_day():
        nonlocal is_night
        is_night = False
        world.set_weather(carla.WeatherParameters.ClearNoon)

    def set_night():
        nonlocal is_night
        is_night = True
        world.set_weather(carla.WeatherParameters.ClearNight)

    def set_rain():
        w = carla.WeatherParameters.WetNoon
        w.rain = 100
        w.precipitation = 100
        world.set_weather(w)

    def set_fog():
        w = carla.WeatherParameters.ClearNoon
        w.fog_density = 70
        w.fog_distance = 25
        world.set_weather(w)

    set_day()  # 默认晴天白天

    # ====================== 4. 前车碰撞预警 FCW 功能 ======================
    safe_dist = 15.0
    warn_dist = 10.0
    danger_dist = 5.0
    flash_count = 0
    alert_text = ""

    def fcw_detect():
        nonlocal flash_count, alert_text
        ego_loc = vehicle.get_location()
        ego_rot = vehicle.get_transform().rotation
        min_front_dist = 999.0

        # 遍历所有车辆，寻找正前方最近车辆
        for actor in world.get_actors():
            if not actor.is_alive or actor.id == vehicle.id:
                continue
            if "vehicle" not in actor.type_id:
                continue

            act_loc = actor.get_location()
            dist = ego_loc.distance(act_loc)
            if dist > 30:
                continue

            # 计算是否在车辆前方
            dx = act_loc.x - ego_loc.x
            dy = act_loc.y - ego_loc.y
            yaw_rad = np.radians(ego_rot.yaw)
            forward_x = np.cos(yaw_rad)
            forward_y = np.sin(yaw_rad)
            dot = dx * forward_x + dy * forward_y

            if dot > 0 and dist < min_front_dist:
                min_front_dist = dist

        # 分级预警：安全 / 警告 / 危险
        if min_front_dist < danger_dist:
            alert_text = f"FCW DANGER! Distance: {min_front_dist:.1f}m"
            flash_count += 1
        elif min_front_dist < warn_dist:
            alert_text = f"FCW WARNING! Distance: {min_front_dist:.1f}m"
            flash_count += 1
        else:
            alert_text = ""
            flash_count = 0

        return alert_text, flash_count

    # ====================== 5. 初始化所有模块 ======================
    npc_manager = NpcManager(world, bp_lib, spawn_points)
    npc_manager.spawn_all()

    sensors = Sensors(world, vehicle)
    sensors.setup_all()

    recorder = Recorder()
    player = None
    blackbox = BlackBox()
    map_drawer = MapDrawer(world, vehicle)

    dash = VirtualDashboard()
    light_monitor = TrafficLightMonitor(world, vehicle)

    is_recording = False
    is_playing = False

    # 创建显示窗口
    cv2.namedWindow("AD Monitor", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Dashboard", cv2.WINDOW_NORMAL)

    # ====================== 6. 主循环 ======================
    try:
        while True:
            world.tick()              # 仿真步进
            update_view()             # 更新俯视视角
            key = cv2.waitKey(1) & 0xFF

            # 前车碰撞预警检测
            fcw_msg, flash_cnt = fcw_detect()

            # 记录数据 + 红绿灯状态更新
            blackbox.record(vehicle)
            light_monitor.update()

            # ====================== 按键：天气切换 ======================
            if key == ord('n'):
                set_night() if not is_night else set_day()
                print("夜间模式" if not is_night else "白天模式")
            if key == ord('r'):
                set_rain()
                print("雨天")
            if key == ord('f'):
                set_fog()
                print("雾天")
            if key == ord('c'):
                set_day()
                print("晴天")

            # ====================== 按键：录制/回放 ======================
            if key == ord('r') and not is_recording:
                is_recording = True
                recorder.start()
            if key == ord('s'):
                is_recording = False
                recorder.save()
            if key == ord('p'):
                vehicle.set_autopilot(False)
                for v in npc_manager.vehicles:
                    v.set_autopilot(False)
                player = Player(world, vehicle, npc_manager.vehicles + npc_manager.walkers)
                player.load()
                is_playing = True

            if is_recording:
                recorder.record_frame(vehicle, npc_manager.vehicles, npc_manager.walkers)
            if is_playing and player:
                if not player.play_frame():
                    is_playing = False

            # ====================== 7. 主画面渲染（相机 + BEV） ======================
            if len(sensors.frame_dict) >= 4 and sensors.lidar_data is not None:
                f, b, l, r = sensors.frame_dict.values()
                cam_mosaic = cv2.resize(np.vstack((np.hstack((f,b)), np.hstack((l,r)))), (1280,960))

                # 绘制BEV鸟瞰图 + 激光雷达点云
                bev = np.zeros((960,640,3), np.uint8)
                cx, cy, scale = 320,480,10
                cv2.circle(bev, (cx,cy),8,(0,255,0),-1)  # 自车绿色点

                # 绘制雷达点云
                for x,y,z in sensors.lidar_data:
                    if abs(x)>45 or abs(y)>45: continue
                    px,py = int(cx+y*scale), int(cy-x*scale)
                    if 0<=px<640 and 0<=py<960:
                        bev[py,px] = 255,255,255

                # 绘制其他NPC车辆（红色点）
                for npc in npc_manager.vehicles:
                    try:
                        dx = npc.get_location().x - vehicle.get_location().x
                        dy = npc.get_location().y - vehicle.get_location().y
                        if abs(dx)>45:continue
                        cv2.circle(bev, (int(cx+dy*scale), int(cy-dx*scale)),5,(0,0,255),-1)
                    except:
                        continue

                # 绘制车道线与可行驶区域
                map_drawer.draw_lanes_and_drivable_area(bev)
                full = np.hstack((cam_mosaic, bev))

                # ====================== FCW预警文字闪烁 ======================
                if fcw_msg:
                    if flash_cnt % 12 < 6:
                        txt_color = (0,0,255) if "DANGER" in fcw_msg else (0,165,255)
                        cv2.putText(full, fcw_msg, (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, txt_color, 3)

                # 显示提示文字
                cv2.putText(full,"N=夜间 R=雨 F=雾 C=晴 | R=录制",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
                cv2.imshow("AD Monitor", full)

            # ====================== 8. 仪表盘窗口 ======================
            dash_img = dash.render(vehicle)
            light_bar = np.zeros((60,320,3),np.uint8)
            light_monitor.render(light_bar,100,10)
            cv2.imshow("Dashboard", np.vstack([light_bar, dash_img]))

            # ESC退出
            if key == 27:
                break

    # ====================== 9. 退出时释放资源 ======================
    finally:
        blackbox.close()
        try: npc_manager.destroy_all() 
        except: pass
        try: sensors.destroy() 
        except: pass
        try:
            if vehicle.is_alive:
                vehicle.destroy()
        except: pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()