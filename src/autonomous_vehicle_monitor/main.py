import carla
import cv2
import numpy as np
from core.recorder import Recorder
from core.player import Player
from core.npc_manager import NpcManager
from core.sensors import Sensors
from core.blackbox import BlackBox
from core.map_drawer import MapDrawer
from core.ui_dashboard import VirtualDashboard
from core.traffic_light_monitor import TrafficLightMonitor


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    world = client.get_world()
    tm = client.get_trafficmanager()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    vehicle = None
    for spawn in np.random.permutation(spawn_points):
        try:
            vehicle = world.spawn_actor(bp_lib.filter('vehicle.*model3*')[0], spawn)
            break
        except:
            continue
    if not vehicle:
        return

    vehicle.set_autopilot(True)
    spectator = world.get_spectator()

    def update_view():
        t = vehicle.get_transform()
        spectator.set_transform(carla.Transform(
            t.location + carla.Location(z=20),
            carla.Rotation(pitch=-90, yaw=t.rotation.yaw)
        ))

    # ====================== 天气控制函数 ======================
    weather = carla.WeatherParameters.ClearNoon
    is_night = False

    def set_daytime():
        nonlocal is_night
        is_night = False
        world.set_weather(carla.WeatherParameters.ClearNoon)

    def set_nighttime():
        nonlocal is_night
        is_night = True
        world.set_weather(carla.WeatherParameters.ClearNight)

    def set_rain():
        world.set_weather(carla.WeatherParameters.WetNoon)
        weather.rain = 100
        weather.precipitation = 100
        weather.precipitation_deposits = 100
        world.set_weather(weather)

    def set_fog():
        w = carla.WeatherParameters.ClearNoon
        w.fog_density = 80
        w.fog_distance = 20
        world.set_weather(w)

    set_daytime()  # 默认白天

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

    cv2.namedWindow("AD Monitor", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Dashboard", cv2.WINDOW_NORMAL)

    try:
        while True:
            world.tick()
            update_view()
            key = cv2.waitKey(1) & 0xFF

            blackbox.record(vehicle)
            light_monitor.update()

            # ====================== 天气切换按键 ======================
            if key == ord('n'):
                if not is_night:
                    set_nighttime()
                    print("🌙 已切换：夜间模式")
                else:
                    set_daytime()
                    print("☀️ 已切换：白天模式")

            if key == ord('r'):
                set_rain()
                print("🌧️ 已切换：雨天")

            if key == ord('f'):
                set_fog()
                print("🌫️ 已切换：雾天")

            if key == ord('c'):
                set_daytime()
                print("☀️ 已清空天气：晴天")

            # 录制 / 回放
            if key == ord('r') and not is_recording:
                is_recording = True
                recorder.start()
                print("🔴 开始录制")
            if key == ord('s'):
                is_recording = False
                recorder.save()
                print("💾 已保存录制")
            if key == ord('p'):
                vehicle.set_autopilot(False)
                for v in npc_manager.vehicles:
                    v.set_autopilot(False)
                player = Player(world, vehicle, npc_manager.vehicles + npc_manager.walkers)
                player.load()
                is_playing = True
                print("▶️ 开始回放")

            if is_recording:
                recorder.record_frame(vehicle, npc_manager.vehicles + npc_manager.walkers)
            if is_playing and player:
                if not player.play_frame():
                    is_playing = False
                    print("✅ 回放完成")

            # 主窗口
            if len(sensors.frame_dict) >= 4 and sensors.lidar_data is not None:
                f, b, l, r = sensors.frame_dict.values()
                cam_mosaic = cv2.resize(np.vstack((np.hstack((f, b)), np.hstack((l, r)))), (1280, 960))

                bev = np.zeros((960, 640, 3), np.uint8)
                cx, cy, scale = 320, 480, 10
                cv2.circle(bev, (cx, cy), 8, (0, 255, 0), -1)

                for x, y, z in sensors.lidar_data:
                    if abs(x) > 45 or abs(y) > 45: continue
                    px, py = int(cx + y * scale), int(cy - x * scale)
                    if 0 <= px < 640 and 0 <= py < 960:
                        bev[py, px] = 255, 255, 255

                for npc in npc_manager.vehicles:
                    try:
                        dx = npc.get_location().x - vehicle.get_location().x
                        dy = npc.get_location().y - vehicle.get_location().y
                        if abs(dx) > 45: continue
                        cv2.circle(bev, (int(cx + dy * scale), int(cy - dx * scale)), 5, (0, 0, 255), -1)
                    except:
                        continue

                map_drawer.draw_lanes_and_drivable_area(bev)
                full_view = np.hstack((cam_mosaic, bev))

                # 显示天气提示
                weather_text = "N=night R=rain F=fog C=clear  "
                cv2.putText(full_view, weather_text + "R=Rec S=Save P=Play ESC=Exit",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                cv2.imshow("AD Monitor", full_view)

            # 独立仪表盘
            dash_img = dash.render(vehicle)
            light_bar = np.zeros((60, 320, 3), dtype=np.uint8)
            light_monitor.render(light_bar, 100, 10)
            dashboard_full = np.vstack([light_bar, dash_img])
            cv2.imshow("Dashboard", dashboard_full)

            if key == 27:
                break

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