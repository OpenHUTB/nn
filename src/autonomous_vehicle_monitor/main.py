import carla
import cv2
import numpy as np
from core.sensors import Sensors
from core.npc_manager import NpcManager
from core.recorder import Recorder
from core.player import Player
from core.blackbox import Blackbox
from core.map_drawer import MapDrawer
from core.ui_dashboard import VirtualDashboard
from core.traffic_light_monitor import TrafficLightMonitor
from core.kpi_evaluator import KPIEvaluator  # <--- 导入KPI模块

def main():
    # ====================== 连接CARLA ======================
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

    # ====================== 生成自车 ======================
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

    # ====================== 天气系统 ======================
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
        world.set_weather(w)

    def set_fog():
        w = carla.WeatherParameters.ClearNoon
        w.fog_density = 70
        world.set_weather(w)

    set_day()

    # ====================== FCW 前车碰撞预警 ======================
    warn_dist = 10.0
    danger_dist = 5.0
    front_dist = 999.0
    alert_text = ""
    flash_cnt = 0

    def fcw_detect():
        nonlocal front_dist, alert_text, flash_cnt
        ego_loc = vehicle.get_location()
        ego_yaw = vehicle.get_transform().rotation.yaw
        min_front_dist = 999.0

        for actor in world.get_actors():
            if not actor.is_alive or actor.id == vehicle.id:
                continue
            if "vehicle" not in actor.type_id:
                continue

            dx = actor.get_location().x - ego_loc.x
            dy = actor.get_location().y - ego_loc.y
            dist = np.hypot(dx, dy)
            if dist > 30:
                continue

            yaw_rad = np.radians(ego_yaw)
            dot = dx * np.cos(yaw_rad) + dy * np.sin(yaw_rad)
            if dot > 0 and dist < min_front_dist:
                min_front_dist = dist

        front_dist = min_front_dist

        if min_front_dist < danger_dist:
            alert_text = f"FCW DANGER! {min_front_dist:.1f}m"
            flash_cnt += 1
        elif min_front_dist < warn_dist:
            alert_text = f"FCW WARNING! {min_front_dist:.1f}m"
            flash_cnt += 1
        else:
            alert_text = ""
            flash_cnt = 0

    # ====================== 初始化所有模块 ======================
    npc_manager = NpcManager(world, bp_lib, spawn_points)
    npc_manager.spawn_all()
    sensors = Sensors(world, vehicle)
    sensors.setup_all()
    recorder = Recorder()
    player = None
    blackbox = Blackbox()
    map_drawer = MapDrawer(world, vehicle)
    dash = VirtualDashboard()
    light_monitor = TrafficLightMonitor(world, vehicle)
    kpi = KPIEvaluator()  # <--- KPI模块初始化

    is_recording = False
    is_playing = False
    last_in_lane = True

    cv2.namedWindow("AD Monitor", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Dashboard", cv2.WINDOW_NORMAL)

    # ====================== 主循环 ======================
    try:
        while True:
            world.tick()
            update_view()
            key = cv2.waitKey(1) & 0xFF

            # 前车检测
            fcw_detect()

            # 车道保持判断
            try:
                wp = world.get_map().get_waypoint(vehicle.get_location(), project_to_road=True)
                is_in_lane = vehicle.get_location().distance(wp.transform.location) < 3.5
                last_in_lane = is_in_lane
            except:
                is_in_lane = last_in_lane

            # 更新KPI
            kpi.update(vehicle, front_dist, is_in_lane)

            blackbox.record(vehicle)
            light_monitor.update()

            # 天气按键
            if key == ord('n'):
                set_night() if not is_night else set_day()
            if key == ord('r'):
                set_rain()
            if key == ord('f'):
                set_fog()
            if key == ord('c'):
                set_day()

            # 录制回放
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

            # ====================== 画面渲染 ======================
            if len(sensors.frame_dict) >= 4 and sensors.lidar_data is not None:
                f, b, l, r = sensors.frame_dict.values()
                cam_mosaic = cv2.resize(np.vstack((np.hstack((f, b)), np.hstack((l, r)))), (1280, 960))

                bev = np.zeros((960, 640, 3), np.uint8)
                cx, cy, scale = 320, 480, 10
                cv2.circle(bev, (cx, cy), 8, (0, 255, 0), -1)

                for x, y, z in sensors.lidar_data:
                    if abs(x) > 45 or abs(y) > 45:
                        continue
                    px, py = int(cx + y * scale), int(cy - x * scale)
                    if 0 <= px < 640 and 0 <= py < 960:
                        bev[py, px] = 255, 255, 255

                for npc in npc_manager.vehicles:
                    try:
                        dx = npc.get_location().x - vehicle.get_location().x
                        dy = npc.get_location().y - vehicle.get_location().y
                        if abs(dx) > 45:
                            continue
                        cv2.circle(bev, (int(cx + dy * scale), int(cy - dx * scale)), 5, (0, 0, 255), -1)
                    except:
                        continue

                map_drawer.draw_lanes_and_drivable_area(bev)
                full = np.hstack((cam_mosaic, bev))

                # FCW 闪烁报警
                if alert_text and flash_cnt % 12 < 6:
                    color = (0, 0, 255) if "DANGER" in alert_text else (0, 165, 255)
                    cv2.putText(full, alert_text, (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

                # 显示 KPI
                kpi.draw_on_screen(full)

                cv2.putText(full, "N=夜间 R=雨 F=雾 C=晴 | R=录制", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow("AD Monitor", full)

            # 仪表盘
            dash_img = dash.render(vehicle)
            light_bar = np.zeros((60, 320, 3), np.uint8)
            light_monitor.render(light_bar, 100, 10)
            cv2.imshow("Dashboard", np.vstack([light_bar, dash_img]))

            if key == 27:
                break

    finally:
        kpi.save_report()  # <--- 退出时自动保存KPI报告
        print("\n✅ KPI 报告已保存：kpi_report.txt")

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