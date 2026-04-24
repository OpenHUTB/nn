import carla
import cv2
import numpy as np
from recorder import Recorder
from player import Player
from npc_manager import NpcManager
from sensors import Sensors
from blackbox import BlackBox
from map_drawer import MapDrawer
from ui_dashboard import VirtualDashboard
from traffic_light_monitor import TrafficLightMonitor


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

            if key == ord('r'):
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
                cv2.putText(full_view, "R=Rec S=Save P=Play ESC=Exit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
                cv2.imshow("AD Monitor", full_view)

            # ================== 独立仪表盘窗口（红绿灯移到顶部，不遮挡） ==================
            dash_img = dash.render(vehicle)

            # 创建一个顶部条专门放红绿灯
            light_bar = np.zeros((60, 320, 3), dtype=np.uint8)
            light_monitor.render(light_bar, 100, 10)  # 居中显示

            # 拼接：顶部红绿灯 + 下面仪表盘
            dashboard_full = np.vstack([light_bar, dash_img])

            cv2.imshow("Dashboard", dashboard_full)

            if key == 27:
                break

    finally:
        blackbox.close()
        try:
            npc_manager.destroy_all()
        except:
            pass
        try:
            sensors.destroy()
        except:
            pass
        try:
            if vehicle.is_alive:
                vehicle.destroy()
        except:
            pass

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()