import carla
import random
import cv2
import numpy as np
import json
import time

# ======================
# 录制模块
# ======================
class Recorder:
    def __init__(self):
        self.frames = []
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        self.frames = []

    def record_frame(self, vehicle, npcs):
        data = {
            "time": time.time() - self.start_time,
            "ego": {
                "x": vehicle.get_transform().location.x,
                "y": vehicle.get_transform().location.y,
                "z": vehicle.get_transform().location.z,
                "pitch": vehicle.get_transform().rotation.pitch,
                "yaw": vehicle.get_transform().rotation.yaw,
                "roll": vehicle.get_transform().rotation.roll,
            },
            "npcs": []
        }
        for npc in npcs:
            if npc.is_alive:
                t = npc.get_transform()
                data["npcs"].append({
                    "id": npc.id,
                    "x": t.location.x,
                    "y": t.location.y,
                    "z": t.location.z,
                    "pitch": t.rotation.pitch,
                    "yaw": t.rotation.yaw,
                    "roll": t.rotation.roll
                })
        self.frames.append(data)

    def save(self, path="recording.json"):
        with open(path, "w") as f:
            json.dump(self.frames, f)
        print(f"✅ 录制完成：{path}")

# ======================
# 回放模块
# ======================
class Player:
    def __init__(self, world, vehicle, npcs):
        self.world = world
        self.ego = vehicle
        self.npc_dict = {n.id: n for n in npcs}
        self.frames = []
        self.index = 0

    def load(self, path="recording.json"):
        with open(path, "r") as f:
            self.frames = json.load(f)
        print(f"✅ 回放加载完成，共 {len(self.frames)} 帧")

    def play_frame(self):
        if self.index >= len(self.frames):
            return False
        data = self.frames[self.index]
        t = carla.Transform(
            carla.Location(data["ego"]["x"], data["ego"]["y"], data["ego"]["z"]),
            carla.Rotation(data["ego"]["pitch"], data["ego"]["yaw"], data["ego"]["roll"])
        )
        self.ego.set_transform(t)
        for npc_data in data["npcs"]:
            nid = npc_data["id"]
            if nid in self.npc_dict:
                npc = self.npc_dict[nid]
                t = carla.Transform(
                    carla.Location(npc_data["x"], npc_data["y"], npc_data["z"]),
                    carla.Rotation(npc_data["pitch"], npc_data["yaw"], npc_data["roll"])
                )
                npc.set_transform(t)
        self.index += 1
        return True

# ======================
# 主程序
# ======================
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    tm = client.get_trafficmanager()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.*model3*')[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = None

    for spawn in random.sample(spawn_points, len(spawn_points)):
        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn)
            break
        except:
            continue
    if not vehicle:
        return

    vehicle.set_autopilot(True)
    spectator = world.get_spectator()

    def update_top_view():
        trans = vehicle.get_transform()
        camera_loc = trans.location + carla.Location(z=20)
        camera_rot = carla.Rotation(pitch=-90, yaw=trans.rotation.yaw)
        spectator.set_transform(carla.Transform(camera_loc, camera_rot))

    # ======================
    # 生成 NPC
    # ======================
    vehicle_list = []
    walker_list = []
    all_actors = []

    # 车辆
    for _ in range(15):
        v_bp = random.choice(bp_lib.filter('vehicle.*'))
        spawn = random.choice(spawn_points)
        try:
            npc = world.spawn_actor(v_bp, spawn)
            npc.set_autopilot(True)
            vehicle_list.append(npc)
            all_actors.append(npc)
        except:
            continue

    # 行人
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    for _ in range(20):
        try:
            loc = world.get_random_location_from_navigation()
            if not loc: continue
            walker = world.try_spawn_actor(random.choice(walker_bps), carla.Transform(loc))
            if walker:
                walker_list.append(walker)
                all_actors.append(walker)
        except:
            continue

    # 行人控制器
    controller_bp = bp_lib.find('controller.ai.walker')
    for walker in walker_list:
        try:
            ctrl = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            all_actors.append(ctrl)
            ctrl.start()
            ctrl.go_to_location(world.get_random_location_from_navigation())
        except:
            continue

    # ======================
    # 4路相机
    # ======================
    cam_w, cam_h = 640, 480
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(cam_w))
    camera_bp.set_attribute('image_size_y', str(cam_h))
    cameras = []
    frame_dict = {}

    def callback(data, name):
        arr = np.frombuffer(data.raw_data, dtype=np.uint8)
        arr = arr.reshape((cam_h, cam_w, 4))[:, :, :3]
        frame_dict[name] = arr

    cam_configs = [
        {"name": "front",  "x":1.8,"y":0,"z":1.8,"pitch":0,"yaw":0},
        {"name": "back",   "x":-2,"y":0,"z":1.8,"pitch":0,"yaw":180},
        {"name": "left",   "x":0,"y":-1,"z":1.8,"pitch":0,"yaw":-90},
        {"name": "right",  "x":0,"y":1,"z":1.8,"pitch":0,"yaw":90},
    ]
    for cfg in cam_configs:
        trans = carla.Transform(
            carla.Location(x=cfg['x'], y=cfg['y'], z=cfg['z']),
            carla.Rotation(pitch=cfg['pitch'], yaw=cfg['yaw'])
        )
        cam = world.spawn_actor(camera_bp, trans, attach_to=vehicle)
        cam.listen(lambda data, name=cfg["name"]: callback(data, name))
        cameras.append(cam)

    # ======================
    # 激光雷达
    # ======================
    lidar_data = None
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '50')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_bp.set_attribute('channels', '64')

    lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2.0))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    def lidar_callback(data):
        nonlocal lidar_data
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4')).reshape(-1, 4)
        lidar_data = points[:, :3]

    lidar.listen(lidar_callback)
    cameras.append(lidar)

    # ======================
    # 录制回放
    # ======================
    recorder = Recorder()
    player = None
    is_recording = False
    is_playing = False

    cv2.namedWindow("AD Monitor", cv2.WINDOW_NORMAL)

    try:
        while True:
            world.tick()
            update_top_view()

            
            key = cv2.waitKey(1) & 0xFF

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
                for n in vehicle_list:
                    n.set_autopilot(False)
                player = Player(world, vehicle, vehicle_list + walker_list)
                player.load()
                is_playing = True
                print("▶️ 开始回放")

            # 录制逻辑
            if is_recording:
                recorder.record_frame(vehicle, vehicle_list + walker_list)

            # 回放逻辑
            if is_playing and player:
                if not player.play_frame():
                    is_playing = False
                    print("✅ 回放完成")

            # 画面显示
            if len(frame_dict) >= 4 and lidar_data is not None:
                front = frame_dict["front"]
                back = frame_dict["back"]
                left = frame_dict["left"]
                right = frame_dict["right"]

                cam_mosaic = np.vstack((
                    np.hstack((front, back)),
                    np.hstack((left, right))
                ))
                cam_mosaic = cv2.resize(cam_mosaic, (1280, 960))

                bev_width = 640
                bev_height = 960
                bev = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
                cx, cy, scale = bev_width//2, bev_height//2, 10

                cv2.circle(bev, (cx, cy), 8, (0, 255, 0), -1)

                for x, y, z in lidar_data:
                    if abs(x) > 45 or abs(y) > 45:
                        continue
                    px = int(cx + y * scale)
                    py = int(cy - x * scale)
                    if 0 <= px < bev_width and 0 <= py < bev_height:
                        bev[py, px] = (255, 255, 255)

                for npc in vehicle_list:
                    try:
                        dx = npc.get_location().x - vehicle.get_location().x
                        dy = npc.get_location().y - vehicle.get_location().y
                        if abs(dx) > 45 or abs(dy) > 45:
                            continue
                        px = int(cx + dy * scale)
                        py = int(cy - dx * scale)
                        if 0 <= px < bev_width and 0 <= py < bev_height:
                            cv2.circle(bev, (px, py), 5, (0, 0, 255), -1)
                    except:
                        continue

                full = np.hstack((cam_mosaic, bev))
                cv2.putText(full, "R=Record  S=Save  P=Play", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("AD Monitor", full)

            if key == 27:
                break

    finally:
        for a in all_actors:
            try:
                if a.is_alive: a.destroy()
            except: pass
        for c in cameras:
            try:
                if c.is_alive: c.destroy()
            except: pass
        try:
            if vehicle.is_alive: vehicle.destroy()
        except: pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()