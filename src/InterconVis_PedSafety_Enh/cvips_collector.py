import carla
import random
import queue
import os
import argparse
import time
import threading

import json
import numpy as np
import cv2
import cvips_utils as utils

# ================= 配置区域 =================
OUTPUT_FOLDER = "_out_dataset_final"
SAVE_INTERVAL = 5        
TARGET_FPS = 20          
FIXED_DELTA_SECONDS = 1.0 / TARGET_FPS

IMAGE_W = 1280
IMAGE_H = 720
FOV = 90.0
# ===========================================

writing_thread_running = True

def configure_weather(world, weather_arg, time_arg):
    """设置天气和时间的核心逻辑"""
    presets = {
        'Clear': carla.WeatherParameters.ClearNoon,
        'Cloudy': carla.WeatherParameters.CloudyNoon,
        'Wet': carla.WeatherParameters.WetNoon,
        'Rain': carla.WeatherParameters.HardRainNoon,
        'Storm': carla.WeatherParameters.HardRainSunset
    }
    weather = presets.get(weather_arg, carla.WeatherParameters.ClearNoon)
    
    # 修改太阳高度角模拟时间
    if time_arg == 'Day': weather.sun_altitude_angle = 60.0
    elif time_arg == 'Sunset': weather.sun_altitude_angle = 10.0
    elif time_arg == 'Night': weather.sun_altitude_angle = -90.0
    
    world.set_weather(weather)
    print(f"🌦️ 环境已切换: {weather_arg} | {time_arg}")

def is_visible(world, camera_actor, target_actor):
    """射线检测：判断是否被遮挡"""
    cam_trans = camera_actor.get_transform()
    cam_loc = cam_trans.location
    forward = cam_trans.get_forward_vector()
    # 起点偏移防止撞到自己车头
    ray_start = cam_loc + carla.Location(x=forward.x*2.5, y=forward.y*2.5, z=forward.z*2.5)
    
    target_loc = target_actor.get_transform().location
    target_loc.z += 1.0 

    hit_points = world.cast_ray(ray_start, target_loc)
    for hit in hit_points:
        if hit.location.distance(ray_start) < target_loc.distance(ray_start) - 2.0:
            return False
    return True

def draw_box_simple(img, target, w2c, K):
    """绘制3D绿框"""
    loc, rot, extent = target['location'], target['rotation'], target['extent']
    offset = target.get('center_offset', [0, 0, 0])
    obj_t = carla.Transform(carla.Location(x=loc[0], y=loc[1], z=loc[2]),
                            carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]))
    obj_to_world = utils.get_matrix(obj_t)
    dx, dy, dz = extent[0], extent[1], extent[2]
    ox, oy, oz = offset[0], offset[1], offset[2]
    corners_local = np.array([
        [ox+dx, oy+dy, oz+dz, 1], [ox+dx, oy-dy, oz+dz, 1], [ox+dx, oy-dy, oz-dz, 1], [ox+dx, oy+dy, oz-dz, 1],
        [ox-dx, oy+dy, oz+dz, 1], [ox-dx, oy-dy, oz+dz, 1], [ox-dx, oy-dy, oz-dz, 1], [ox-dx, oy+dy, oz-dz, 1]
    ])
    img_pts = []
    points_behind = 0
    for pt in corners_local:
        w_pos = np.dot(obj_to_world, pt)
        p = utils.get_image_point(carla.Location(x=w_pos[0], y=w_pos[1], z=w_pos[2]), K, w2c)
        if p is None: 
            points_behind += 1
            img_pts.append((0,0))
        else: img_pts.append(p)
    if points_behind > 4: return img
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    color = (0, 255, 0) if target['type'] == 'vehicle' else (0, 0, 255)
    for s, e in edges:
        if img_pts[s] != (0,0) and img_pts[e] != (0,0):
            cv2.line(img, tuple(img_pts[s]), tuple(img_pts[e]), color, 2)
    return img

def save_worker(q):
    while writing_thread_running or not q.empty():
        try:
            task = q.get(timeout=0.1)
            path, fid = task['path'], task['fid']
            task['ego_img'].save_to_disk(f"{path}/ego_rgb/{fid:08d}.jpg")
            task['rsu_img'].save_to_disk(f"{path}/rsu_rgb/{fid:08d}.jpg")
            with open(f"{path}/label/{fid:08d}.json", 'w') as f:
                json.dump(task['json_data'], f, indent=2)
            q.task_done()
        except: continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--town', default='Town05')
    parser.add_argument('--weather', default='Clear', choices=['Clear', 'Cloudy', 'Wet', 'Rain', 'Storm'])
    parser.add_argument('--time', default='Day', choices=['Day', 'Sunset', 'Night'])
    parser.add_argument('--num_vehicles', default=50, type=int)
    parser.add_argument('--num_walkers', default=50, type=int) # 行人数量
    parser.add_argument('--max_frames', default=1000, type=int)
    args = parser.parse_args()

    scene_id = f"{args.town}_{args.weather}_{args.time}"
    save_path = os.path.join(OUTPUT_FOLDER, scene_id)
    for sub in ['ego_rgb', 'rsu_rgb', 'label']: os.makedirs(os.path.join(save_path, sub), exist_ok=True)

    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.load_world(args.town)
    
    # 核心：设置天气和同步模式
    configure_weather(world, args.weather, args.time)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    actor_list = []
    try:
        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        
        # 1. 主车
        ego_vehicle = world.spawn_actor(bp_lib.find('vehicle.tesla.model3'), spawn_points[0])
        ego_vehicle.set_autopilot(True, tm.get_port())
        actor_list.append(ego_vehicle)

        # 2. 背景车辆
        for i in range(1, min(len(spawn_points), args.num_vehicles)):
            bp = random.choice(bp_lib.filter('vehicle.*'))
            npc = world.try_spawn_actor(bp, spawn_points[i])
            if npc:
                npc.set_autopilot(True, tm.get_port())
                actor_list.append(npc)

        # 3. 生成行人
        print(f"🚶 正在生成 {args.num_walkers} 个行人...")
        walker_bp_lib = bp_lib.filter('walker.pedestrian.*')
        for _ in range(args.num_walkers):
            loc = world.get_random_location_from_navigation()
            if loc:
                walker_bp = random.choice(walker_bp_lib)
                w = world.try_spawn_actor(walker_bp, carla.Transform(loc))
                if w: actor_list.append(w)

        # 4. 传感器
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(IMAGE_W)); cam_bp.set_attribute('image_size_y', str(IMAGE_H))
        cam_bp.set_attribute('fov', str(FOV))
        ego_cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=2.0, x=1.2)), attach_to=ego_vehicle)
        rsu_cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=20, y=10, z=15), carla.Rotation(pitch=-60)))
        actor_list.extend([ego_cam, rsu_cam])

        image_queue = queue.Queue()
        ego_cam.listen(lambda data: image_queue.put(('ego', data)))
        rsu_cam.listen(lambda data: image_queue.put(('rsu', data)))
        
        save_queue = queue.Queue()
        threading.Thread(target=save_worker, args=(save_queue,), daemon=True).start()

        spectator = world.get_spectator()
        K = utils.build_projection_matrix(IMAGE_W, IMAGE_H, FOV)

        print("\n🚀 运行中！[Q]键退出 | 天气:", args.weather, "| 时间:", args.time)
        
        for frame_idx in range(args.max_frames):
            world.tick()
            
            # 自动跟随视角
            ego_t = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(ego_t.location + carla.Location(x=-12, z=6), 
                                                 carla.Rotation(pitch=-25, yaw=ego_t.rotation.yaw)))

            data = {}
            for _ in range(2):
                name, img = image_queue.get()
                data[name] = img

            ego_w2c = utils.build_world_to_camera_matrix(ego_cam.get_transform())
            rsu_w2c = utils.build_world_to_camera_matrix(rsu_cam.get_transform())
            
            # 寻找范围内物体
            all_actors = list(world.get_actors().filter('vehicle.*')) + list(world.get_actors().filter('walker.pedestrian.*'))
            visible_targets = []
            for actor in all_actors:
                if actor.id == ego_vehicle.id: continue
                if actor.get_transform().location.distance(ego_t.location) > 80: continue
                
                # 遮挡检测逻辑
                if is_visible(world, ego_cam, actor):
                    t, bb = actor.get_transform(), actor.bounding_box
                    visible_targets.append({
                        "id": actor.id, "type": "vehicle" if actor.type_id.startswith('vehicle') else "walker",
                        "location": [t.location.x, t.location.y, t.location.z],
                        "rotation": [t.rotation.pitch, t.rotation.yaw, t.rotation.roll],
                        "extent": [bb.extent.x, bb.extent.y, bb.extent.z],
                        "center_offset": [bb.location.x, bb.location.y, bb.location.z]
                    })

            # 实时预览
            array = np.frombuffer(data['ego'].raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (IMAGE_H, IMAGE_W, 4))
            frame = array[:, :, :3].copy()
            for t in visible_targets: frame = draw_box_simple(frame, t, ego_w2c, K)
            cv2.imshow("Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # 保存
            if frame_idx % SAVE_INTERVAL == 0:
                json_data = {"frame_id": frame_idx, "camera_params": {"fov": FOV, "w": IMAGE_W, "h": IMAGE_H},
                             "matrices": {"ego_w2c": ego_w2c.tolist(), "rsu_w2c": rsu_w2c.tolist()},
                             "targets": visible_targets}
                save_queue.put({"path": save_path, "fid": frame_idx, "ego_img": data['ego'], "rsu_img": data['rsu'], "json_data": json_data})

    finally:
        global writing_thread_running
        writing_thread_running = False
        cv2.destroyAllWindows()
        settings = world.get_settings(); settings.synchronous_mode = False; world.apply_settings(settings)
        for actor in actor_list: 
            if actor: actor.destroy()
        print("\n清理完毕。")


if __name__ == '__main__':
    main()