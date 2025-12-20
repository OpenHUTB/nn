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
    """
    【完整保留】最初的天气和时间逻辑
    """
    presets = {
        'Clear': carla.WeatherParameters.ClearNoon,
        'Cloudy': carla.WeatherParameters.CloudyNoon,
        'Wet': carla.WeatherParameters.WetNoon,
        'Rain': carla.WeatherParameters.HardRainNoon,
        'Storm': carla.WeatherParameters.HardRainSunset
    }
    
    weather = presets.get(weather_arg, carla.WeatherParameters.ClearNoon)
    
    # 修改太阳高度角模拟时间
    if time_arg == 'Day':
        weather.sun_altitude_angle = 60.0
    elif time_arg == 'Sunset':
        weather.sun_altitude_angle = 10.0
    elif time_arg == 'Night':
        weather.sun_altitude_angle = -90.0
        
    world.set_weather(weather)
    print(f"🌦️ 环境设置成功: {weather_arg} | {time_arg}")

def is_visible_refined(world, camera_actor, target_actor):
    """
    【修复版】可见性检测：针对两轮车做了高度优化
    """
    cam_trans = camera_actor.get_transform()
    # 起点：相机位置 + 向上提一点，彻底避开主车车身
    ray_start = cam_trans.location + carla.Location(z=0.5) + cam_trans.get_forward_vector() * 2.0
    
    # 终点：两轮车较低，瞄准 z=0.6，轿车/行人瞄准 z=1.0
    target_z = 0.6 if 'bike' in target_actor.type_id or 'bicycle' in target_actor.type_id or 'motorcycle' in target_actor.type_id else 1.0
    target_loc = target_actor.get_transform().location + carla.Location(z=target_z)

    hit_points = world.cast_ray(ray_start, target_loc)
    if not hit_points: return True # 路径没撞击到任何东西，肯定可见
    
    # 只要第一个碰撞点离目标中心较近，就判定为可见（防止因车体大而误判）
    return hit_points[0].location.distance(target_loc) < 3.0

def draw_box(img, target, w2c, K):
    """
    【稳定版】绘制3D框，行人红色，车绿色，两轮车紫色
    """
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
    for pt in corners_local:
        w_pos = np.dot(obj_to_world, pt)
        p = utils.get_image_point(carla.Location(x=w_pos[0], y=w_pos[1], z=w_pos[2]), K, w2c)
        img_pts.append(p)

    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    if target['type'] == 'walker': color = (0, 0, 255)
    elif target['type'] == 'bike': color = (255, 0, 255)
    else: color = (0, 255, 0)
    
    for s, e in edges:
        if img_pts[s] and img_pts[e]:
            cv2.line(img, tuple(img_pts[s]), tuple(img_pts[e]), color, 2)
    return img

def main():
    # 【恢复】所有原始参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--town', default='Town01')
    parser.add_argument('--weather', default='Clear', choices=['Clear', 'Cloudy', 'Wet', 'Rain', 'Storm'])
    parser.add_argument('--time', default='Day', choices=['Day', 'Sunset', 'Night'])
    parser.add_argument('--num_vehicles', default=50, type=int)
    parser.add_argument('--num_walkers', default=30, type=int)
    parser.add_argument('--max_frames', default=500, type=int)
    args = parser.parse_args()

    scene_name = f"{args.town}_{args.weather}_{args.time}"
    save_path = os.path.join(OUTPUT_FOLDER, scene_name)
    for sub in ['ego_rgb', 'rsu_rgb', 'label']:
        os.makedirs(os.path.join(save_path, sub), exist_ok=True)

    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    if args.town not in world.get_map().name:
        world = client.load_world(args.town)
    
    # 【恢复】环境配置
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
        random.shuffle(spawn_points)

        # 1. 主车 (Ego)
        ego_bp = bp_lib.find('vehicle.tesla.model3')
        ego_vehicle = world.spawn_actor(ego_bp, spawn_points[0])
        ego_vehicle.set_autopilot(True, tm.get_port())
        actor_list.append(ego_vehicle)

        # 2. 生成背景车 (包含自行车/摩托车)
        print(f"🚗 正在生成 {args.num_vehicles} 辆背景车...")
        batch = []
        for i, transform in enumerate(spawn_points[1:]):
            if i >= args.num_vehicles: break
            bp = random.choice(bp_lib.filter('vehicle.*'))
            batch.append(carla.command.SpawnActor(bp, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))
        
        for r in client.apply_batch_sync(batch):
            if not r.error: actor_list.append(world.get_actor(r.actor_id))

        # 3. 【恢复】生成行人
        print(f"🚶 正在生成 {args.num_walkers} 个行人...")
        for _ in range(args.num_walkers):
            loc = world.get_random_location_from_navigation()
            if loc:
                w = world.try_spawn_actor(random.choice(bp_lib.filter('walker.pedestrian.*')), carla.Transform(loc))
                if w: actor_list.append(w)

        # 4. 相机设置
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(IMAGE_W)); cam_bp.set_attribute('image_size_y', str(IMAGE_H))
        cam_bp.set_attribute('fov', str(FOV))
        ego_cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=2.0, x=1.0)), attach_to=ego_vehicle)
        rsu_cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=20, y=10, z=15), carla.Rotation(pitch=-60)))
        actor_list.extend([ego_cam, rsu_cam])

        image_queue = queue.Queue()
        ego_cam.listen(lambda data: image_queue.put(('ego', data)))
        rsu_cam.listen(lambda data: image_queue.put(('rsu', data)))
        
        save_queue = queue.Queue()
        threading.Thread(target=save_worker, args=(save_queue,), daemon=True).start()

        K = utils.build_projection_matrix(IMAGE_W, IMAGE_H, FOV)
        spectator = world.get_spectator()

        print("\n🚀 运行中... 按 Q 退出预览")
        
        for frame_idx in range(args.max_frames):
            world.tick()
            
            # 【保留】自动视角跟随
            ego_t = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(ego_t.location + carla.Location(x=-12, z=6), 
                                                 carla.Rotation(pitch=-25, yaw=ego_t.rotation.yaw)))

            data = {}
            for _ in range(2):
                name, img = image_queue.get()
                data[name] = img

            ego_w2c = utils.build_world_to_camera_matrix(ego_cam.get_transform())
            rsu_w2c = utils.build_world_to_camera_matrix(rsu_cam.get_transform())
            
            # 标注可见目标
            all_actors = world.get_actors().filter('vehicle.*')
            all_walkers = world.get_actors().filter('walker.pedestrian.*')
            
            visible_targets = []
            for actor in list(all_actors) + list(all_walkers):
                if actor.id == ego_vehicle.id: continue
                if actor.get_transform().location.distance(ego_t.location) > 80: continue
                
                # 遮挡检测判定
                if is_visible_refined(world, ego_cam, actor):
                    t, bb = actor.get_transform(), actor.bounding_box
                    
                    # 识别类型
                    atype = 'vehicle'
                    if 'walker' in actor.type_id: atype = 'walker'
                    elif any(x in actor.type_id for x in ['bike', 'bicycle', 'motorcycle']): atype = 'bike'
                    
                    visible_targets.append({
                        "id": actor.id, "type": atype,
                        "location": [t.location.x, t.location.y, t.location.z],
                        "rotation": [t.rotation.pitch, t.rotation.yaw, t.rotation.roll],
                        "extent": [bb.extent.x, bb.extent.y, bb.extent.z],
                        "center_offset": [bb.location.x, bb.location.y, bb.location.z]
                    })

            # 预览
            array = np.frombuffer(data['ego'].raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (IMAGE_H, IMAGE_W, 4))
            frame = array[:, :, :3].copy()
            for t in visible_targets: frame = draw_box(frame, t, ego_w2c, K)
            cv2.imshow("Final Stable Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # 保存
            if frame_idx % SAVE_INTERVAL == 0:
                json_data = {
                    "frame_id": frame_idx, 
                    "camera_params": {"fov": FOV, "w": IMAGE_W, "h": IMAGE_H},
                    "matrices": {"ego_w2c": ego_w2c.tolist(), "rsu_w2c": rsu_w2c.tolist()}, 
                    "targets": visible_targets
                }
                save_queue.put({"path": save_path, "fid": frame_idx, "ego_img": data['ego'], "rsu_img": data['rsu'], "json_data": json_data})

    finally:
        global writing_thread_running
        writing_thread_running = False
        cv2.destroyAllWindows()
        settings = world.get_settings(); settings.synchronous_mode = False; world.apply_settings(settings)
        for actor in actor_list: 
            if actor: actor.destroy()

def save_worker(q):
    while writing_thread_running or not q.empty():
        try:
            task = q.get(timeout=0.1)
            task['ego_img'].save_to_disk(f"{task['path']}/ego_rgb/{task['fid']:08d}.jpg")
            task['rsu_img'].save_to_disk(f"{task['path']}/rsu_rgb/{task['fid']:08d}.jpg")
            with open(f"{task['path']}/label/{task['fid']:08d}.json", 'w') as f: json.dump(task['json_data'], f, indent=2)
        except: continue

if __name__ == '__main__':
    main()