import carla
import random
import queue
import os
import argparse
import time

# ================= 配置区域 =================
OUTPUT_FOLDER = "_out_cvips_data"

# 【重要】改成 True 开启保存。建议先跑一次 False 确认画面正常
ENABLE_SAVING = True 

# 每隔几帧保存一次？(建议 10-20，太小会卡，太大错过细节)
SAVE_INTERVAL = 10 
# ===========================================

def main():
    argparser = argparse.ArgumentParser(description="CVIPS 数据采集脚本 V4")
    argparser.add_argument('--town', default='Town01', help='地图名称')
    argparser.add_argument('--num_vehicles', default=20, type=int, help='背景车辆数')
    argparser.add_argument('--num_walkers', default=30, type=int, help='行人数')
    args = argparser.parse_args()

    # 创建目录
    if ENABLE_SAVING:
        os.makedirs(f"{OUTPUT_FOLDER}/ego_rgb", exist_ok=True)
        os.makedirs(f"{OUTPUT_FOLDER}/rsu_rgb", exist_ok=True)
        print(f"📁 图片将保存在: {os.path.abspath(OUTPUT_FOLDER)}")

    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)
    
    print(f"正在加载地图 {args.town} (可能需要几秒)...")
    world = client.load_world(args.town)
    
    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.05 
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    # 这里的队列用于接收传感器数据
    sensor_queue = queue.Queue()
    actor_list = [] 

    try:
        # --- 1. 生成环境 ---
        print("正在构建场景...")
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        ego_spawn_point = spawn_points[0]
        npc_spawn_points = spawn_points[1:]

        # 生成背景车辆
        bg_vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        for _ in range(args.num_vehicles):
            t = random.choice(npc_spawn_points)
            actor = world.try_spawn_actor(bg_vehicle_bp, t)
            if actor:
                actor.set_autopilot(True)
                actor_list.append(actor)
        
        # 生成行人
        walker_bp = blueprint_library.filter('walker.pedestrian.*')[0]
        for _ in range(args.num_walkers):
            loc = world.get_random_location_from_navigation()
            if loc:
                w = world.try_spawn_actor(walker_bp, carla.Transform(loc))
                if w: actor_list.append(w)

        # --- 2. 主车 (Ego) ---
        print("生成主车...")
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'hero')
        ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
        ego_vehicle.set_autopilot(True)
        actor_list.append(ego_vehicle)

        # --- 3. RSU (路侧单元) ---
        rsu_loc = ego_spawn_point.location
        rsu_loc.z += 8.0 
        rsu_loc.x += 5.0
        # 俯视 45 度
        rsu_transform = carla.Transform(rsu_loc, carla.Rotation(pitch=-45, yaw=ego_spawn_point.rotation.yaw))

        # --- 4. 传感器设置 (关键修改) ---
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90') # 视野广一点

        # 【关键修改：微调主车相机位置】
        # 之前的 x=1.5, z=2.4 可能在某些车型的车顶里。
        # 改为 x=1.0 (靠后一点), z=2.0 (低一点)，通常在挡风玻璃内侧。
        cam_transform = carla.Transform(carla.Location(x=1.0, z=2.0))
        ego_cam = world.spawn_actor(camera_bp, cam_transform, attach_to=ego_vehicle)
        actor_list.append(ego_cam)
        
        rsu_cam = world.spawn_actor(camera_bp, rsu_transform)
        actor_list.append(rsu_cam)

        # 监听数据
        ego_cam.listen(lambda image: sensor_queue.put((image.frame, 'ego_rgb', image)))
        rsu_cam.listen(lambda image: sensor_queue.put((image.frame, 'rsu_rgb', image)))

        print("\n🔥 正在预热仿真 (Warm Up) ... 等待 50 帧让画面稳定")
        # --- 5. 热身阶段 (不保存数据) ---
        for _ in range(50):
            world.tick()
            # 把产生的垃圾数据从队列里清空
            try:
                for _ in range(2): sensor_queue.get(timeout=1.0)
            except: pass

        print("🚀 仿真正式开始！正在采集数据...")
        
        # --- 6. 正式循环 ---
        frame_number = 0
        spectator = world.get_spectator() 

        while True:
            # 1. 推动世界一帧
            world.tick()
            w_frame = world.get_snapshot().frame
            
            # 2. 视角跟随 (保持 V3 的跟随逻辑)
            ego_tf = ego_vehicle.get_transform()
            ego_fv = ego_tf.get_forward_vector()
            spectator_loc = ego_tf.location - (ego_fv * 6.0) + carla.Location(z=3.0)
            spectator_rot = ego_tf.rotation
            spectator_rot.pitch = -15.0
            spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))

            # 3. 严格的数据获取逻辑
            # 我们需要确保取出的 2 张图，确实属于当前的这一帧 w_frame
            try:
                current_frame_data = {} # 用字典存：{'ego_rgb': img, 'rsu_rgb': img}
                
                # 尝试从队列取数据，直到把这一帧的两个相机都取到
                # 设置超时防止死循环
                timeout_counter = 0
                while len(current_frame_data) < 2 and timeout_counter < 10:
                    data = sensor_queue.get(timeout=1.0)
                    frame_id, s_type, img_obj = data
                    
                    # 只有当数据帧号 == 世界帧号，才算有效数据
                    # (允许有 1 帧的误差，因为CARLA有时候会差1帧)
                    if abs(frame_id - w_frame) <= 1:
                        current_frame_data[s_type] = img_obj
                    else:
                        # 丢弃旧数据
                        pass
                    timeout_counter += 1

                # 4. 保存数据
                if ENABLE_SAVING and (frame_number % SAVE_INTERVAL == 0):
                    # 确保两个相机的数据都齐了才保存
                    if len(current_frame_data) == 2:
                        print(f"💾 保存帧 [{w_frame}] | Ego & RSU OK", end='\r')
                        
                        # 保存 Ego
                        fname_ego = f"{OUTPUT_FOLDER}/ego_rgb/{w_frame:06d}.png"
                        current_frame_data['ego_rgb'].save_to_disk(fname_ego)
                        
                        # 保存 RSU
                        fname_rsu = f"{OUTPUT_FOLDER}/rsu_rgb/{w_frame:06d}.png"
                        current_frame_data['rsu_rgb'].save_to_disk(fname_rsu)
                    else:
                        print(f"⚠️ 丢帧: 数据不完整", end='\r')

            except queue.Empty:
                print("⚠️ 传感器数据超时")
                continue
            
            frame_number += 1

    except KeyboardInterrupt:
        print("\n🛑 用户停止")
    finally:
        print("\n🧹 清理现场...")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for actor in actor_list:
            if actor.is_alive: actor.destroy()
        print("✅ 完成。")

if __name__ == '__main__':
    main()