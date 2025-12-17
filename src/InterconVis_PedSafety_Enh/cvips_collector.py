# --- START OF FILE cvips_collector_final.py ---

import carla
import random
import queue
import os
import argparse
import time
import threading
import json
import numpy as np
import sys
import cvips_utils as utils  # 必须确保同目录下有 cvips_utils.py

# ================= 配置区域 =================
OUTPUT_FOLDER = "_out_dataset_final"  # 修改输出目录名以示区别
ENABLE_SAVING = True
SAVE_INTERVAL = 10  # 每 10 帧存一次
TARGET_FPS = 30
FIXED_DELTA_SECONDS = 1.0 / TARGET_FPS

# 相机参数 (必须与 spawn_actor 时设置的一致)
IMAGE_W = 1920
IMAGE_H = 1080
FOV = 90.0 
# ===========================================

# 全局控制标志
writing_thread_running = True

def configure_weather(world, weather_type, time_of_day):
    """设置天气和光照"""
    weather_presets = {
        'clear': carla.WeatherParameters.ClearNoon,
        'cloudy': carla.WeatherParameters.CloudyNoon,
        'rainy': carla.WeatherParameters.HardRainNoon,
        'wet': carla.WeatherParameters.WetNoon,
    }
    weather = weather_presets.get(weather_type, carla.WeatherParameters.ClearNoon)

    if time_of_day == 'day':
        weather.sun_altitude_angle = 75.0
    elif time_of_day == 'sunset':
        weather.sun_altitude_angle = 10.0
    elif time_of_day == 'night':
        weather.sun_altitude_angle = -90.0

    world.set_weather(weather)


def cleanup_previous_hero(world):
    """启动前清理可能残留的主车"""
    actors = world.get_actors()
    potential_heroes = [x for x in actors if
                        x.type_id.startswith('vehicle') and x.attributes.get('role_name') == 'hero']
    if potential_heroes:
        print(f"🧹 发现 {len(potential_heroes)} 辆残留的主车，正在清理...")
        for h in potential_heroes:
            h.destroy()


class pygame_clock:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0

    def tick(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.start_time = time.time()
            self.frame_count = 0
        return self.fps


def save_worker(q):
    """
    后台保存线程：同时保存图片和 JSON 标签
    """
    while writing_thread_running or not q.empty():
        try:
            # 获取任务包
            task = q.get(timeout=0.1)
            path = task['scene_path']
            
            # 1. 保存图片
            images = task['image_data']
            ego_img = images['ego_rgb']
            rsu_img = images['rsu_rgb']
            
            # 这里的 frame 编号统一使用 task 里传过来的，保证对齐
            fid = task['frame_id']
            
            ego_img.save_to_disk(f"{path}/ego_rgb/{fid:08d}.jpg")
            rsu_img.save_to_disk(f"{path}/rsu_rgb/{fid:08d}.jpg")
            
            # 2. 保存 JSON 标签
            label_data = task['label_data']
            with open(f"{path}/label/{fid:08d}.json", 'w') as f:
                json.dump(label_data, f, indent=2)
                
            q.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            print(f"写入错误: {e}")


def get_environment_objects(world, ego_id):
    """
    获取环境中的车辆和行人信息
    """
    objects = []
    ego_actor = world.get_actor(ego_id)
    ego_loc = ego_actor.get_transform().location

    for actor in world.get_actors():
        # 跳过主车自己（如果你想把主车也算作 RSU 的检测目标，可以注释掉下面两行）
        if actor.id == ego_id: 
            continue 
        
        if actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker'):
            # 距离过滤：只记录主车周围 100 米内的物体
            dist = actor.get_transform().location.distance(ego_loc)
            if dist < 100:
                bb = actor.bounding_box
                t_loc = actor.get_transform().location
                t_rot = actor.get_transform().rotation
                
                # 构建单个目标的数据字典
                obj_data = {
                    "id": actor.id,
                    "type": "vehicle" if actor.type_id.startswith('vehicle') else "walker",
                    "dist_to_ego": dist,
                    # 保存位置 (x, y, z)
                    "location": [t_loc.x, t_loc.y, t_loc.z],
                    # 保存旋转 (pitch, yaw, roll)
                    "rotation": [t_rot.pitch, t_rot.yaw, t_rot.roll],
                    # 保存包围盒半长宽高 (extent)
                    "extent": [bb.extent.x, bb.extent.y, bb.extent.z],
                    # 保存包围盒中心相对于物体原点的偏移
                    "center_offset": [bb.location.x, bb.location.y, bb.location.z]
                }
                objects.append(obj_data)
    return objects


def main():
    argparser = argparse.ArgumentParser(description="CVIPS Final - 带标签采集版")
    argparser.add_argument('--town', default='Town01', help='地图名称')
    argparser.add_argument('--num_vehicles', default=40, type=int, help='车辆数')
    argparser.add_argument('--num_walkers', default=40, type=int, help='行人数')
    argparser.add_argument('--weather', default='clear', choices=['clear', 'rainy', 'cloudy', 'wet'], help='天气')
    argparser.add_argument('--time_of_day', default='day', choices=['day', 'sunset', 'night'], help='时间')
    argparser.add_argument('--max_frames', default=1000, type=int, help='采集多少帧后自动停止')

    args = argparser.parse_args()

    # 路径准备
    scene_name = f"{args.town}_{args.weather}_{args.time_of_day}"
    scene_output_path = os.path.join(OUTPUT_FOLDER, scene_name)

    if ENABLE_SAVING:
        os.makedirs(f"{scene_output_path}/ego_rgb", exist_ok=True)
        os.makedirs(f"{scene_output_path}/rsu_rgb", exist_ok=True)
        os.makedirs(f"{scene_output_path}/label", exist_ok=True)
        print(f"📂 数据将保存在: {scene_output_path}")

    client = None
    world = None
    actor_list = []

    # 传感器数据队列 (只用于暂时接收 Carla 回调数据)
    sensor_queue = queue.Queue()
    # 存盘队列 (用于发送给 save_worker)
    save_queue = queue.Queue()
    
    global writing_thread_running

    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)

        world = client.get_world()
        if world.get_map().name.split('/')[-1] != args.town:
            print(f"🗺️  正在切换地图至 {args.town} ...")
            world = client.load_world(args.town)
        else:
            print(f"🗺️  当前已是 {args.town}，准备就绪。")

        cleanup_previous_hero(world)
        configure_weather(world, args.weather, args.time_of_day)

        # 设置同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        # 启动保存线程
        writing_thread_running = True
        save_thread = threading.Thread(target=save_worker, args=(save_queue,))
        save_thread.daemon = True
        save_thread.start()

        # --- 1. 生成交通流 (NPC) ---
        print("🚗 生成交通流...")
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        ego_spawn_point = spawn_points[0]
        npc_spawn_points = spawn_points[1:]

        # 生成车辆
        bg_vehicle_bp = blueprint_library.filter('vehicle.*')
        bg_vehicle_bp = [x for x in bg_vehicle_bp if int(x.get_attribute('number_of_wheels')) == 4]

        batch = []
        for n, transform in enumerate(npc_spawn_points):
            if n >= args.num_vehicles: break
            bp = random.choice(bg_vehicle_bp)
            if bp.has_attribute('color'):
                bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
            batch.append(carla.command.SpawnActor(bp, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, True):
            if not response.error: actor_list.append(response.actor_id)

        # 生成行人
        walker_bp = blueprint_library.filter('walker.pedestrian.*')[0]
        for _ in range(args.num_walkers):
            loc = world.get_random_location_from_navigation()
            if loc:
                w = world.try_spawn_actor(walker_bp, carla.Transform(loc))
                if w: actor_list.append(w.id)
                
                # (可选) 给行人加上控制器，让他们动起来，这里简化处理，如有需要需添加 WalkerController

        # --- 2. 生成主车 (Ego) ---
        print("🚘 生成主车...")
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'hero')
        ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
        ego_vehicle.set_autopilot(True)
        actor_list.append(ego_vehicle.id)

        # --- 3. 生成传感器 (Ego & RSU) ---
        # 3.1 确定位置
        # RSU: 主车生成位置上方 10米，稍微偏一点
        rsu_loc = ego_spawn_point.location
        rsu_loc.z += 10.0
        rsu_loc.x += 8.0
        rsu_transform = carla.Transform(rsu_loc, carla.Rotation(pitch=-60, yaw=ego_spawn_point.rotation.yaw))
        
        # Ego Cam: 挂载在主车上
        cam_transform = carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(pitch=-15))

        # 3.2 配置蓝图
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IMAGE_W))
        camera_bp.set_attribute('image_size_y', str(IMAGE_H))
        camera_bp.set_attribute('fov', str(FOV))
        camera_bp.set_attribute('sensor_tick', str(FIXED_DELTA_SECONDS))

        # 3.3 Spawn
        ego_cam = world.spawn_actor(camera_bp, cam_transform, attach_to=ego_vehicle)
        rsu_cam = world.spawn_actor(camera_bp, rsu_transform) # RSU 是固定的(static)，不 attach
        actor_list.append(ego_cam.id)
        actor_list.append(rsu_cam.id)

        # 3.4 监听数据
        ego_cam.listen(lambda image: sensor_queue.put((image.frame, 'ego_rgb', image)))
        rsu_cam.listen(lambda image: sensor_queue.put((image.frame, 'rsu_rgb', image)))

        print(f"\n🚀 采集开始! 目标: {args.max_frames} 帧. 按 Ctrl+C 停止")

        frame_number = 0
        spectator = world.get_spectator()
        clock = pygame_clock()

        # --- 4. 主循环 ---
        while True:
            # 仿真步进
            world.tick()
            w_frame = world.get_snapshot().frame
            fps = clock.tick()
            
            # 更新观察者视角跟随主车
            spectator.set_transform(ego_cam.get_transform())

            if args.max_frames > 0 and frame_number >= args.max_frames:
                print("\n✅ 已达到目标帧数，自动停止。")
                break

            try:
                # 获取 RGB 图片数据
                current_frame_images = {}
                timeout = 0
                # 等待两个相机的图片都到齐
                while len(current_frame_images) < 2 and timeout < 10:
                    data = sensor_queue.get(timeout=1.0)
                    fid, stype, img = data
                    # 只收当前帧的数据，防止错位
                    if abs(fid - w_frame) <= 2:
                        current_frame_images[stype] = img
                    timeout += 1

                # 只有当两个相机数据都齐了，并且满足保存间隔时，才进行标签计算和保存
                if len(current_frame_images) == 2 and ENABLE_SAVING and (frame_number % SAVE_INTERVAL == 0):
                    
                    # --- 核心新增逻辑: 计算矩阵和标签 ---
                    
                    # 1. 获取相机位姿
                    ego_trans = ego_cam.get_transform()
                    rsu_trans = rsu_cam.get_transform()
                    
                    # 2. 计算 外参矩阵 (World -> Camera)
                    # 这一步需要 cvips_utils.py 的支持
                    ego_w2c = utils.build_world_to_camera_matrix(ego_trans)
                    rsu_w2c = utils.build_world_to_camera_matrix(rsu_trans)
                    
                    # 3. 获取所有目标物体 (3D框)
                    targets = get_environment_objects(world, ego_vehicle.id)
                    
                    # 4. 组装 JSON 标签数据
                    frame_label_data = {
                        "frame_id": frame_number,
                        "timestamp": world.get_snapshot().timestamp.elapsed_seconds,
                        "camera_params": {
                            "fov": FOV,
                            "width": IMAGE_W,
                            "height": IMAGE_H
                        },
                        "matrices": {
                            "ego_w2c": ego_w2c.tolist(),
                            "rsu_w2c": rsu_w2c.tolist()
                        },
                        "targets": targets
                    }
                    
                    # 5. 打包发送给后台线程
                    save_task = {
                        "scene_path": scene_output_path,
                        "frame_id": frame_number,
                        "image_data": current_frame_images,
                        "label_data": frame_label_data
                    }
                    save_queue.put(save_task)
                    
                    print(f"FPS: {fps:.1f} | Frame: {frame_number} | Saved: ✅ | Queue: {save_queue.qsize()} ", end='\r')
                else:
                    # 不保存的帧，仅打印进度
                    print(f"FPS: {fps:.1f} | Frame: {frame_number} | Saved: ⏭️ | Queue: {save_queue.qsize()} ", end='\r')

            except queue.Empty:
                pass
            
            frame_number += 1

    except KeyboardInterrupt:
        print("\n\n🛑 用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n♻️  正在退出...")

        # 停止写入
        writing_thread_running = False
        if 'save_thread' in locals() and save_thread.is_alive():
            print("⏳ 等待后台写入完成...", end='')
            save_thread.join(timeout=5)
            print("Done")

        # 销毁对象
        if client and actor_list:
            print("💥 销毁 Actor...")
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

        # 恢复异步
        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print("✅ 资源已释放，强制返回终端。")
        os._exit(0)

if __name__ == '__main__':
    main()