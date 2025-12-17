import carla
import random
import queue
import os
import argparse
import time
import threading
import logging
import sys

# ================= 配置区域 =================
OUTPUT_FOLDER = "_out_dataset_v10"
ENABLE_SAVING = True
SAVE_INTERVAL = 10  # 每 10 帧存一次
TARGET_FPS = 30
FIXED_DELTA_SECONDS = 1.0 / TARGET_FPS
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


def main():
    argparser = argparse.ArgumentParser(description="CVIPS Pro - 强制退出版")
    argparser.add_argument('--town', default='Town01', help='地图名称')
    argparser.add_argument('--num_vehicles', default=40, type=int, help='车辆数')
    argparser.add_argument('--num_walkers', default=40, type=int, help='行人数')
    argparser.add_argument('--weather', default='clear', choices=['clear', 'rainy', 'cloudy', 'wet'], help='天气')
    argparser.add_argument('--time_of_day', default='day', choices=['day', 'sunset', 'night'], help='时间')
    argparser.add_argument('--max_frames', default=1000, type=int, help='采集多少帧后自动停止(0为不停止)')

    args = argparser.parse_args()

    scene_name = f"{args.town}_{args.weather}_{args.time_of_day}"
    scene_output_path = os.path.join(OUTPUT_FOLDER, scene_name)

    if ENABLE_SAVING:
        os.makedirs(f"{scene_output_path}/ego_rgb", exist_ok=True)
        os.makedirs(f"{scene_output_path}/rsu_rgb", exist_ok=True)

    client = None
    world = None
    actor_list = []

    sensor_queue = queue.Queue()
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

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        writing_thread_running = True
        save_thread = threading.Thread(target=save_worker, args=(save_queue, scene_output_path))
        # 【修改1】设置为守护线程，主程序一死它必须死
        save_thread.daemon = True
        save_thread.start()

        # --- 生成 Actor ---
        print("🚗 生成交通流...")
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        ego_spawn_point = spawn_points[0]
        npc_spawn_points = spawn_points[1:]

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

        walker_bp = blueprint_library.filter('walker.pedestrian.*')[0]
        for _ in range(args.num_walkers):
            loc = world.get_random_location_from_navigation()
            if loc:
                w = world.try_spawn_actor(walker_bp, carla.Transform(loc))
                if w: actor_list.append(w.id)

        print("🚘 生成主车...")
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'hero')
        ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
        ego_vehicle.set_autopilot(True)
        actor_list.append(ego_vehicle.id)

        rsu_loc = ego_spawn_point.location
        rsu_loc.z += 10.0
        rsu_loc.x += 8.0
        rsu_transform = carla.Transform(rsu_loc, carla.Rotation(pitch=-60, yaw=ego_spawn_point.rotation.yaw))
        cam_transform = carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(pitch=-15))

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('sensor_tick', str(FIXED_DELTA_SECONDS))

        ego_cam = world.spawn_actor(camera_bp, cam_transform, attach_to=ego_vehicle)
        rsu_cam = world.spawn_actor(camera_bp, rsu_transform)
        actor_list.append(ego_cam.id)
        actor_list.append(rsu_cam.id)

        ego_cam.listen(lambda image: sensor_queue.put((image.frame, 'ego_rgb', image)))
        rsu_cam.listen(lambda image: sensor_queue.put((image.frame, 'rsu_rgb', image)))

        print(f"\n🚀 采集开始! 按 Ctrl+C 停止")

        frame_number = 0
        spectator = world.get_spectator()
        clock = pygame_clock()

        while True:
            world.tick()
            w_frame = world.get_snapshot().frame
            fps = clock.tick()
            spectator.set_transform(ego_cam.get_transform())

            if args.max_frames > 0 and frame_number >= args.max_frames:
                print("\n✅ 已达到目标帧数，自动停止。")
                break

            try:
                current_frame_data = {}
                timeout = 0
                while len(current_frame_data) < 2 and timeout < 10:
                    data = sensor_queue.get(timeout=1.0)
                    fid, stype, img = data
                    if abs(fid - w_frame) <= 2:
                        current_frame_data[stype] = img
                    timeout += 1

                if ENABLE_SAVING and (frame_number % SAVE_INTERVAL == 0):
                    if len(current_frame_data) == 2:
                        print(f"FPS: {fps:.1f} | Frame: {frame_number} | 队列: {save_queue.qsize()} ", end='\r')
                        save_queue.put(current_frame_data)
            except queue.Empty:
                pass
            frame_number += 1

    except KeyboardInterrupt:
        print("\n\n🛑 用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
    finally:
        print("\n♻️  正在退出...")

        # 停止写入
        writing_thread_running = False
        if 'save_thread' in locals() and save_thread.is_alive():
            print("⏳ 等待后台写入完成...", end='')
            save_thread.join(timeout=5)  # 最多等5秒，不等了
            print("Done")

        # 销毁对象
        if client and actor_list:
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

        # 恢复异步
        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print("✅ 资源已释放，强制返回终端。")
        # 【修改2】核弹级退出：直接调用系统底层退出，不给 Python 挂起的机会
        os._exit(0)


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


def save_worker(q, path):
    while writing_thread_running or not q.empty():
        try:
            data = q.get(timeout=0.1)
            ego = data['ego_rgb']
            rsu = data['rsu_rgb']
            ego.save_to_disk(f"{path}/ego_rgb/{ego.frame:08d}.jpg")
            rsu.save_to_disk(f"{path}/rsu_rgb/{rsu.frame:08d}.jpg")
            q.task_done()
        except:
            pass


if __name__ == '__main__':
    main()