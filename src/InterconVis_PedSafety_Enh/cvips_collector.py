import carla
import random
import queue
import os
import argparse
import time
import threading

# ================= 配置区域 =================
# 输出文件夹名称
OUTPUT_FOLDER = "_out_cvips_final"

# 是否开启保存 (调试视角时可设为 False)
ENABLE_SAVING = True 

# 采集间隔：每 15 帧保存一次 (平衡 1080P 的存储压力)
SAVE_INTERVAL = 15  
# ===========================================

# 全局标志位，控制后台线程何时停止
writing_thread_running = True

def main():
    argparser = argparse.ArgumentParser(description="CVIPS 最终版数据采集脚本")
    argparser.add_argument('--town', default='Town01', help='地图名称')
    argparser.add_argument('--num_vehicles', default=25, type=int, help='背景车辆数')
    argparser.add_argument('--num_walkers', default=40, type=int, help='行人数')
    args = argparser.parse_args()

    # 1. 创建保存目录
    if ENABLE_SAVING:
        os.makedirs(f"{OUTPUT_FOLDER}/ego_rgb", exist_ok=True)
        os.makedirs(f"{OUTPUT_FOLDER}/rsu_rgb", exist_ok=True)
        print(f"📁 数据保存路径: {os.path.abspath(OUTPUT_FOLDER)}")

    # 2. 连接 CARLA (使用 127.0.0.1 避免 Windows 防火墙问题)
    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(60.0)
        print(f"正在加载地图 {args.town} ...")
        world = client.load_world(args.town)
    except RuntimeError as e:
        print(f"❌ 连接失败: {e}")
        print("请确保 CARLA 模拟器已启动！")
        return

    # 3. 设置高画质天气 (正午晴天)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # 4. 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.05 # 固定 20 FPS
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    # 5. 初始化队列和列表
    sensor_queue = queue.Queue() # 接收原始数据
    save_queue = queue.Queue()   # 后台保存队列
    actor_list = [] 

    # 6. 启动后台保存线程
    global writing_thread_running
    writing_thread_running = True
    save_thread = threading.Thread(target=save_worker, args=(save_queue,))
    save_thread.start()
    print("✅ 后台保存服务已启动")

    try:
        # --- 生成环境 ---
        print("正在构建交通场景...")
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        # 分离主车点和NPC点，防止碰撞
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

        # --- 生成主车 (Ego) ---
        print("生成主车...")
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'hero')
        ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
        ego_vehicle.set_autopilot(True)
        actor_list.append(ego_vehicle)

        # --- 生成 RSU (路侧单元) ---
        rsu_loc = ego_spawn_point.location
        rsu_loc.z += 12.0 # 12米高空
        rsu_loc.x += 5.0
        rsu_transform = carla.Transform(rsu_loc, carla.Rotation(pitch=-70, yaw=ego_spawn_point.rotation.yaw))

        # --- 传感器设置 (1080P 高画质) ---
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '90')
        # 优化画质属性
        camera_bp.set_attribute('exposure_mode', 'histogram') 
        camera_bp.set_attribute('motion_blur_intensity', '0.2')

        # 主车相机：第三人称 (车后6米，高3米)，防遮挡
        cam_transform = carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-20))
        
        ego_cam = world.spawn_actor(camera_bp, cam_transform, attach_to=ego_vehicle)
        actor_list.append(ego_cam)
        
        rsu_cam = world.spawn_actor(camera_bp, rsu_transform)
        actor_list.append(rsu_cam)

        # 监听数据
        ego_cam.listen(lambda image: sensor_queue.put((image.frame, 'ego_rgb', image)))
        rsu_cam.listen(lambda image: sensor_queue.put((image.frame, 'rsu_rgb', image)))

        print("\n🔥 正在预热 (Warm Up)... 请保持 CARLA 窗口在前台！")
        for _ in range(50):
            world.tick()
            try:
                # 清空预热期的垃圾数据
                for _ in range(2): sensor_queue.get(timeout=1.0)
            except: pass

        print("🚀 采集开始！按 Ctrl+C 优雅退出...")
        
        frame_number = 0
        spectator = world.get_spectator() 

        while True:
            # 1. 物理计算一帧
            world.tick()
            w_frame = world.get_snapshot().frame
            
            # 2. 移动观众视角跟随主车 (方便你观察)
            spectator.set_transform(ego_cam.get_transform())

            try:
                # 3. 获取数据
                current_frame_data = {}
                timeout_counter = 0
                # 尝试凑齐两个相机的数据
                while len(current_frame_data) < 2 and timeout_counter < 10:
                    data = sensor_queue.get(timeout=1.0)
                    frame_id, s_type, img_obj = data
                    # 允许 1 帧的误差
                    if abs(frame_id - w_frame) <= 1:
                        current_frame_data[s_type] = img_obj
                    timeout_counter += 1

                # 4. 放入后台队列
                if ENABLE_SAVING and (frame_number % SAVE_INTERVAL == 0):
                    if len(current_frame_data) == 2:
                        print(f"Frame: {w_frame} | 待存队列: {save_queue.qsize()}", end='\r')
                        save_queue.put(current_frame_data)
                        
            except queue.Empty:
                continue
            
            frame_number += 1

    except KeyboardInterrupt:
        print("\n🛑 用户请求停止")

    finally:
        print("\n🧹 正在执行清理程序...")
        
        # 1. 停止后台线程
        writing_thread_running = False
        
        # 2. 等待剩余照片保存完毕 (解决报错的关键)
        if not save_queue.empty():
            print(f"⏳ 正在保存剩余的 {save_queue.qsize()} 张照片，请不要关闭窗口...", end='', flush=True)
            save_thread.join()
            print(" 保存完毕！")
        else:
            save_thread.join()

        # 3. 恢复 CARLA 设置 (防止下次启动变卡)
        try:
            if world:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
        except:
            pass

        # 4. 安全销毁所有对象
        print("🗑️ 销毁车辆和传感器...")
        for actor in actor_list:
            try:
                if actor.is_alive:
                    actor.destroy()
            except:
                pass # 忽略销毁时的错误
                
        print("✅ 全部完成，程序安全退出。")

# --- 后台工作线程 ---
def save_worker(q):
    while writing_thread_running or not q.empty():
        try:
            data_dict = q.get(timeout=1.0) 
            ego_img = data_dict['ego_rgb']
            rsu_img = data_dict['rsu_rgb']
            
            # 保存 Ego
            path_ego = f"{OUTPUT_FOLDER}/ego_rgb/{ego_img.frame:06d}.png"
            ego_img.save_to_disk(path_ego)
            
            # 保存 RSU
            path_rsu = f"{OUTPUT_FOLDER}/rsu_rgb/{rsu_img.frame:06d}.png"
            rsu_img.save_to_disk(path_rsu)
            
            q.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"保存错误: {e}")

if __name__ == '__main__':
    main()