import glob
import os
import sys
import cv2
import carla
import math
import random
from queue import Queue
from queue import Empty
import numpy as np
import time
import logging
import argparse
from tqdm import tqdm  # 新增：导入进度条库

# from pascal_voc_writer import Writer
import utils.cva_utils as cva_utils
import utils.world_utils as world_utils
import utils.img_utils as img_utils
import utils.bbox_utils as bbox_utils
import utils.server_utils as server_utils

log = logging.getLogger(__name__)

def get_checkpoint():
    num_save = 0
    try:
        with open('checkpoint.txt', 'r') as f:
            num_save_ckpt = int(f.read().strip())
        print("Loading checkpoint!")
        return num_save_ckpt
    except:
        print("Checkpoint not found")
    return num_save

def retrieve_data(sensor_queue, frame, timeout=5):
    while True:
        try:
            data = sensor_queue.get(True, timeout)
        except Empty:
            return None
        if data.frame == frame:
            return data

def main(args):
    output_path = args.output_path
    if args.save:
        if not os.path.exists(output_path): 
            os.makedirs(output_path)
    
    # ==============================================
    # Set up CARLA world
    # ==============================================

    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0) # 给地图加载预留更多时间
    print("Loading world", args.map)
    client.load_world(args.map)
    world  = client.get_world()
    original_settings = world.get_settings()

    image_count = 0
    num_saved = get_checkpoint()
    save_every = 40
    stop_count = 0

    weather_every = 80
    weather_tick = 20

    # 初始化变量
    actor_list, walkers_list, sensor_list = [], [], []
# 新增：如果是保存模式，则初始化进度条
    pbar = None
    if args.save:
        pbar = tqdm(total=args.num_save, initial=num_saved, desc="数据采集进度", unit="img")

    try:
        bp_lib = world.get_blueprint_library()

        # Set up the simulator in synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True # Enables synchronous mode
        settings.fixed_delta_seconds = 0.03
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        # Reset simulation if car is stuck
        # Reset simulation if car is stuck
        def reset():
            # 移除 original_settings 退化机制，保持严格同步模式
            print('清理旧实体，准备重置场景...')
            # 使用外层已存在的列表进行销毁，防止内存泄漏
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            if walkers_list:
                client.apply_batch([carla.command.DestroyActor(x['id']) for x in walkers_list if 'id' in x])
            for sensor in sensor_list:
                sensor.destroy()
            
            # 新增：强制推进一帧，让服务器物理回收上述垃圾，避免 ID 冲突
            world.tick()
            spawn_success = False
            while not spawn_success:
                try:
                    actor_list_local, walkers_list_local, all_id = world_utils.spawn_actors(client, world, args.num_vehicles, args.num_walkers)
                    vehicle = actor_list_local[0]
            
                    sensor_list_local, q_list, sensor_idxs = world_utils.spawn_sensors(world, vehicle)
                    camera = sensor_list_local[0]
            
                    spawn_success = True # 全部成功才退出循环
                except Exception as e:
                    log.warning(f"[WARN] 场景初始化失败，正在重试... 错误: {e}")
                    time.sleep(1)
            return actor_list_local, walkers_list_local, all_id, vehicle, sensor_list_local, q_list, sensor_idxs, camera
        
        # 初始化所有变量
        actor_list, walkers_list, all_id, vehicle, sensor_list, q_list, sensor_idxs, camera = reset()

        traffic_signs = world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
        
        while (not args.save) or (num_saved < args.num_save):
            # Track vehicle velocity to ensure that car isn't stuck
            velocity = vehicle.get_velocity()
            velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) 
            if velocity < 0.01:
                stop_count += 1
            else:
                stop_count = 0

            # Reset simulation if car is stuck
            if stop_count == 120:
                print("Car is stuck! Resetting...")
                actor_list, walkers_list, all_id, vehicle, sensor_list, q_list, sensor_idxs, camera = reset()
                
            # Change weather
            if weather_tick == 0 and not args.constant_weather:
                print("Changing weather...")
                world.set_weather(random.choice(world_utils.weather_presets))
            weather_tick = (weather_tick + 1) % weather_every

            # Retrieve the image
            nowFrame = world.tick()

            data = [cva_utils.retrieve_data(q,nowFrame) for q in q_list]
            assert all(x.frame == nowFrame for x in data if x is not None)

            # Skip if any sensor data is not available
            if None in data:
                continue
                
            snap = data[sensor_idxs['tick']]
            image = data[sensor_idxs['rgb']]
            depth_image = data[sensor_idxs['depth']]
            lidar_image = data[sensor_idxs['lidar']]
            semantic_image = data[sensor_idxs['semantic']]

            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            clean_img = np.copy(img)
            boundingbox_path = os.path.join(output_path, "boundingbox")
            if args.save:
                if not os.path.exists(boundingbox_path): 
                    os.makedirs(boundingbox_path)

            bbox_draw = []
            bbox_save = []

            # Get all vehicle bounding boxes
            vehicles_raw = world.get_actors().filter('*vehicle*')
            vehicles = cva_utils.snap_processing(vehicles_raw, snap)
            vehicle_bbox_draw, vehicle_bbox_save = bbox_utils.actor_bbox_lidar(
                actor_list=vehicles,
                camera=camera,
                image=image,
                lidar_image=lidar_image,
                max_dist=40,
                min_detect=6,
                class_id=0
            )
            bbox_draw.extend(vehicle_bbox_draw)
            bbox_save.extend(vehicle_bbox_save)

            # Get all pedestrian bounding boxes
            pedestrians_raw = world.get_actors().filter('*pedestrian*')
            pedestrians = cva_utils.snap_processing(pedestrians_raw, snap)
            depth_meter = cva_utils.extract_depth(depth_image)

            walker_bbox_draw, walker_bbox_save = bbox_utils.actor_bbox_depth_semantic(
                actor_list=pedestrians, 
                camera=camera, 
                image=image,
                semantic_image=semantic_image,
                depth_image=depth_meter,
                max_dist=40, 
                depth_margin=8, 
                patch_ratio=0.4, 
                resize_ratio=0.5, 
                semantic_label=12,
                semantic_threshold=0.3,
                class_id=1
            )
            bbox_draw.extend(walker_bbox_draw)
            bbox_save.extend(walker_bbox_save)

            # Getting traffic light bounding boxes
            traffic_lights = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
            depth_meter = cva_utils.extract_depth(depth_image)
            tl_bbox_draw, tl_bbox_save = bbox_utils.object_bbox_depth_semantic(
                bbox_list=traffic_lights, 
                camera=camera, 
                image=image,
                semantic_image=semantic_image,
                depth_image=depth_meter, 
                vehicle=vehicle, 
                max_dist=30, 
                semantic_threshold=0.5,
                semantic_label=7,
                class_id=2
            )

            bbox_draw.extend(tl_bbox_draw)
            bbox_save.extend(tl_bbox_save)
            
            # Getting stop sign bounding boxes
            depth_meter = cva_utils.extract_depth(depth_image)
            ts_bbox_draw, ts_bbox_save = bbox_utils.object_bbox_depth_semantic(
                bbox_list=traffic_signs, 
                camera=camera, 
                image=image,
                semantic_image=semantic_image,
                depth_image=depth_meter, 
                vehicle=vehicle, 
                max_dist=25,
                semantic_threshold=0.5,
                semantic_label=8,
                class_id=3
            )
            bbox_draw.extend(ts_bbox_draw)
            bbox_save.extend(ts_bbox_save)

            # Now, draw bboxes and show images
            annotation_str = ""
            for bbox_d, bbox_s in zip(bbox_draw, bbox_save):
                u1, v1, u2, v2 = bbox_d
                class_id, x_center, y_center, width, height = bbox_s

                img_utils.draw_boundingbox(img, u1, v1, u2, v2)
                annotation_str += f"{class_id} {x_center} {y_center} {width} {height}\n"
            
            # Save image and labels if settings allow it
            if args.save and image_count % save_every == 0:
                if len(bbox_save) < args.num_detections_save:
                    print("Minimum detections not reached, skipping...")
                else:
                    num_saved += 1
                    # 新增：更新进度条，移除原本刷屏的 print
                    if pbar:
                        pbar.update(1)
                    cv2.imwrite(os.path.join(output_path, args.map + '_' + '%06d.png' % image.frame), clean_img)
                    with open(os.path.join(output_path, args.map + '_' + '%06d.txt' % image.frame), "a") as f:
                        f.write(annotation_str)
                
                    # Save image with bounding box
                    output_file_path = os.path.join(boundingbox_path, args.map + '_' + '%06d_b.png' % image.frame)
                    cv2.imwrite(output_file_path, img)
                    
            # Show image with bounding box
            if args.show:
                cv2.imshow('ImageWindowName',img)
                if cv2.waitKey(1) == ord('q'):
                    break

            image_count += 1

        print('Data collection finished!')
        if os.path.exists('checkpoint.txt'):
            os.remove('checkpoint.txt')
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Simulation crashed: {e}")
        import traceback
        traceback.print_exc()
        with open('checkpoint.txt', 'w') as f:
            f.write(str(num_saved))

    finally:
        # 新增：安全关闭进度条
        if pbar:
            pbar.close()
        world.apply_settings(original_settings)
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        if 'walkers_list' in locals() and len(walkers_list) > 0:
            client.apply_batch([carla.command.DestroyActor(x['id']) for x in walkers_list if 'id' in x])
        for sensor in sensor_list:
            sensor.destroy()
        print('done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--map', 
        type=str, 
        default='Town01',
        help="CARLA map that data will be collected on"
    )

    parser.add_argument(
        '--num_vehicles',
        type=int,
        default=70,
        help="Number of vehicles spawned in simulation"
    )

    parser.add_argument(
        '--num_walkers',
        type=int,
        default=150,
        help="Number of pedestrians spawned in simulation"
    )

    parser.add_argument(
        '--constant_weather',
        action='store_true',
        help="Turns mid-simulation weather switching off during data collection"
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Shows annotations on screen during data collection'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='Saves image and label data to disk'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default='./carla_data',
        help='Relative path where raw data is saved'
    )

    parser.add_argument(
        '--num_save',
        type=int,
        default=50,
        help='Number of images to save for this run.'
    )

    parser.add_argument(
        '--num_detections_save',
        type=int,
        default=1,
        help='Minimum number of detections per collected datapoint'
    )

    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')