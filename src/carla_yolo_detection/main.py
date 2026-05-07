import carla
import random
import time
import numpy as np
import cv2
import torch
import warnings

warnings.filterwarnings("ignore")

print("正在加载 YOLOv5 神经网络模型...")
# 明确告诉模型，如果有GPU就用GPU，没有就用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
print(f"模型加载完毕！当前使用的计算设备是: {device.upper()}")

# 全局变量，只保存“最新的一帧”
latest_image = None

def camera_callback(image):
    """摄像头只负责把最新画面存起来，不进行任何复杂计算"""
    global latest_image
    latest_image = image

def spawn_traffic(client, world, number_of_vehicles=30):
    bp_lib = world.get_blueprint_library()
    tm = client.get_trafficmanager(8000)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_synchronous_mode(False) 

    vehicle_bps = bp_lib.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    
    temp_actors = []
    for i in range(min(number_of_vehicles, len(spawn_points))):
        bp = random.choice(vehicle_bps)
        vehicle = world.try_spawn_actor(bp, spawn_points[i])
        if vehicle:
            vehicle.set_autopilot(True, tm.get_port())
            temp_actors.append(vehicle)
    return temp_actors

def collision_handler(event):
    print(f"\n[💥碰撞预警] 发生碰撞! 撞到了: {event.other_actor.type_id}")

def main():
    global latest_image
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()

        # 生成自车
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        vehicle.set_autopilot(True)

        # 生成背景车流
        traffic_actors = spawn_traffic(client, world, 30)
        actor_list.extend(traffic_actors)

        # 挂载摄像头 (降低分辨率，640x480对YOLOv5s刚刚好，能大幅提升速度)
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '640')
        cam_bp.set_attribute('image_size_y', '480')
        cam_bp.set_attribute('fov', '90')
        camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
        actor_list.append(camera)
        
        # 绑定轻量级回调
        camera.listen(camera_callback)

        # 挂载碰撞传感器
        col_bp = bp_lib.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        actor_list.append(collision_sensor)
        collision_sensor.listen(collision_handler)

        print("\n✅ 系统启动！按 Ctrl+C 退出...")
        
        # 主循环：专门用来做深度学习推理
        while True:
            if latest_image is not None:
                # 记录开始时间，计算FPS
                start_time = time.time()
                
                # 提取当前最新帧，并立刻清空，等待下一帧
                img_data = latest_image
                latest_image = None
                
                # 图像格式转换
                i = np.array(img_data.raw_data)
                i2 = i.reshape((img_data.height, img_data.width, 4))
                img_bgr = i2[:, :, :3]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # YOLO推理
                results = model(img_rgb)
                img_with_boxes = results.render()[0] 
                img_display = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
                
                # 计算并显示FPS
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(img_display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Optimized Detection", img_display)
                cv2.waitKey(1)
            else:
                # 如果没有新照片，稍微睡一下，防止CPU占用100%
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n正在关闭系统...")
    finally:
        for actor in actor_list:
            if actor is not None:
                actor.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()