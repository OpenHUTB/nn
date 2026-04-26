import carla
import random
import cv2
import numpy as np
import time
from ultralytics import YOLO

# 加载 YOLOv8n 预训练模型 (类别 11 为 stop sign)
model = YOLO("yolov8n.pt") 

# 声明一个全局变量，用于在主线程中显示图像
current_frame = None
latest_annotated_frame = None
frame_count = 0
INFER_EVERY_N_FRAMES = 4

def camera_callback(image):
    """
    传感器毁调：只负责处理图像和进行模型推理
    """
    global current_frame, latest_annotated_frame, frame_count
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    frame = array[:, :, :3]  

    frame_count += 1
    if frame_count % INFER_EVERY_N_FRAMES == 0:
        # YOLO 降频推理，避免每帧都推理导致主循环卡顿
        results = model(frame, classes=[11], verbose=False)
        latest_annotated_frame = results[0].plot()

    # 无论当前帧是否推理，都更新显示画面
    if latest_annotated_frame is not None:
        current_frame = latest_annotated_frame
    else:
        current_frame = frame

def main():
    # 1. 连接 Carla 模拟器
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # 2. 生成车辆 (Ego Vehicle)
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("✅ 车辆已生成！(已关闭自动驾驶)")

    # 3. 生成并安装 RGB 摄像头
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '360')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # 4. 开启摄像头监听
    camera.listen(lambda image: camera_callback(image))
    
    # 5. 初始化车辆控制对象
    control = carla.VehicleControl()
    control.throttle = 0.0
    control.steer = 0.0
    control.brake = 0.0
    last_input_time = 0.0
    key_hold_timeout = 0.18
    target_throttle = 0.0
    target_steer = 0.0
    target_reverse = False

    print("\n=========================================")
    print("🚗 准备就绪！请点击 OpenCV 窗口激活键盘焦点。")
    print("键盘操作说明 (需在 OpenCV 窗口内按键)：")
    print("  [W] : 油门前进")
    print("  [S] : 刹车 / 倒车")
    print("  [A] : 左转")
    print("  [D] : 右转")
    print("  [Q] : 退出程序")
    print("=========================================\n")

    try:
        while True:
            # 等待世界更新
            world.wait_for_tick()

            # 如果有画面，则在主线程中显示
            if current_frame is not None:
                cv2.imshow("Carla Traffic Sign Recognition", current_frame)

            key = cv2.waitKey(1) & 0xFF
            now = time.monotonic()

            if key == ord('q'):
                break
            elif key == ord('w'):
                target_throttle = 0.55
                target_steer = 0.0
                target_reverse = False
                last_input_time = now
            elif key == ord('s'):
                target_throttle = 0.35
                target_steer = 0.0
                target_reverse = True
                last_input_time = now
            elif key == ord('a'):
                target_steer = -0.45
                last_input_time = now
            elif key == ord('d'):
                target_steer = 0.45
                last_input_time = now
            elif now - last_input_time > key_hold_timeout:
                target_throttle = 0.0
                target_steer = 0.0

            # 在接近静止时再切换前进/倒车，避免频繁换挡导致顿挫
            current_speed = vehicle.get_velocity().length()
            if current_speed < 0.2:
                control.reverse = target_reverse

            # 平滑逼近目标控制，减少“一卡一卡”的体感
            throttle_step = 0.06
            steer_step = 0.10
            control.throttle += np.clip(target_throttle - control.throttle, -throttle_step, throttle_step)
            control.steer += np.clip(target_steer - control.steer, -steer_step, steer_step)
            control.brake = 0.0

            # 7. 将控制指令发送给车辆
            vehicle.apply_control(control)

    except KeyboardInterrupt:
        print("\n手动中断运行...")
    finally:
        # 清理战场，防止 Carla 里塞满垃圾车辆
        print("正在清理车辆和传感器...")
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("清理完成，程序退出。")

if __name__ == '__main__':
    main()