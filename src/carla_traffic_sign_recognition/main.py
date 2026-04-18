import carla
import random
import cv2
import numpy as np
from ultralytics import YOLO

# 加载 YOLOv8n 预训练模型 (类别 11 为 stop sign)
model = YOLO("yolov8n.pt") 

# 声明一个全局变量，用于在主线程中显示图像
current_frame = None

def camera_callback(image):
    """
    传感器毁调：只负责处理图像和进行模型推理
    """
    global current_frame
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    frame = array[:, :, :3]  
    
    # YOLO 推理
    results = model(frame, classes=[11], verbose=False)
    # 画上识别框并存入全局变量
    current_frame = results[0].plot()

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
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # 4. 开启摄像头监听
    camera.listen(lambda image: camera_callback(image))
    
    # 5. 初始化车辆控制对象
    control = carla.VehicleControl()

    print("\n=========================================")
    print("🚗 准备就绪！请务必点击弹出的 OpenCV 窗口激活它！")
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
            
            # 6. 核心修改：通过 OpenCV 捕获键盘按键并转换为车辆控制指令
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'): # 按 Q 退出主循环
                break
            elif key == ord('w'):
                control.throttle = 0.5   # 踩一半油门
                control.steer = 0.0      # 方向盘回正
                control.brake = 0.0
            elif key == ord('s'):
                control.throttle = 0.0
                control.steer = 0.0
                control.brake = 0.8      # 踩刹车
            elif key == ord('a'):
                control.throttle = 0.3   # 转弯时给点油门
                control.steer = -0.5     # 方向盘左打一半
                control.brake = 0.0
            elif key == ord('d'):
                control.throttle = 0.3
                control.steer = 0.5      # 方向盘右打一半
                control.brake = 0.0
            else:
                # 没有任何按键时，松开油门和刹车，方向盘回正
                control.throttle = 0.0
                control.brake = 0.0
                control.steer = 0.0

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