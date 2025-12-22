# git commit -m "V2X路侧感知系统（动态画面版+参数修复）"
# 修复：draw_boxes参数缺失 + 动态画面静止问题
# ==============================================
import sys
import os
import time
import cv2
import numpy as np


# 1. 强制加载Carla

CARLA_EGG_PATH = r"D:\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg"
if not os.path.exists(CARLA_EGG_PATH):
    print(f"❌ Carla egg文件不存在！路径：{CARLA_EGG_PATH}")
    sys.exit(1)
sys.path.insert(0, CARLA_EGG_PATH)
try:
    import carla
    print("✅ Carla模块加载成功！")
except Exception as e:
    print(f"❌ Carla导入失败：{e}")
    sys.exit(1)


# 2. 核心功能函数（修复参数传递）
# --------------------------
def init_carla_camera():
    """初始化Carla摄像头（开启同步模式，修复画面静止）"""
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world()

        # 解除Carla服务端暂停，开启同步模式
        world_settings = world.get_settings()
        world_settings.synchronous_mode = True
        world_settings.fixed_delta_seconds = 0.05
        world.apply_settings(world_settings)
        print("✅ Carla仿真模式已开启（同步模式+20fps）")

        # 创建摄像头蓝图（关闭缓存）
        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.0')  # 无延迟

        # 摄像头位置
        spawn_point = carla.Transform(
            carla.Location(x=10.0, y=0.0, z=5.0),
            carla.Rotation(pitch=-10.0, yaw=0.0, roll=0.0)
        )

        # 生成摄像头Actor
        camera = world.spawn_actor(camera_bp, spawn_point)
        image_data = {'image': None, 'new_frame': False}

        # 回调函数：仅接收最新帧
        def camera_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (480, 640, 4))[:, :, :3][:, :, ::-1]
            image_data['image'] = array
            image_data['new_frame'] = True

        camera.listen(camera_callback)
        print("✅ Carla摄像头初始化成功！画面将实时刷新")
        return client, world, camera, image_data
    except Exception as e:
        print(f"❌ Carla摄像头初始化失败：{e}")
        print("⚠️  请确保：1. CarlaUE4.exe已启动  2. 地图加载完成")
        return None, None, None, None

def simulate_detection(image):
    """模拟检测框（随画面动态刷新）"""
    h, w = image.shape[:2]
    t = time.time() % 10
    offset_x = int(np.sin(t) * 10)
    offset_y = int(np.cos(t) * 5)
    boxes = [
        [w//4 + offset_x, h//4 + offset_y, w//3 + offset_x, h//3 + offset_y],
        [w//2 - offset_x, h//2 - offset_y, w//1.5 - offset_x, h//1.5 - offset_y],
    ]
    return np.array(boxes)

def draw_boxes(image, boxes):
    """绘制动态检测框（参数完整：image + boxes）"""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, 639))
        y1 = max(0, min(y1, 479))
        x2 = max(0, min(x2, 639))
        y2 = max(0, min(y2, 479))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, "Real-time Carla View (20fps)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return image

# --------------------------
# 3. 主程序（修复draw_boxes参数传递）
# --------------------------
def main():
    print("===== V2X路侧感知系统（动态画面版）=====")

    # 初始化Carla
    client, world, camera, image_data = init_carla_camera()
    if camera is None:
        return

    # 主循环（同步仿真帧）
    print("✅ 系统启动成功！按 'q' 键退出 | 可在CarlaUE4.exe中移动视角")
    try:
        while True:
            # 触发Carla仿真步，生成新帧
            world.tick()
            time.sleep(0.01)

            # 处理新帧（核心修复：传递完整参数）
            if image_data['new_frame'] and image_data['image'] is not None:
                frame = image_data['image'].copy()
                image_data['new_frame'] = False

                # 1. 生成检测框
                boxes = simulate_detection(frame)
                # 2. 绘制检测框（修复：传递frame + boxes两个参数）
                frame = draw_boxes(frame, boxes)

                # 显示动态画面
                cv2.imshow("V2X Road-Side Perception (Dynamic View)", frame)

            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n⚠️  用户手动中断程序")
    finally:
        # 恢复Carla默认设置
        if world:
            world_settings = world.get_settings()
            world_settings.synchronous_mode = False
            world.apply_settings(world_settings)
        # 清理资源
        if camera:
            camera.stop()
            camera.destroy()
        cv2.destroyAllWindows()
        print("✅ 系统已退出，Carla设置已恢复默认！")

if __name__ == "__main__":
    main()