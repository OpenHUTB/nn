"""
第一次作业：车辆生成与基础自动驾驶
环境：
/Carla
功能：生成车辆 + 自动驾驶 + 摄像头绑定 + 自动视角
"""
import carla
import time

def run():
    # 1. 连接模拟器
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 2. 生成车辆
    spawn_point = world.get_map().get_spawn_points()[0]
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # 3. 开启自动驾驶
    vehicle.set_autopilot(True)

    # 4. 生成车头摄像头（保证你能看到车）
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_transform = carla.Transform(carla.Location(x=5, z=2.0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # 5. 自动把视角切换到摄像头前
    world.get_spectator().set_transform(camera.get_transform())

    print("✅ 作业运行成功：车辆已生成，自动驾驶开启，视角已切换")

    time.sleep(10)   # 运行 10 秒

    # 6. 退出前清理
    vehicle.destroy()
    camera.destroy()
    print("✅ 车辆已销毁，作业结束")

    return "作业完成"

# 必须保留 - 用于老师平台提交
if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print("❌ 作业运行出错：", e)