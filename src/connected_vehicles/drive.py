import sys

sys.path.append(r"D:\hutb\PythonAPI\carla\dist")
import carla
import time
import keyboard

# 连接CARLA
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)
world = client.get_world()
carla_map = world.get_map()

# 车辆只在道路合法点位生成
road_spawns = carla_map.get_spawn_points()
spawn_point = road_spawns[0]

# 生成车辆
bp_lib = world.get_blueprint_library()
car_bp = bp_lib.filter("vehicle")[0]
car = world.spawn_actor(car_bp, spawn_point)
spectator = world.get_spectator()

print("↑前进 ↓倒车 ←左转 →右转  ESC退出")

try:
    while True:
        # 车辆控制
        ctrl = carla.VehicleControl()
        if keyboard.is_pressed("up"):
            ctrl.throttle = 1.0
        else:
            ctrl.throttle = 0.0

        if keyboard.is_pressed("down"):
            ctrl.reverse = True
            ctrl.throttle = 1.0
        else:
            ctrl.reverse = False

        if keyboard.is_pressed("left"):
            ctrl.steer = -0.45
        elif keyboard.is_pressed("right"):
            ctrl.steer = 0.45
        else:
            ctrl.steer = 0.0

        car.apply_control(ctrl)

        # ==============================================
        # 视角在车正后上方
        # ==============================================
        trans = car.get_transform()
        forward = trans.get_forward_vector()

        # 相机固定在车后上方
        camera_loc = trans.location - forward * 10 + carla.Location(z=4)

        # 相机看向车
        camera_rot = trans.rotation
        camera_rot.pitch = -20  # 固定俯视角度

        spectator.set_transform(carla.Transform(camera_loc, camera_rot))

        time.sleep(0.02)
        if keyboard.is_pressed("esc"):
            break

finally:
    car.destroy()
    print("✅ 程序退出，车辆销毁")