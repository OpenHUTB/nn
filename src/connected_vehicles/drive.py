import sys
from pathlib import Path
import carla
import time
import keyboard

# 导入独立的红绿灯模块和碰撞监测模块
from traffic_light_controller import set_all_traffic_lights, get_current_light_state
from collision_monitor import create_collision_sensor, stop_collision_monitor  # 新增导入

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "PythonAPI" / "carla" / "dist"))

# 连接CARLA
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)
world = client.get_world()
carla_map = world.get_map()

# 车辆生成
road_spawns = carla_map.get_spawn_points()
spawn_point = road_spawns[0]

bp_lib = world.get_blueprint_library()
car_bp = bp_lib.filter("vehicle")[0]
car = world.spawn_actor(car_bp, spawn_point)
spectator = world.get_spectator()

# 为车辆挂载碰撞传感器（新增）
collision_sensor = create_collision_sensor(world, car)

print("↑前进 ↓倒车 ←左转 →右转  ESC退出")
print("S键刹车 | 限速50km/h | 红绿灯10秒切换")

MAX_SPEED_KMH = 50

# 设置红绿灯（调用外部函数）
set_all_traffic_lights(world, green_time=10.0, red_time=10.0, yellow_time=2.0)

print_counter = 0

try:
    while True:
        # 速度计算
        velocity = car.get_velocity()
        speed_ms = (velocity.x**2 + velocity.y**2)**0.5
        current_speed = 3.6 * speed_ms

        # 打印信息（每20帧一次）
        print_counter += 1
        if print_counter % 20 == 0:
            light_state = get_current_light_state(car)
            print(f"\r速度：{current_speed:.1f} km/h | 灯：{light_state}", end="")

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

        if keyboard.is_pressed("s"):
            ctrl.brake = 1.0
        else:
            ctrl.brake = 0.0

        # 限速
        if current_speed > MAX_SPEED_KMH:
            ctrl.throttle = 0.0

        car.apply_control(ctrl)

        # 视角
        trans = car.get_transform()
        forward = trans.get_forward_vector()
        camera_loc = trans.location - forward * 10 + carla.Location(z=4)
        camera_rot = trans.rotation
        camera_rot.pitch = -20
        spectator.set_transform(carla.Transform(camera_loc, camera_rot))

        time.sleep(0.02)

        if keyboard.is_pressed("esc"):
            break

finally:
    # 销毁传感器和车辆
    collision_sensor.destroy()
    car.destroy()
    # 停止碰撞监测，确保剩余数据写入文件（新增）
    stop_collision_monitor()
    print("\n✅ 退出成功（碰撞日志已保存到 collision_logs.txt）")