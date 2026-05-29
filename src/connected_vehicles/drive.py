import sys
import time
import signal
import keyboard
from pathlib import Path
import carla
import vehicle_status_gui as vsg
from utils import calculate_vehicle_speed_kmh  # 仅保留车速计算，移除碰撞车速相关
from collision_monitor import (
    create_collision_sensor,
    stop_collision_monitor,
    get_collision_occurred,
    reset_collision_occurred
)
from environment_controller import set_weather, get_current_environment_state
from traffic_light_controller import check_red_light_violation

# 全局变量
exit_flag = False
car = None
collision_sensor = None

# 天气循环配置 + 防抖标记
WEATHER_LIST = ["clear", "rain", "fog", "night"]
current_weather_idx = 0
w_key_triggered = False
c_key_triggered = False

# 信号处理（退出时清理资源）
def handle_exit(_sig, _frame):
    global exit_flag, car, collision_sensor
    if exit_flag:
        return
    exit_flag = True
    # 停止车况监控窗口
    vsg.stop_gui()

    print("\n⚠️  程序终止信号触发")
    stop_collision_monitor()
    if car is not None:
        car.destroy()
    if collision_sensor is not None:
        collision_sensor.destroy()
    sys.exit(0)

# 注册退出信号
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

def main():
    global exit_flag, car, collision_sensor, current_weather_idx, w_key_triggered, c_key_triggered

    try:
        # 初始化CARLA路径
        BASE_DIR = Path(__file__).parent
        sys.path.append(str(BASE_DIR / "PythonAPI" / "carla" / "dist"))

        # ========== 仅保留核心CARLA连接逻辑 ==========
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        # ========== 车辆初始化 ==========
        carla_map = world.get_map()
        # 调整车辆生成位置（后退10米）
        spawn_point = carla_map.get_spawn_points()[0]
        spawn_point.location -= spawn_point.get_forward_vector() * 10

        car_bp = world.get_blueprint_library().filter("vehicle")[0]
        car = world.spawn_actor(car_bp, spawn_point)
        spectator = world.get_spectator()

        # 初始化碰撞传感器
        collision_sensor = create_collision_sensor(world, car)

        # 初始化天气（默认晴天）
        set_weather(world, WEATHER_LIST[current_weather_idx])

        # ========== 仅保留车况监控窗口启动（核心） ==========
        vsg.create_status_window()

        # ========== 操作说明 ==========
        MAX_SPEED = 100
        print("=" * 80)
        print("操作说明：")
        print("↑：前进 | ↓：倒车 | ←：左转 | →：右转 | 空格键：急刹 | C：模拟碰撞 | ESC：退出")
        print("W键：循环切换天气（晴天→雨天→雾天→夜间→晴天...）")
        print("📊 实时监测：车速 | 天气 | 能见度 | 碰撞状态 | 红绿灯违规")
        print("=" * 80)

        print_counter = 0
        red_light_violation_flag = False

        # ========== 主循环 ==========
        while not exit_flag:
            # 计算当前车速
            current_speed = calculate_vehicle_speed_kmh(car)

            # 更新GUI的车速数据
            vsg.update_vehicle_status("speed", current_speed)

            # 定期打印实时信息（每20帧）
            print_counter += 1
            if print_counter % 20 == 0:
                env_state = get_current_environment_state()
                env_info = f"天气：{env_state['weather_type']} | 能见度：{env_state['visibility']}%"
                # 从GUI获取碰撞车速（不再用utils的全局变量）
                collision_speed = vsg.vehicle_status["collision_speed"]
                collision_info = f"碰撞车速：{collision_speed} km/h"
                print(f"\r速度：{current_speed:.1f} km/h | {env_info} | {collision_info} | 闯红灯：否", end="")

                # 更新GUI的天气、能见度数据
                vsg.update_vehicle_status("weather", env_state['weather_type'])
                vsg.update_vehicle_status("visibility", env_state['visibility'])

            # ========== 车辆控制逻辑 ==========
            ctrl = carla.VehicleControl()
            ctrl.hand_brake = False
            ctrl.gear = 1

            if keyboard.is_pressed("space"):
                ctrl.brake = 1.0
                ctrl.hand_brake = True
                ctrl.throttle = 0.0
                print(f"\n🛑 急刹触发！当前车速：{current_speed} km/h")
            else:
                if keyboard.is_pressed("up"):
                    ctrl.throttle = 1.0
                    ctrl.reverse = False
                    ctrl.gear = 1
                elif keyboard.is_pressed("down"):
                    ctrl.throttle = 1.0
                    ctrl.reverse = True
                    ctrl.gear = -1
                else:
                    ctrl.throttle = 0.0

                if keyboard.is_pressed("left"):
                    ctrl.steer = -0.5
                elif keyboard.is_pressed("right"):
                    ctrl.steer = 0.5
                else:
                    ctrl.steer = 0.0

                ctrl.brake = 1.0 if keyboard.is_pressed("s") else 0.0

            # 限速
            if current_speed > MAX_SPEED:
                ctrl.throttle = 0.2

            car.apply_control(ctrl)

            # 视角跟随车辆
            trans = car.get_transform()
            cam_loc = trans.location - trans.get_forward_vector() * 10 + carla.Location(z=4)
            cam_rot = trans.rotation
            cam_rot.pitch = -20
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # ========== 碰撞处理 ==========
            collision_occurred = get_collision_occurred()
            # 更新GUI的碰撞状态（碰撞车速已在collision_monitor中直接更新）
            vsg.update_vehicle_status("collision_occurred", collision_occurred)

            if collision_occurred:
                reset_collision_occurred()

            # ========== 模拟碰撞（防抖） ==========
            if keyboard.is_pressed("c") and not c_key_triggered:
                c_key_triggered = True
                if current_speed > 0:
                    # 直接更新GUI的碰撞车速
                    vsg.update_vehicle_status("collision_speed", current_speed)
                    print(f"\n⚠️  模拟碰撞：车速{current_speed} km/h")
                    # 模拟碰撞时更新GUI碰撞状态
                    vsg.update_vehicle_status("collision_occurred", True)
                else:
                    print("\n⚠️  车辆静止，无法模拟碰撞")
            elif not keyboard.is_pressed("c"):
                c_key_triggered = False

            # ========== W键循环切换天气 ==========
            if keyboard.is_pressed("w") and not w_key_triggered:
                w_key_triggered = True
                current_weather_idx = (current_weather_idx + 1) % len(WEATHER_LIST)
                set_weather(world, WEATHER_LIST[current_weather_idx])
            elif not keyboard.is_pressed("w"):
                w_key_triggered = False

            # ========== 红绿灯检测 ==========
            red_light_violation = check_red_light_violation(world, car)
            # 更新GUI的闯红灯状态
            vsg.update_vehicle_status("red_light_violation", red_light_violation)

            if red_light_violation and not red_light_violation_flag:
                red_light_violation_flag = True
                print("\n🚨 检测到闯红灯行为！")
            elif not red_light_violation:
                red_light_violation_flag = False

            # ========== 退出程序 ==========
            if keyboard.is_pressed("esc") and not exit_flag:
                exit_flag = True
                break

            time.sleep(0.01)

    # ========== 异常处理 ==========
    except ConnectionRefusedError:
        print("\n❌ 连接CARLA失败！请确认：")
        print("1. CARLA模拟器已启动（端口2000）")
        print("2. 本地IP/端口配置正确（127.0.0.1:2000）")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序异常：{str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细错误栈，便于排查
    finally:
        # 确保GUI停止
        vsg.stop_gui()
        # 清理CARLA资源
        if collision_sensor is not None:
            collision_sensor.destroy()
        if car is not None:
            car.destroy()
        stop_collision_monitor()
        print("\n✅ 程序正常退出")

if __name__ == "__main__":
    main()