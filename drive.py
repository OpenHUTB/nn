import sys
import time
import signal
import keyboard
from pathlib import Path
import carla
from utils import (
    calculate_vehicle_speed_kmh,
    set_global_collision_speed,
    get_global_collision_speed
)
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
WEATHER_LIST = ["clear", "rain", "fog", "night"]  # 天气循环顺序
current_weather_idx = 0  # 当前天气索引
w_key_triggered = False  # W键防抖
c_key_triggered = False  # C键防抖

# 信号处理（退出时清理资源）
def handle_exit(_sig, _frame):
    global exit_flag, car, collision_sensor
    if exit_flag:
        return
    exit_flag = True

    print("\n⚠️  程序终止信号触发")
    stop_collision_monitor()
    # 清理资源
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

        # 连接CARLA服务器
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
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

        # 配置
        MAX_SPEED = 100
        print("="*80)
        print("操作说明：")
        print("↑：前进 | ↓：倒车 | ←：左转 | →：右转 | 空格键：急刹 | C：模拟碰撞 | ESC：退出")
        print("W键：循环切换天气（晴天→雨天→雾天→夜间→晴天...）")
        print("📊 实时监测：车速 | 天气 | 能见度 | 碰撞状态 | 红绿灯违规")
        print("="*80)

        print_counter = 0

        # 主循环
        while not exit_flag:
            # 计算当前车速
            current_speed = calculate_vehicle_speed_kmh(car)

            # 打印实时信息
            print_counter += 1
            if print_counter % 20 == 0:
                env_state = get_current_environment_state()
                env_info = f"天气：{env_state['weather_type']} | 能见度：{env_state['visibility']}%"
                collision_info = f"碰撞车速：{get_global_collision_speed()} km/h"
                print(f"\r速度：{current_speed:.1f} km/h | {env_info} | {collision_info} | 闯红灯：否", end="")

            # ========== 重构车辆控制逻辑（适配CARLA底层规则） ==========
            ctrl = carla.VehicleControl()
            ctrl.hand_brake = False  # 重置手刹
            ctrl.gear = 1  # 默认前进挡

            # 1. 空格键急刹（最高优先级：手刹+满刹车）
            if keyboard.is_pressed("space"):
                ctrl.brake = 1.0  # 满刹车力度
                ctrl.hand_brake = True  # 拉手刹（强制急刹）
                ctrl.throttle = 0.0  # 急刹时切断油门
                print(f"\n🛑 急刹触发！当前车速：{current_speed} km/h")
            else:
                # 2. 方向键控制（无急刹时生效）
                # ↑前进：油门拉满，前进挡
                if keyboard.is_pressed("up"):
                    ctrl.throttle = 1.0
                    ctrl.reverse = False
                    ctrl.gear = 1
                # ↓倒车：油门拉满，倒挡（核心修复：倒车需同时设置reverse+gear）
                elif keyboard.is_pressed("down"):
                    ctrl.throttle = 1.0
                    ctrl.reverse = True
                    ctrl.gear = -1  # 显式设置倒挡
                # 无前进/倒车时，油门复位
                else:
                    ctrl.throttle = 0.0

                # ←左转 / →右转（转向灵敏度优化）
                if keyboard.is_pressed("left"):
                    ctrl.steer = -0.5  # 左转
                elif keyboard.is_pressed("right"):
                    ctrl.steer = 0.5   # 右转
                else:
                    ctrl.steer = 0.0   # 回正方向

                # 常规刹车（S键，可选保留）
                ctrl.brake = 1.0 if keyboard.is_pressed("s") else 0.0

            # 限速逻辑
            if current_speed > MAX_SPEED:
                ctrl.throttle = 0.2  # 超速时降低油门，而非直接归零

            # 强制应用控制指令（核心：CARLA需持续发送指令）
            car.apply_control(ctrl)

            # 视角跟随车辆
            trans = car.get_transform()
            cam_loc = trans.location - trans.get_forward_vector() * 10 + carla.Location(z=4)
            cam_rot = trans.rotation
            cam_rot.pitch = -20
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # ========== 碰撞处理 ==========
            if get_collision_occurred():
                reset_collision_occurred()

            # ========== 模拟碰撞（防抖） ==========
            if keyboard.is_pressed("c") and not c_key_triggered:
                c_key_triggered = True
                if current_speed > 0:
                    set_global_collision_speed(current_speed)
                    print(f"\n⚠️  模拟碰撞：车速{current_speed} km/h")
                else:
                    print("\n⚠️  车辆静止，无法模拟碰撞")
            elif not keyboard.is_pressed("c"):
                c_key_triggered = False

            # ========== W键循环切换天气 ==========
            if keyboard.is_pressed("w") and not w_key_triggered:
                w_key_triggered = True
                # 循环更新天气索引
                current_weather_idx = (current_weather_idx + 1) % len(WEATHER_LIST)
                # 应用新天气
                set_weather(world, WEATHER_LIST[current_weather_idx])
            elif not keyboard.is_pressed("w"):
                w_key_triggered = False

            # ========== 红绿灯检测 ==========
            if check_red_light_violation(world, car):
                print("\n🚨 检测到闯红灯行为！")

            # ========== 退出程序 ==========
            if keyboard.is_pressed("esc") and not exit_flag:
                exit_flag = True
                break

            # 小幅休眠，保证指令持续发送
            time.sleep(0.01)

    except ConnectionRefusedError:
        print("\n❌ 连接CARLA失败！请确认：")
        print("1. CARLA模拟器已启动（端口2000）")
        print("2. 本地IP/端口配置正确（127.0.0.1:2000）")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序异常：{str(e)}")
    finally:
        # 最终资源清理
        if collision_sensor is not None:
            collision_sensor.destroy()
        if car is not None:
            car.destroy()
        stop_collision_monitor()
        print("\n✅ 程序正常退出")

if __name__ == "__main__":
    main()