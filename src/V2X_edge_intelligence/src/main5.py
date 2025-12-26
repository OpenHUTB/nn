#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA 0.9.10 车路协同避障
"""
import sys
import time
import math

# ====================== 1. 导入CARLA（无绝对路径，依赖环境配置） ======================
try:
    import carla

    print("✅ CARLA模块导入成功！")
except ImportError as e:
    print("❌ CARLA模块导入失败！请按以下步骤配置环境：")
    print("  1. 确保CARLA 0.9.10服务器已启动")
    print("  2. 将CARLA安装目录下的PythonAPI路径加入sys.path，示例：")
    print("     sys.path.append('/path/to/CARLA_0.9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg')")
    print("  3. 或设置环境变量PYTHONPATH包含上述egg文件路径")
    sys.exit(1)
except Exception as e:
    print(f"❌ 导入CARLA时发生未知错误：{e}")
    sys.exit(1)

# ====================== 2. 核心参数（远距离停止+渐进减速） ======================
DECEL_DISTANCE = 20.0  # 距离<20米开始减速（提前缓冲）
STOP_DISTANCE = 12.0  # 距离<12米完全停止（远离蓝车，不撞）
NORMAL_THROTTLE = 0.7  # 正常直行油门
DECEL_THROTTLE = 0.1  # 减速阶段油门（缓慢靠近）
OBSTACLE_DISTANCE = 25.0  # 蓝车在红车同车道正前方25米（更远初始距离）
BRAKE_FORCE = 1.0  # 满刹车（停止彻底）


# ====================== 3. 计算两车距离 ======================
def calculate_distance(actor1, actor2):
    loc1 = actor1.get_transform().location
    loc2 = actor2.get_transform().location
    return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)


# ====================== 4. 主程序（远距离停止+渐进减速） ======================
def main():
    try:
        # 1. 连接CARLA+加载地图
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.load_world('Town01')
        world.set_weather(carla.WeatherParameters.ClearNoon)
        print("✅ 连接CARLA成功！加载Town01场景")

        # 2. 清理残留Actor
        for actor in world.get_actors():
            if actor.type_id in ['vehicle.*', 'static.prop.*', 'sensor.*']:
                actor.destroy()

        # 3. 生成红色主车（同车道起点，手动挂前进挡）
        blueprint_lib = world.get_blueprint_library()
        main_car_bp = blueprint_lib.filter('vehicle.tesla.model3')[0]
        main_car_bp.set_attribute('color', '255,0,0')  # 红色
        spawn_points = world.get_map().get_spawn_points()
        main_car_spawn = spawn_points[5]  # 开阔直车道生成点（无围栏）
        main_car = world.spawn_actor(main_car_bp, main_car_spawn)

        # 适配0.9.10：手动挂前进挡+解除手刹
        init_control = carla.VehicleControl(
            throttle=NORMAL_THROTTLE,
            steer=0.0,  # 全程直行，不转向
            manual_gear_shift=True,  # 开启手动换挡
            gear=1,  # 前进挡
            hand_brake=False,
            reverse=False
        )
        main_car.apply_control(init_control)
        print("✅ 生成红色主车：同车道起点，手动挂前进挡（直行）")

        # 4. 生成蓝色障碍车（红车同车道正前方25米，y坐标一致=同车道）
        obstacle_car_bp = blueprint_lib.filter('vehicle.tesla.model3')[0]
        obstacle_car_bp.set_attribute('color', '0,0,255')  # 蓝色
        obstacle_transform = carla.Transform(
            carla.Location(
                x=main_car_spawn.location.x + OBSTACLE_DISTANCE,  # 正前方25米
                y=main_car_spawn.location.y,  # 同一车道（y坐标一致）
                z=main_car_spawn.location.z
            ),
            main_car_spawn.rotation
        )
        obstacle_car = world.spawn_actor(obstacle_car_bp, obstacle_transform)
        obstacle_car.apply_control(carla.VehicleControl(hand_brake=True))  # 蓝车静止
        print(f"✅ 生成蓝色障碍车：红车同车道正前方{OBSTACLE_DISTANCE}米")

        # 5. 生成路侧边缘节点（V2X感知设备）
        edge_node_bp = blueprint_lib.filter('static.prop.*')[0]
        edge_node_transform = carla.Transform(
            carla.Location(
                x=main_car_spawn.location.x + 15,
                y=main_car_spawn.location.y + 3,
                z=3.0
            ),
            main_car_spawn.rotation
        )
        edge_node = world.spawn_actor(edge_node_bp, edge_node_transform)
        print("✅ 生成路侧边缘节点（感知障碍）")

        # 6. 初始近视角（紧贴红车，看清同车道蓝车）
        spectator = world.get_spectator()
        spectator_transform = carla.Transform(
            carla.Location(
                x=main_car_spawn.location.x + 4,
                y=main_car_spawn.location.y,
                z=main_car_spawn.location.z + 6  # 稍高，看清25米外蓝车
            ),
            carla.Rotation(pitch=-45, yaw=main_car_spawn.rotation.yaw)  # 直视同车道
        )
        spectator.set_transform(spectator_transform)
        print("✅ 初始视角设置完成：紧贴红车，看清同车道蓝车")

        # 7. 运行提示
        print("\n======= 车路协同避障仿真（远距离停止版） =======")
        print(f"✅ 红蓝车：同一车道，蓝车在红车正前方{OBSTACLE_DISTANCE}米")
        print("✅ 红车逻辑：直行→20米处减速→12米处完全停止（远离蓝车不撞）")
        print("✅ 镜头：自由操作（左键旋转/滚轮缩放/WASD平移）")
        print("✅ 退出方式：Ctrl+C 停止程序")
        print("==============================================\n")

        main_car_control = init_control
        is_stopped = False  # 红车停止标记

        while True:
            # 计算红车与蓝车的实时距离
            current_distance = calculate_distance(main_car, obstacle_car)

            # 核心逻辑：渐进减速+远距离停止（避免碰撞）
            if not is_stopped:
                if current_distance > DECEL_DISTANCE:
                    # 阶段1：距离>20米，正常直行（无减速）
                    main_car_control.throttle = NORMAL_THROTTLE
                    main_car_control.brake = 0.0
                    current_speed = math.hypot(main_car.get_velocity().x, main_car.get_velocity().y)
                    print(f"\r【直行中】距离蓝车：{current_distance:.1f}米 | 当前速度：{current_speed:.2f}m/s", end="")
                elif DECEL_DISTANCE >= current_distance > STOP_DISTANCE:
                    # 阶段2：20米≥距离>12米，渐进减速（缓慢靠近）
                    main_car_control.throttle = DECEL_THROTTLE
                    main_car_control.brake = 0.0
                    current_speed = math.hypot(main_car.get_velocity().x, main_car.get_velocity().y)
                    print(f"\r【减速中】距离蓝车：{current_distance:.1f}米 | 当前速度：{current_speed:.2f}m/s", end="")
                else:
                    # 阶段3：距离≤12米，满刹车完全停止（远离蓝车，不撞）
                    main_car_control.throttle = 0.0
                    main_car_control.brake = BRAKE_FORCE
                    print(f"\r【已停止】距离蓝车：{current_distance:.1f}米 → 远离蓝车，完全停止", end="")
                    is_stopped = True
            else:
                # 保持停止状态，避免再次移动
                main_car_control.throttle = 0.0
                main_car_control.brake = BRAKE_FORCE
                print(f"\r【保持停止】距离蓝车：{current_distance:.1f}米 | 红车静止不动", end="")

            # 持续发送控制指令，确保状态生效
            main_car.apply_control(main_car_control)
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\n🛑 程序终止，清理资源...")
    except Exception as e:
        print(f"\n⚠️  运行错误：{e} | 请确认CARLA 0.9.10已启动（localhost:2000）")
    finally:
        # 清理所有资源
        for actor_name in ['main_car', 'obstacle_car', 'edge_node']:
            if actor_name in locals():
                locals()[actor_name].destroy()
        print("✅ 资源清理完成，程序退出！")


# ====================== 程序入口（仅main.py） ======================
if __name__ == "__main__":
    main()