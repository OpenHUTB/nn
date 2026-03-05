#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA 0.9.10+ 自动驾驶控制程序
特性：
1. 完全动态加载CARLA，无任何硬编码绝对路径
2. 增强功能：多地图切换、实时数据监控、日志记录、智能避障
3. 保留核心：晚转弯、大转向角度、平稳速度控制
"""
import sys
import os
import time
import math
import logging
from datetime import datetime
import carla

# ====================== 1. 日志配置（新增功能） ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'carla_drive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ====================== 2. CARLA动态加载（优化版，无绝对路径） ======================
def load_carla_dynamically():
    """动态加载CARLA，优先环境变量，其次相对路径"""
    try:
        import carla
        logger.info("✅ CARLA模块已直接加载")
        return carla
    except ImportError:
        # 动态路径列表（无任何硬编码绝对路径）
        carla_search_paths = [
            os.path.join(os.environ.get('CARLA_ROOT', ''), 'PythonAPI', 'carla', 'dist'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../PythonAPI/carla/dist'),
            os.path.expanduser('~/CARLA/PythonAPI/carla/dist'),
            os.path.expanduser('~/Documents/CARLA/PythonAPI/carla/dist'),
            os.path.expanduser('~/carla/PythonAPI/carla/dist'),
            os.path.expanduser('~/.local/share/carla/PythonAPI/carla/dist')
        ]

        # 遍历查找egg文件
        carla_egg_path = None
        for search_path in carla_search_paths:
            if not search_path or not os.path.exists(search_path):
                continue
            for file in os.listdir(search_path):
                if file.endswith('.egg') and 'carla' in file:
                    carla_egg_path = os.path.join(search_path, file)
                    break
            if carla_egg_path:
                break

        if carla_egg_path:
            sys.path.append(carla_egg_path)
            import carla
            logger.info(f"✅ 动态加载CARLA成功：{carla_egg_path}")
            return carla
        else:
            error_msg = (
                "\n❌ CARLA加载失败！请按以下方式配置：\n"
                "1. 配置CARLA_ROOT环境变量（推荐）：\n"
                "   Windows: set CARLA_ROOT=你的CARLA安装目录\n"
                "   Linux/Mac: export CARLA_ROOT=你的CARLA安装目录\n"
                "2. 确保PythonAPI路径正确：PythonAPI/carla/dist 下有carla-*.egg文件"
            )
            logger.error(error_msg)
            sys.exit(1)


# 执行动态加载
carla = load_carla_dynamically()

# ====================== 3. 配置参数（保留核心，新增多地图配置） ======================
# 速度参数
BASE_SPEED = 1.5  # 直道速度
CURVE_TARGET_SPEED = 1.0  # 弯道速度
SPEED_DEADZONE = 0.1
ACCELERATION_FACTOR = 0.04
DECELERATION_FACTOR = 0.06
SPEED_TRANSITION_RATE = 0.03

# 转向参数（晚转弯+大角度核心）
LOOKAHEAD_DISTANCE = 20.0
WAYPOINT_STEP = 1.0
CURVE_DETECTION_THRESHOLD = 2.0
TURN_TRIGGER_DISTANCE_IDX = 4  # 前方5米触发转向
STEER_ANGLE_MAX = 0.85  # 最大转向角
STEER_RESPONSE_FACTOR = 0.4  # 转向响应速度
STEER_AMPLIFY = 1.6  # 转向放大系数
MIN_STEER = 0.2  # 最小转向角

# 生成点偏移
SPAWN_OFFSET_X = -2.0
SPAWN_OFFSET_Y = 0.0
SPAWN_OFFSET_Z = 0.0

# 新增：多地图配置
SUPPORTED_MAPS = {
    "Town01": "城镇1（简单道路）",
    "Town02": "城镇2（乡村道路）",
    "Town03": "城镇3（高速公路）",
    "Town04": "城镇4（混合道路）"
}
DEFAULT_MAP = "Town01"

# 新增：控制配置
CONTROL_CONFIG = {
    "init_control_times": 12,
    "init_control_interval": 0.05,
    "init_total_delay": 0.8,
    "normal_throttle": 0.85,
    "avoid_throttle": 0.5,
    "avoid_steer": 0.6,
    "loop_interval": 0.008,
    "detect_distance": 10.0,
    "stuck_reset_dist": 2.0
}


# ====================== 4. 核心工具函数（增强版） ======================
def get_road_direction_ahead(vehicle, world):
    """获取前方道路方向，晚转弯逻辑"""
    vehicle_transform = vehicle.get_transform()
    carla_map = world.get_map()

    waypoints = []
    current_wp = carla_map.get_waypoint(vehicle_transform.location)
    next_wp = current_wp

    for _ in range(int(LOOKAHEAD_DISTANCE / WAYPOINT_STEP)):
        next_wps = next_wp.next(WAYPOINT_STEP)
        if not next_wps:
            break
        next_wp = next_wps[0]
        waypoints.append(next_wp)

    if len(waypoints) < 3:
        return vehicle_transform.rotation.yaw, False, 0.0

    target_wp_idx = min(TURN_TRIGGER_DISTANCE_IDX, len(waypoints) - 1)
    target_wp = waypoints[target_wp_idx]
    target_yaw = target_wp.transform.rotation.yaw

    current_yaw = vehicle_transform.rotation.yaw
    yaw_diff = target_yaw - current_yaw
    yaw_diff = (yaw_diff + 180) % 360 - 180
    is_curve = abs(yaw_diff) > CURVE_DETECTION_THRESHOLD

    return target_yaw, is_curve, yaw_diff


def calculate_steer_angle(current_yaw, target_yaw):
    """计算超大角度转向"""
    yaw_diff = target_yaw - current_yaw
    yaw_diff = (yaw_diff + 180) % 360 - 180

    steer = (yaw_diff / 180.0 * STEER_ANGLE_MAX) * STEER_AMPLIFY
    steer = max(-STEER_ANGLE_MAX, min(STEER_ANGLE_MAX, steer))

    if abs(steer) > 0.05 and abs(steer) < MIN_STEER:
        steer = MIN_STEER * (1 if steer > 0 else -1)

    return steer


def detect_obstacle_enhanced(vehicle, world, detect_distance=10.0):
    """增强版障碍物检测：检测车辆、行人、静态障碍物"""
    trans = vehicle.get_transform()
    vehicle_location = trans.location
    vehicle_forward = trans.get_forward_vector()

    # 1. 检测道路合法性
    for check_dist in range(2, int(detect_distance) + 1, 2):
        check_loc = vehicle_location + vehicle_forward * check_dist
        waypoint = world.get_map().get_waypoint(check_loc, project_to_road=False)
        if not waypoint or waypoint.lane_type != carla.LaneType.Driving:
            return True

    # 2. 检测其他车辆/行人（新增功能）
    actors = world.get_actors()
    for actor in actors:
        if actor.type_id.startswith(("vehicle", "walker")) and actor.id != vehicle.id:
            actor_loc = actor.get_location()
            distance = vehicle_location.distance(actor_loc)
            # 检测前方detect_distance米内的障碍物
            if distance < detect_distance:
                # 计算障碍物在车辆前方的角度
                vec = actor_loc - vehicle_location
                dot = vec.x * vehicle_forward.x + vec.y * vehicle_forward.y
                if dot > 0:  # 只检测前方障碍物
                    return True

    return False


def follow_vehicle_enhanced(vehicle, spectator, follow_mode="third_person"):
    """增强版视角跟随：支持第三人称/俯视视角"""
    trans = vehicle.get_transform()
    if follow_mode == "third_person":
        # 第三人称视角
        spectator_loc = carla.Location(
            x=trans.location.x - math.cos(math.radians(trans.rotation.yaw)) * 7,
            y=trans.location.y - math.sin(math.radians(trans.rotation.yaw)) * 7,
            z=trans.location.z + 4.5
        )
        spectator_rot = carla.Rotation(pitch=-30, yaw=trans.rotation.yaw)
    elif follow_mode == "top_down":
        # 俯视视角
        spectator_loc = carla.Location(
            x=trans.location.x,
            y=trans.location.y,
            z=40.0
        )
        spectator_rot = carla.Rotation(pitch=-85, yaw=trans.rotation.yaw)

    spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))


def print_drive_status(vehicle, run_time, has_obstacle, steer):
    """打印增强版行驶状态（新增功能）"""
    velocity = vehicle.get_velocity()
    current_speed_mps = math.hypot(velocity.x, velocity.y)
    current_speed_kmh = current_speed_mps * 3.6
    location = vehicle.get_location()

    status = "⚠️ 避障中" if has_obstacle else "✅ 正常行驶"
    status_msg = (
        f"\r{status} | 行驶时长：{run_time:.0f}s "
        f"| 速度：{current_speed_kmh:.0f}km/h "
        f"| 转向：{steer:.2f} "
        f"| 位置：({location.x:.1f}, {location.y:.1f})"
    )
    print(status_msg, end="")
    logger.info(status_msg.strip())


# ====================== 5. 主函数（增强版） ======================
def main(selected_map=DEFAULT_MAP):
    """主函数：增强版自动驾驶逻辑"""
    # 初始化资源
    client = None
    world = None
    vehicle = None
    camera_sensor = None
    collision_sensor = None
    spectator = None
    is_vehicle_alive = False
    run_time = 0

    try:
        # 1. 连接CARLA服务器
        client = carla.Client("localhost", 2000)
        client.set_timeout(60.0)
        world = client.load_world(selected_map)
        logger.info(f"✅ 成功连接CARLA，加载地图：{selected_map} ({SUPPORTED_MAPS[selected_map]})")

        # 配置世界设置
        world_settings = world.get_settings()
        world_settings.synchronous_mode = False
        world_settings.fixed_delta_seconds = 0.1
        world.apply_settings(world_settings)

        # 设置天气（新增功能）
        world.set_weather(carla.WeatherParameters.ClearNoon)
        logger.info("✅ 设置天气为：晴朗正午")

        # 2. 清理旧Actor
        for actor in world.get_actors().filter('vehicle.*'):
            actor.destroy()
        logger.info("✅ 清理旧车辆完成")

        # 3. 生成车辆
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute('color', '255,0,0')  # 红色车身
        logger.info("✅ 选择特斯拉Model3，红色车身")

        # 获取生成点并偏移
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise Exception("❌ 未找到生成点")

        original_spawn_point = spawn_points[0]
        spawn_point = carla.Transform(
            carla.Location(
                x=original_spawn_point.location.x + SPAWN_OFFSET_X,
                y=original_spawn_point.location.y + SPAWN_OFFSET_Y,
                z=original_spawn_point.location.z + SPAWN_OFFSET_Z
            ),
            original_spawn_point.rotation
        )
        logger.info(f"✅ 生成点偏移：左移{abs(SPAWN_OFFSET_X)}米")

        # 重试生成车辆
        max_spawn_retry = 5
        for retry in range(max_spawn_retry):
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                if vehicle and vehicle.is_alive:
                    vehicle.set_simulate_physics(True)
                    vehicle.set_autopilot(False)
                    is_vehicle_alive = True
                    logger.info(f"✅ 车辆生成成功（重试{retry + 1}次），ID：{vehicle.id}")
                    break
            except Exception as e:
                if retry == max_spawn_retry - 1:
                    raise Exception(f"❌ 车辆生成失败：{e}")
                time.sleep(0.8)

        # 4. 初始化车辆控制
        control = carla.VehicleControl()
        control.hand_brake = False
        control.manual_gear_shift = False
        control.gear = 1

        # 激活车辆（确保能动）
        logger.info("🔋 激活车辆物理状态...")
        for _ in range(CONTROL_CONFIG["init_control_times"]):
            vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, steer=0.0, brake=0.0
            ))
            time.sleep(CONTROL_CONFIG["init_control_interval"])
        time.sleep(CONTROL_CONFIG["init_total_delay"])

        # 5. 设置视角
        spectator = world.get_spectator()
        follow_vehicle_enhanced(vehicle, spectator, "third_person")
        logger.info("✅ 视角已绑定车辆（第三人称）")

        # 6. 挂载传感器（新增功能：可选保存图片）
        # 碰撞传感器
        try:
            collision_bp = bp_lib.find("sensor.other.collision")
            collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

            def collision_cb(event):
                nonlocal steer
                logger.warning("💥 检测到碰撞，自动调整方向！")
                steer = -steer if abs(steer) > 0 else -CONTROL_CONFIG["avoid_steer"]
                vehicle.apply_control(carla.VehicleControl(
                    throttle=CONTROL_CONFIG["avoid_throttle"],
                    steer=steer,
                    brake=0.0
                ))

            collision_sensor.listen(collision_cb)
            logger.info("🛡️ 碰撞传感器挂载成功")
        except Exception as e:
            logger.warning(f"⚠️ 碰撞传感器挂载失败：{e}")

        # RGB摄像头（可选启用）
        enable_camera = False  # 可改为True启用
        if enable_camera:
            try:
                # 创建保存目录（相对路径，无绝对路径）
                camera_dir = os.path.join(os.path.dirname(__file__), "camera_images")
                os.makedirs(camera_dir, exist_ok=True)

                camera_bp = bp_lib.find("sensor.camera.rgb")
                camera_bp.set_attribute('image_size_x', '800')
                camera_bp.set_attribute('image_size_y', '600')
                camera_bp.set_attribute('fov', '90')
                camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
                camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

                def camera_callback(image):
                    image.save_to_disk(os.path.join(camera_dir, f"frame_{image.frame_number}.png"))

                camera_sensor.listen(camera_callback)
                logger.info(f"📹 摄像头挂载成功，图片保存至：{camera_dir}")
            except Exception as e:
                logger.warning(f"⚠️ 摄像头挂载失败：{e}")

        # 7. 核心自动驾驶循环
        logger.info("\n🚗 开始自动驾驶（按Ctrl+C停止）")
        print("\n" + "=" * 80)
        current_steer = 0.0
        current_target_speed = BASE_SPEED
        last_throttle = 0.0
        last_brake = 0.0
        steer = 0.0

        while True:
            # 检查车辆状态
            if not vehicle or not vehicle.is_alive:
                logger.error("❌ 车辆异常消失")
                break

            # 更新视角
            follow_vehicle_enhanced(vehicle, spectator, "third_person")

            # 障碍物检测（增强版）
            has_obstacle = detect_obstacle_enhanced(vehicle, world, CONTROL_CONFIG["detect_distance"])

            if has_obstacle:
                # 避障逻辑
                steer = CONTROL_CONFIG["avoid_steer"]
                throttle = CONTROL_CONFIG["avoid_throttle"]
            else:
                # 正常行驶逻辑
                # 获取道路方向
                target_yaw, is_curve, yaw_diff = get_road_direction_ahead(vehicle, world)

                # 弯道速度控制
                if is_curve:
                    current_target_speed = max(CURVE_TARGET_SPEED, current_target_speed - SPEED_TRANSITION_RATE)
                else:
                    current_target_speed = min(BASE_SPEED, current_target_speed + SPEED_TRANSITION_RATE / 2)
                    steer = steer * 0.9 if abs(steer) > 0.05 else 0.0

                # 计算转向
                target_steer = calculate_steer_angle(vehicle.get_transform().rotation.yaw, target_yaw)
                current_steer = current_steer + (target_steer - current_steer) * STEER_RESPONSE_FACTOR
                steer = current_steer
                throttle = CONTROL_CONFIG["normal_throttle"]

                # 速度控制
                current_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
                speed_error = current_target_speed - current_speed

                if abs(speed_error) < SPEED_DEADZONE:
                    control.throttle = last_throttle * 0.85
                    control.brake = 0.0
                elif speed_error > 0:
                    control.throttle = min(last_throttle + ACCELERATION_FACTOR, 0.25)
                    control.brake = 0.0
                else:
                    control.brake = min(last_brake + DECELERATION_FACTOR, 0.2)
                    control.throttle = 0.0

                last_throttle = control.throttle
                last_brake = control.brake

            # 应用控制
            control.steer = steer
            control.throttle = throttle
            vehicle.apply_control(control)

            # 卡停处理
            current_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
            if current_speed < 0.1:
                logger.warning("⚠️ 车辆卡停，重置位置")
                new_loc = vehicle.get_transform().location + carla.Location(x=CONTROL_CONFIG["stuck_reset_dist"])
                vehicle.set_transform(carla.Transform(new_loc, vehicle.get_transform().rotation))
                vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

            # 打印状态
            run_time += CONTROL_CONFIG["loop_interval"]
            print_drive_status(vehicle, run_time, has_obstacle, steer)

            time.sleep(CONTROL_CONFIG["loop_interval"])

    except KeyboardInterrupt:
        logger.info(f"\n🛑 手动终止程序，总行驶时长：{run_time:.0f}秒")
    except Exception as e:
        logger.error(f"\n❌ 程序异常：{str(e)}", exc_info=True)
        print("\n🔧 修复建议：")
        print("1. 关闭CARLA，结束任务管理器中的CarlaUE4.exe进程")
        print("2. 重启CARLA：CarlaUE4.exe -windowed -ResX=800 -ResY=600")
        print("3. 确保CARLA_ROOT环境变量配置正确")
    finally:
        # 资源清理
        logger.info("\n🧹 清理资源...")

        # 停车
        if vehicle and is_vehicle_alive:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            time.sleep(1)
            vehicle.destroy()
            logger.info("🗑️ 车辆已销毁")

        # 销毁传感器
        if collision_sensor:
            collision_sensor.stop()
            collision_sensor.destroy()
            logger.info("🗑️ 碰撞传感器已销毁")

        if camera_sensor:
            camera_sensor.stop()
            camera_sensor.destroy()
            logger.info("🗑️ 摄像头已销毁")

        # 恢复世界设置
        if world:
            world_settings = world.get_settings()
            world_settings.synchronous_mode = False
            world.apply_settings(world_settings)

        logger.info("✅ 所有资源清理完成！")
        print("\n✅ 程序正常退出")


# ====================== 运行入口 ======================
if __name__ == "__main__":
    # 支持命令行指定地图（新增功能）
    selected_map = DEFAULT_MAP
    if len(sys.argv) > 1 and sys.argv[1] in SUPPORTED_MAPS:
        selected_map = sys.argv[1]

    print(f"📌 即将启动CARLA自动驾驶程序")
    print(f"🗺️ 选择地图：{selected_map} ({SUPPORTED_MAPS[selected_map]})")
    print(f"💡 支持的地图：{', '.join(SUPPORTED_MAPS.keys())}")
    print(f"💡 示例：python {sys.argv[0]} Town02\n")

    main(selected_map)