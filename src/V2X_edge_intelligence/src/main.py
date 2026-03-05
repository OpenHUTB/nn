# v2x_balance_zones.py（三区平均分配+低速精准控速）
import sys
import os
import time
import json
import math
import logging

# ===================== 1. 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ===================== 1. 动态配置CARLA路径（无硬编码绝对路径） =====================
def setup_carla_path():
    """
    动态查找并配置CARLA路径（优先级：
    1. 环境变量 CARLA_PYTHON_API
    2. 当前目录及子目录
    3. 用户主目录
    """
    # 尝试从环境变量获取
    carla_egg_env = os.getenv('CARLA_PYTHON_API')
    if carla_egg_env and os.path.exists(carla_egg_env):
        egg_path = carla_egg_env
        logger.info(f"🔍 从环境变量获取CARLA路径：{egg_path}")
    else:
        # 动态搜索常见位置（无硬编码绝对路径）
        search_paths = [
            os.getcwd(),  # 当前目录
            os.path.expanduser("~"),  # 用户主目录
            # 相对路径搜索（CARLA通常的PythonAPI相对位置）
            os.path.join(os.getcwd(), "PythonAPI", "carla", "dist"),
            os.path.join(os.path.dirname(os.getcwd()), "PythonAPI", "carla", "dist")
        ]

        egg_path = None
        # 搜索所有.py3.7相关的egg文件（适配0.9.10）
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
            for file in os.listdir(search_path):
                if file.startswith("carla-0.9.10-py3.7") and file.endswith(".egg"):
                    egg_path = os.path.join(search_path, file)
                    logger.info(f"🔍 自动找到CARLA egg文件：{egg_path}")
                    break
            if egg_path:
                break

    # 验证并添加到路径
    if egg_path and os.path.exists(egg_path):
        if egg_path not in sys.path:
            sys.path.insert(0, egg_path)
        logger.info(f"✅ CARLA egg路径已添加：{egg_path}")
        return True
    else:
        logger.error("\n❌ 未找到CARLA egg文件！")
        logger.info("📌 请通过以下方式配置：")
        logger.info("   1. 设置环境变量：CARLA_PYTHON_API=你的egg文件路径")
        logger.info("   2. 或将egg文件放到当前脚本目录")
        logger.info("   3. 确保CARLA版本为0.9.10（py3.7）")
        return False


# 配置CARLA路径
if not setup_carla_path():
    sys.exit(1)

# 导入CARLA（动态路径配置后）
try:
    import carla

    logger.info("✅ CARLA模块导入成功！")
except Exception as e:
    logger.error(f"\n❌ 导入CARLA失败：{str(e)}")
    sys.exit(1)


# ===================== 2. 核心：三区平均分配+低速精准控速 =====================
class RoadSideUnit:
    def __init__(self, carla_world, vehicle):
        self.world = carla_world
        self.vehicle = vehicle
        # 1. 三区坐标（等距分配，每区长度一致）
        spawn_loc = vehicle.get_location()
        # 高速区：生成位置前5-15米（长度10米）
        self.high_zone_start = carla.Location(spawn_loc.x, spawn_loc.y + 5, spawn_loc.z)
        self.high_zone_end = carla.Location(spawn_loc.x, spawn_loc.y + 15, spawn_loc.z)
        # 中速区：生成位置前15-25米（长度10米）
        self.mid_zone_start = carla.Location(spawn_loc.x, spawn_loc.y + 15, spawn_loc.z)
        self.mid_zone_end = carla.Location(spawn_loc.x, spawn_loc.y + 25, spawn_loc.z)
        # 低速区：生成位置前25-35米（长度10米）
        self.low_zone_start = carla.Location(spawn_loc.x, spawn_loc.y + 25, spawn_loc.z)
        self.low_zone_end = carla.Location(spawn_loc.x, spawn_loc.y + 35, spawn_loc.z)

        # 2. 三区计时（确保每区停留约10秒）
        self.current_zone = "high"  # 初始区：高速
        self.zone_start_time = time.time()
        self.zone_duration = 10  # 每区停留10秒（30秒测试，三区各10秒）
        self.speed_map = {"high": 40, "mid": 25, "low": 10}

    def get_balance_speed_limit(self):
        """核心：计时强制切换+位置双重判断，确保三区平均分配"""
        current_time = time.time()
        vehicle_loc = self.vehicle.get_location()
        vehicle_y = vehicle_loc.y  # 沿行驶方向的核心坐标

        # 1. 计时判断：每区停留10秒强制切换
        if current_time - self.zone_start_time > self.zone_duration:
            if self.current_zone == "high":
                self.current_zone = "mid"
            elif self.current_zone == "mid":
                self.current_zone = "low"
            elif self.current_zone == "low":
                self.current_zone = "high"  # 循环切换（避免一直停低速）
            self.zone_start_time = current_time  # 重置计时
            logger.info(f"⏰ 计时触发区域切换：{self.current_zone}")

        # 2. 位置双重验证：确保区域与位置匹配
        spawn_y = self.vehicle.get_location().y
        if spawn_y + 5 <= vehicle_y < spawn_y + 15:
            self.current_zone = "high"
        elif spawn_y + 15 <= vehicle_y < spawn_y + 25:
            self.current_zone = "mid"
        elif spawn_y + 25 <= vehicle_y < spawn_y + 35:
            self.current_zone = "low"

        # 返回对应速度和区域名称
        speed_limit = self.speed_map[self.current_zone]
        zone_name = {
            "high": "高速区(40km/h)",
            "mid": "中速区(25km/h)",
            "low": "低速区(10km/h)"
        }[self.current_zone]
        return speed_limit, zone_name

    def send_speed_command(self, vehicle_id, speed_limit, zone_type):
        command = {
            "vehicle_id": vehicle_id,
            "speed_limit_kmh": speed_limit,
            "zone_type": zone_type,
            "timestamp": time.time()
        }
        logger.info(f"\n📡 路侧V2X指令：{json.dumps(command, indent=2, ensure_ascii=False)}")
        return command


class VehicleUnit:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.vehicle.set_autopilot(False)
        self.control = carla.VehicleControl()
        self.control.steer = 0.0  # 强制直行
        self.control.hand_brake = False
        logger.info("✅ 车辆已设置为手动直行（精准控速）")

    def get_actual_speed(self):
        """获取车辆实际速度（km/h）"""
        velocity = self.vehicle.get_velocity()
        speed_kmh = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6
        return round(speed_kmh, 1)

    def precise_speed_control(self, target_speed):
        """核心修复：低速区加大油门，精准到10km/h"""
        actual_speed = self.get_actual_speed()

        # 1. 高速区：38-42km/h（精准控速）
        if target_speed == 40:
            if actual_speed > 42:
                self.control.throttle = 0.0
                self.control.brake = 0.4
            elif actual_speed < 38:
                self.control.throttle = 0.9
                self.control.brake = 0.0
            else:
                self.control.throttle = 0.2
                self.control.brake = 0.0

        # 2. 中速区：23-27km/h（精准控速）
        elif target_speed == 25:
            if actual_speed > 27:
                self.control.throttle = 0.0
                self.control.brake = 0.3
            elif actual_speed < 23:
                self.control.throttle = 0.6
                self.control.brake = 0.0
            else:
                self.control.throttle = 0.1
                self.control.brake = 0.0

        # 3. 低速区：9-11km/h（加大油门，确保到10km/h）
        elif target_speed == 10:
            if actual_speed > 11:
                self.control.throttle = 0.0
                self.control.brake = 0.2
            elif actual_speed < 9:
                self.control.throttle = 0.4  # 加大油门（原0.2→0.4）
                self.control.brake = 0.0
            else:
                self.control.throttle = 0.15  # 维持油门
                self.control.brake = 0.0

        self.vehicle.apply_control(self.control)
        return actual_speed

    def receive_speed_command(self, command):
        """接收并执行速度指令"""
        target_speed = command["speed_limit_kmh"]
        actual_speed = self.precise_speed_control(target_speed)
        logger.info(
            f"🚗 车载执行：目标{target_speed}km/h → 实际{actual_speed}km/h | 油门={round(self.control.throttle, 1)} 刹车={round(self.control.brake, 1)}")


# ===================== 3. 近距离视角 =====================
def set_near_observation_view(world, vehicle):
    """设置车辆后方近距离观察视角"""
    try:
        spectator = world.get_spectator()
        vehicle_transform = vehicle.get_transform()
        forward_vector = vehicle_transform.rotation.get_forward_vector()
        right_vector = vehicle_transform.rotation.get_right_vector()
        view_location = vehicle_transform.location - forward_vector * 8 + right_vector * 2 + carla.Location(z=2)
        view_rotation = carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw, roll=0)
        spectator.set_transform(carla.Transform(view_location, view_rotation))
        logger.info("✅ 初始视角已设置：车辆后方近距离")
        logger.info("📌 视角操作：鼠标拖拽=旋转 | 滚轮=缩放 | WASD=移动")
    except Exception as e:
        logger.warning(f"⚠️ 设置视角失败：{e}")


def get_valid_spawn_point(world):
    """获取有效生成点（容错处理）"""
    try:
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise Exception("无可用生成点")
        valid_spawn = spawn_points[10] if len(spawn_points) >= 10 else spawn_points[0]
        logger.info(f"✅ 车辆生成位置：(x={valid_spawn.location.x:.1f}, y={valid_spawn.location.y:.1f})")
        return valid_spawn
    except Exception as e:
        logger.error(f"❌ 获取生成点失败：{e}")
        raise


# ===================== 4. 辅助函数：获取CARLA启动指令（无绝对路径） =====================
def get_carla_launch_cmd():
    """获取CARLA启动指令（适配不同系统）"""
    if sys.platform == "win32":
        return "CarlaUE4.exe"  # Windows（需在CARLA根目录运行）
    elif sys.platform == "linux":
        return "./CarlaUE4.sh"  # Linux
    else:
        return "CarlaUE4"  # 其他系统


# ===================== 4. 主逻辑 =====================
def main():
    # 1. 连接CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world()
        logger.info(f"\n✅ 连接CARLA成功！服务器版本：{client.get_server_version()}")
    except Exception as e:
        logger.error(f"\n❌ 连接CARLA失败：{str(e)}")
        logger.info(f"📌 请先启动CARLA服务器：{get_carla_launch_cmd()}")
        sys.exit(1)

    # 2. 生成车辆
    vehicle = None
    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute('color', '255,0,0')  # 红色车身
        valid_spawn = get_valid_spawn_point(world)
        vehicle = world.spawn_actor(vehicle_bp, valid_spawn)
        logger.info(f"✅ 车辆生成成功，ID：{vehicle.id}（红色车身）")
    except Exception as e:
        logger.error(f"\n❌ 生成车辆失败：{str(e)}")
        sys.exit(1)

    # 3. 初始化V2X+视角
    try:
        rsu = RoadSideUnit(world, vehicle)
        vu = VehicleUnit(vehicle)
        set_near_observation_view(world, vehicle)

        # 4. 均衡测试（30秒，三区各10秒）
        logger.info("\n✅ 开始三区均衡变速测试（30秒）...")
        logger.info("📌 高速/中速/低速区各停留10秒，低速精准到10km/h！")
        start_time = time.time()

        # 设置同步模式（提高控速精度）
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        try:
            while time.time() - start_time < 30:
                speed_limit, zone_type = rsu.get_balance_speed_limit()
                command = rsu.send_speed_command(vehicle.id, speed_limit, zone_type)
                vu.receive_speed_command(command)
                world.tick()  # 同步物理帧
                time.sleep(0.1)  # 提高响应速度
        except KeyboardInterrupt:
            logger.info("\n⚠️  用户中断测试")
        finally:
            # 恢复异步模式
            settings.synchronous_mode = False
            world.apply_settings(settings)

    except Exception as e:
        logger.error(f"\n❌ 测试过程出错：{e}")
    finally:
        # 紧急停车+资源清理（容错处理）
        if vehicle:
            try:
                vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0, steer=0.0))
                time.sleep(2)
                vehicle.destroy()
                logger.info("\n✅ 测试结束，车辆已销毁")
            except Exception as e:
                logger.warning(f"⚠️  清理车辆失败：{e}")


if __name__ == "__main__":
    # 打印系统信息（便于调试）
    logger.info(f"🔍 当前Python解释器路径：{sys.executable}")
    logger.info(f"🔍 当前Python版本：{sys.version.split()[0]}")
    logger.info(f"🔍 操作系统：{sys.platform}")

    main()