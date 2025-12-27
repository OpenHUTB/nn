# main.py（CARLA V2X三区均衡变速测试 - 唯一入口+无绝对路径）
import sys
import os
import time
import json
import math

# ===================== 1. 自动适配CARLA路径（无绝对路径） =====================
def setup_carla_path():
    """
    自动配置CARLA PythonAPI路径（优先级：环境变量 > 相对路径 > 手动输入）
    彻底移除硬编码绝对路径，适配不同环境/安装位置
    """
    # 优先级1：读取系统环境变量（推荐长期使用）
    carla_api_env = os.environ.get("CARLA_PYTHON_API_PATH")
    if carla_api_env and os.path.exists(carla_api_env):
        egg_files = [f for f in os.listdir(carla_api_env) if f.endswith(".egg")]
        if egg_files:
            carla_egg = os.path.join(carla_api_env, egg_files[0])
            print(f"🔍 从环境变量加载CARLA egg：{carla_egg}")
            sys.path.insert(0, carla_egg)
            return True

    # 优先级2：自动查找常见相对路径（适配多数用户目录结构）
    common_relative_paths = [
        "./PythonAPI/carla/dist",          # 当前目录下的CARLA API
        "../WindowsNoEditor/PythonAPI/carla/dist",  # 上级目录的CARLA
        "./WindowsNoEditor/PythonAPI/carla/dist"    # 当前目录的CARLA
    ]
    for path in common_relative_paths:
        if os.path.exists(path):
            egg_files = [f for f in os.listdir(path) if f.endswith(".egg")]
            if egg_files:
                carla_egg = os.path.join(path, egg_files[0])
                print(f"🔍 自动找到CARLA egg：{carla_egg}")
                sys.path.insert(0, carla_egg)
                return True

    # 优先级3：提示用户手动输入（兜底方案）
    print("\n⚠️  未自动识别CARLA PythonAPI路径！")
    print("📌 请先配置环境变量（推荐）：")
    print("   Windows: set CARLA_PYTHON_API_PATH=你的CARLA路径\\PythonAPI\\carla\\dist")
    print("   Linux/Mac: export CARLA_PYTHON_API_PATH=你的CARLA路径/PythonAPI/carla/dist")
    manual_path = input("\n请输入CARLA egg文件所在目录（留空退出）：").strip()
    if manual_path and os.path.exists(manual_path):
        egg_files = [f for f in os.listdir(manual_path) if f.endswith(".egg")]
        if egg_files:
            carla_egg = os.path.join(manual_path, egg_files[0])
            sys.path.insert(0, carla_egg)
            print(f"✅ 手动加载CARLA egg：{carla_egg}")
            return True

    return False

# 初始化CARLA路径（无绝对路径）
print(f"🔍 当前Python解释器路径：{sys.executable}")
print(f"🔍 当前Python版本：{sys.version.split()[0]}")

if not setup_carla_path():
    print("\n❌ 无法加载CARLA PythonAPI，请检查路径配置！")
    sys.exit(1)

# 导入CARLA（适配0.9.10+版本）
try:
    import carla
    print("✅ CARLA模块导入成功！")
except Exception as e:
    print(f"\n❌ CARLA导入失败：{str(e)}")
    sys.exit(1)

# ===================== 2. 核心逻辑：三区均衡分配+低速精准控速 =====================
class RoadSideUnit:
    def __init__(self, carla_world, vehicle):
        self.world = carla_world
        self.vehicle = vehicle
        # 三区等距坐标（基于车辆生成位置动态计算，无绝对坐标）
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

        # 三区计时逻辑（确保每区停留10秒）
        self.current_zone = "high"  # 初始区：高速
        self.zone_start_time = time.time()
        self.zone_duration = 10  # 每区停留10秒（30秒测试周期）
        self.speed_map = {"high": 40, "mid": 25, "low": 10}

    def get_balance_speed_limit(self):
        """计时+位置双重判断，确保三区平均分配"""
        current_time = time.time()
        vehicle_loc = self.vehicle.get_location()
        vehicle_y = vehicle_loc.y
        spawn_y = self.vehicle.get_location().y

        # 1. 计时强制切换：每区停留10秒必切换
        if current_time - self.zone_start_time > self.zone_duration:
            zone_switch = {"high": "mid", "mid": "low", "low": "high"}
            self.current_zone = zone_switch[self.current_zone]
            self.zone_start_time = current_time

        # 2. 位置验证：确保区域与物理位置匹配
        if spawn_y + 5 <= vehicle_y < spawn_y + 15:
            self.current_zone = "high"
        elif spawn_y + 15 <= vehicle_y < spawn_y + 25:
            self.current_zone = "mid"
        elif spawn_y + 25 <= vehicle_y < spawn_y + 35:
            self.current_zone = "low"

        # 返回速度和区域名称
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
        print(f"\n📡 路侧V2X指令：{json.dumps(command, indent=2, ensure_ascii=False)}")
        return command

class VehicleUnit:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.vehicle.set_autopilot(False)
        self.control = carla.VehicleControl()
        self.control.steer = 0.0  # 强制直行
        self.control.hand_brake = False
        print("✅ 车辆已设置为手动直行（三区精准控速）")

    def get_actual_speed(self):
        """计算车辆实际速度（km/h）"""
        velocity = self.vehicle.get_velocity()
        speed_kmh = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6
        return round(speed_kmh, 1)

    def precise_speed_control(self, target_speed):
        """三区精准控速，低速区加大油门确保到10km/h"""
        actual_speed = self.get_actual_speed()

        # 高速区：38-42km/h
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

        # 中速区：23-27km/h
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

        # 低速区：9-11km/h（0.4油门确保速度达标）
        elif target_speed == 10:
            if actual_speed > 11:
                self.control.throttle = 0.0
                self.control.brake = 0.2
            elif actual_speed < 9:
                self.control.throttle = 0.4  # 加大油门确保到10km/h
                self.control.brake = 0.0
            else:
                self.control.throttle = 0.15
                self.control.brake = 0.0

        self.vehicle.apply_control(self.control)
        return actual_speed

    def receive_speed_command(self, command):
        target_speed = command["speed_limit_kmh"]
        actual_speed = self.precise_speed_control(target_speed)
        print(
            f"🚗 车载执行：目标{target_speed}km/h → 实际{actual_speed}km/h | 油门={round(self.control.throttle, 1)} 刹车={round(self.control.brake, 1)}")

# ===================== 3. 近距离视角配置 =====================
def set_near_observation_view(world, vehicle):
    """设置车辆后方近距离视角（无绝对坐标）"""
    spectator = world.get_spectator()
    vehicle_transform = vehicle.get_transform()
    forward_vector = vehicle_transform.rotation.get_forward_vector()
    right_vector = vehicle_transform.rotation.get_right_vector()
    view_location = vehicle_transform.location - forward_vector * 8 + right_vector * 2 + carla.Location(z=2)
    view_rotation = carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw, roll=0)
    spectator.set_transform(carla.Transform(view_location, view_rotation))
    print("✅ 初始视角已设置：车辆后方近距离")
    print("📌 视角操作：鼠标拖拽=旋转 | 滚轮=缩放 | WASD=移动")

def get_valid_spawn_point(world):
    """获取道路有效生成点（无绝对坐标）"""
    spawn_points = world.get_map().get_spawn_points()
    valid_spawn = spawn_points[10] if len(spawn_points) >= 10 else spawn_points[5]
    print(f"✅ 车辆生成位置：(x={valid_spawn.location.x:.1f}, y={valid_spawn.location.y:.1f})")
    return valid_spawn

# ===================== 4. 主入口逻辑（唯一入口） =====================
def main():
    # 1. 连接CARLA服务器（通用提示，无绝对路径）
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world()
        print(f"\n✅ 连接CARLA成功！服务器版本：{client.get_server_version()}")
    except Exception as e:
        print(f"\n❌ CARLA服务器连接失败：{str(e)}")
        print("📌 请先启动CARLA服务器（通用路径参考）：")
        print("   Windows: ./WindowsNoEditor/CarlaUE4.exe")
        print("   Linux/Mac: ./CarlaUE4.sh")
        sys.exit(1)

    # 2. 生成测试车辆（红色车身）
    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute('color', '255,0,0')
        valid_spawn = get_valid_spawn_point(world)
        vehicle = world.spawn_actor(vehicle_bp, valid_spawn)
        print(f"✅ 车辆生成成功，ID：{vehicle.id}（红色车身）")
    except Exception as e:
        print(f"\n❌ 车辆生成失败：{str(e)}")
        sys.exit(1)

    # 3. 初始化V2X组件+设置视角
    rsu = RoadSideUnit(world, vehicle)
    vu = VehicleUnit(vehicle)
    set_near_observation_view(world, vehicle)

    # 4. 启动三区均衡测试
    print("\n✅ 开始V2X三区均衡变速测试（30秒）...")
    print("📌 高速/中速/低速区各停留10秒，低速精准到10km/h！")
    start_time = time.time()
    try:
        while time.time() - start_time < 30:
            speed_limit, zone_type = rsu.get_balance_speed_limit()
            command = rsu.send_speed_command(vehicle.id, speed_limit, zone_type)
            vu.receive_speed_command(command)
            time.sleep(1)  # 1秒高频更新，响应更快
    except KeyboardInterrupt:
        print("\n⚠️  用户手动中断测试")
    finally:
        # 安全停车并销毁车辆
        vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0, steer=0.0))
        time.sleep(2)
        vehicle.destroy()
        print("\n✅ 测试结束，车辆已销毁")

# 唯一入口（确保仅main.py作为脚本运行）
if __name__ == "__main__":
    main()