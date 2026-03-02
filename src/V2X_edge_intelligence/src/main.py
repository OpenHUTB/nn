# main.py（CARLA V2X低速区专属测试 - 唯一入口+无绝对路径）
import sys
import os
import time
import json
import math

# ===================== 1. 自动适配CARLA路径（无绝对路径） =====================
def setup_carla_path():
    """自动配置CARLA路径（优先级：环境变量 > 相对路径 > 提示用户）"""
    # 优先级1：读取环境变量 CARLA_PYTHON_API_PATH
    carla_api_path = os.environ.get("CARLA_PYTHON_API_PATH")
    if carla_api_path and os.path.exists(carla_api_path):
        egg_files = [f for f in os.listdir(carla_api_path) if f.endswith(".egg")]
        if egg_files:
            carla_egg_path = os.path.join(carla_api_path, egg_files[0])
            print(f"🔍 从环境变量加载CARLA egg：{carla_egg_path}")
            sys.path.insert(0, carla_egg_path)
            return True

    # 优先级2：自动查找常见的相对路径
    common_paths = [
        "./PythonAPI/carla/dist",
        "../WindowsNoEditor/PythonAPI/carla/dist",
        "./WindowsNoEditor/PythonAPI/carla/dist"
    ]
    for path in common_paths:
        if os.path.exists(path):
            egg_files = [f for f in os.listdir(path) if f.endswith(".egg")]
            if egg_files:
                carla_egg_path = os.path.join(path, egg_files[0])
                print(f"🔍 自动找到CARLA egg：{carla_egg_path}")
                sys.path.insert(0, carla_egg_path)
                return True

    # 优先级3：提示用户手动输入路径
    print("\n⚠️  未自动找到CARLA PythonAPI路径！")
    print("📌 请先设置环境变量 CARLA_PYTHON_API_PATH，例如：")
    print("   Windows: set CARLA_PYTHON_API_PATH=D:\\WindowsNoEditor\\PythonAPI\\carla\\dist")
    print("   Linux/Mac: export CARLA_PYTHON_API_PATH=/path/to/Carla/PythonAPI/carla/dist")
    manual_path = input("\n请输入CARLA egg文件所在目录（留空退出）：").strip()
    if manual_path and os.path.exists(manual_path):
        egg_files = [f for f in os.listdir(manual_path) if f.endswith(".egg")]
        if egg_files:
            carla_egg_path = os.path.join(manual_path, egg_files[0])
            sys.path.insert(0, carla_egg_path)
            print(f"✅ 手动加载CARLA egg：{carla_egg_path}")
            return True

    return False

# 初始化CARLA路径
print(f"🔍 当前Python解释器路径：{sys.executable}")
print(f"🔍 当前Python版本：{sys.version.split()[0]}")

if not setup_carla_path():
    print("\n❌ 无法找到CARLA egg文件，请检查路径配置！")
    sys.exit(1)

# 导入CARLA
try:
    import carla
    print("✅ CARLA模块导入成功！")
except Exception as e:
    print(f"\n❌ CARLA导入失败：{str(e)}")
    sys.exit(1)

# ===================== 2. 核心逻辑：仅保留低速区（10km/h） =====================
class RoadSideUnit:
    def __init__(self, carla_world, vehicle):
        self.world = carla_world
        self.vehicle = vehicle
        # 仅保留低速区：基于车辆生成位置设置低速区坐标
        spawn_loc = vehicle.get_location()
        self.low_speed_zone = carla.Location(spawn_loc.x, spawn_loc.y + 15, spawn_loc.z)
        self.zone_radius = 50  # 扩大低速区范围，确保全程在低速区
        self.speed_map = {"low": 10}  # 仅保留低速

    def get_speed_limit(self):
        """仅返回低速区的速度和类型（全程低速）"""
        return self.speed_map["low"], "低速区(10km/h)"

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
        print("✅ 车辆已设置为手动直行（低速区精准控速）")

    def get_actual_speed(self):
        """获取实际车速（km/h）"""
        velocity = self.vehicle.get_velocity()
        speed_kmh = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6
        return round(speed_kmh, 1)

    def force_stable_low_speed(self):
        """仅保留低速区控速逻辑：精准控制在8-12km/h"""
        target_speed = 10
        actual_speed = self.get_actual_speed()

        # 低速区精准控速逻辑
        if actual_speed > 12:
            self.control.throttle = 0.0
            self.control.brake = 0.5  # 适度刹车降速
        elif actual_speed < 8:
            self.control.throttle = 0.3  # 足够的油门确保到10km/h
            self.control.brake = 0.0
        else:
            self.control.throttle = 0.1  # 小油门维持速度
            self.control.brake = 0.1

        self.vehicle.apply_control(self.control)
        return actual_speed, target_speed

    def receive_speed_command(self, command):
        actual_speed, target_speed = self.force_stable_low_speed()
        print(
            f"🚗 车载执行：目标{target_speed}km/h → 实际{actual_speed}km/h | 油门={round(self.control.throttle, 1)} 刹车={round(self.control.brake, 1)}")

# ===================== 3. 近距离视角配置 =====================
def set_near_observation_view(world, vehicle):
    """设置车辆后方近距离观察视角"""
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
    """获取道路有效生成点"""
    spawn_points = world.get_map().get_spawn_points()
    valid_spawn = spawn_points[10] if len(spawn_points) >= 10 else spawn_points[5]
    print(f"✅ 车辆生成位置：(x={valid_spawn.location.x:.1f}, y={valid_spawn.location.y:.1f})")
    return valid_spawn

# ===================== 4. 主入口逻辑（仅低速区测试） =====================
def main():
    # 1. 连接CARLA服务器
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world()
        print(f"\n✅ 连接CARLA成功！服务器版本：{client.get_server_version()}")
    except Exception as e:
        print(f"\n❌ CARLA服务器连接失败：{str(e)}")
        print("📌 请先启动CARLA服务器（CarlaUE4.exe / CarlaUE4.sh）")
        sys.exit(1)

    # 2. 生成测试车辆
    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute('color', '255,0,0')  # 红色车身
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

    # 4. 启动低速区专属测试
    print("\n✅ 开始V2X低速区稳定测试（30秒）...")
    print("📌 速度严格控制在10km/h左右，全程低速运行！")
    start_time = time.time()
    try:
        while time.time() - start_time < 30:
            speed_limit, zone_type = rsu.get_speed_limit()
            command = rsu.send_speed_command(vehicle.id, speed_limit, zone_type)
            vu.receive_speed_command(command)
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\n⚠️  用户手动中断测试")
    finally:
        # 安全销毁车辆
        vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0, steer=0.0))
        time.sleep(2)
        vehicle.destroy()
        print("\n✅ 测试结束，车辆已销毁")

# 唯一入口
if __name__ == "__main__":
    main()