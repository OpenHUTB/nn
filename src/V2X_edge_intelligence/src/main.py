# v2x_balance_zones.py（三区平均分配+低速精准控速）
import sys
import os
import time
import json
import math

# ===================== 1. 配置CARLA路径 =====================
CARLA_EGG_PATH = r"D:\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg"

print(f"🔍 当前Python解释器路径：{sys.executable}")
print(f"🔍 当前Python版本：{sys.version.split()[0]}")
print(f"🔍 CARLA egg路径：{CARLA_EGG_PATH}")

if not os.path.exists(CARLA_EGG_PATH):
    raise FileNotFoundError(f"\n❌ CARLA egg文件不存在：{CARLA_EGG_PATH}")
if CARLA_EGG_PATH not in sys.path:
    sys.path.insert(0, CARLA_EGG_PATH)

try:
    import carla

    print("✅ CARLA模块导入成功！")
except Exception as e:
    print(f"\n❌ 导入失败：{str(e)}")
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
        print(f"\n📡 路侧V2X指令：{json.dumps(command, indent=2, ensure_ascii=False)}")
        return command


class VehicleUnit:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.vehicle.set_autopilot(False)
        self.control = carla.VehicleControl()
        self.control.steer = 0.0  # 强制直行
        self.control.hand_brake = False
        print("✅ 车辆已设置为手动直行（精准控速）")

    def get_actual_speed(self):
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
        target_speed = command["speed_limit_kmh"]
        actual_speed = self.precise_speed_control(target_speed)
        print(
            f"🚗 车载执行：目标{target_speed}km/h → 实际{actual_speed}km/h | 油门={round(self.control.throttle, 1)} 刹车={round(self.control.brake, 1)}")


# ===================== 3. 近距离视角 =====================
def set_near_observation_view(world, vehicle):
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
    spawn_points = world.get_map().get_spawn_points()
    valid_spawn = spawn_points[10] if len(spawn_points) >= 10 else spawn_points[5]
    print(f"✅ 车辆生成位置：(x={valid_spawn.location.x:.1f}, y={valid_spawn.location.y:.1f})")
    return valid_spawn


# ===================== 4. 主逻辑 =====================
def main():
    # 1. 连接CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world()
        print(f"\n✅ 连接CARLA成功！服务器版本：{client.get_server_version()}")
    except Exception as e:
        print(f"\n❌ 连接失败：{str(e)}")
        print("📌 请先启动：D:\WindowsNoEditor\CarlaUE4.exe")
        sys.exit(1)

    # 2. 生成车辆
    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute('color', '255,0,0')  # 红色车身
        valid_spawn = get_valid_spawn_point(world)
        vehicle = world.spawn_actor(vehicle_bp, valid_spawn)
        print(f"✅ 车辆生成成功，ID：{vehicle.id}（红色车身）")
    except Exception as e:
        print(f"\n❌ 生成车辆失败：{str(e)}")
        sys.exit(1)

    # 3. 初始化V2X+视角
    rsu = RoadSideUnit(world, vehicle)
    vu = VehicleUnit(vehicle)
    set_near_observation_view(world, vehicle)

    # 4. 均衡测试（30秒，三区各10秒）
    print("\n✅ 开始三区均衡变速测试（30秒）...")
    print("📌 高速/中速/低速区各停留10秒，低速精准到10km/h！")
    start_time = time.time()
    try:
        while time.time() - start_time < 30:
            speed_limit, zone_type = rsu.get_balance_speed_limit()
            command = rsu.send_speed_command(vehicle.id, speed_limit, zone_type)
            vu.receive_speed_command(command)
            time.sleep(1)  # 1秒更新，响应更快
    except KeyboardInterrupt:
        print("\n⚠️  用户中断测试")
    finally:
        # 紧急停车
        vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0, steer=0.0))
        time.sleep(2)
        vehicle.destroy()
        print("\n✅ 测试结束，车辆已销毁")


if __name__ == "__main__":
    main()