import carla
import time
import math

# 连接CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# 清空旧车辆
for v in world.get_actors().filter('vehicle.*'):
    v.destroy()

# --------------------------
# 场景0：动态车辆碰撞（车道内障碍车）
# --------------------------
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
spawn_point = spawn_points[86]

# 主车：关闭自动驾驶，脚本直接控制
ego_vehicle = world.spawn_actor(
    bp_lib.find('vehicle.tesla.model3'),
    spawn_point
)
ego_vehicle.set_autopilot(False)

# 关键：障碍车放在主车【同一条车道】的正前方
obs_forward = spawn_point.get_forward_vector()
obs_location = spawn_point.location + obs_forward * 25  # 距离主车25米，同车道
obs_trans = carla.Transform(obs_location, spawn_point.rotation)  # 和主车同方向
obs_vehicle = world.spawn_actor(
    bp_lib.find('vehicle.audi.a2'),
    obs_trans
)

# 开局一次性车后视角，后续可自由移动
spectator = world.get_spectator()
init_view = carla.Transform(
    spawn_point.location + carla.Location(x=-15, z=4),
    carla.Rotation(pitch=-18, yaw=spawn_point.rotation.yaw)
)
spectator.set_transform(init_view)

print("。。。 场景0已启动 | 障碍车已放在主车同车道正前方25米处")
print("。。。 距离控制防碰撞系统已开启")

# --------------------------
# 分级速度控制：正常行驶+自动避让
# --------------------------
try:
    while True:
        # 计算两车距离
        ego_loc = ego_vehicle.get_transform().location
        obs_loc = obs_vehicle.get_transform().location
        distance = math.sqrt((ego_loc.x-obs_loc.x)**2 + (ego_loc.y-obs_loc.y)**2)

        # 获取当前车速（km/h）
        v = ego_vehicle.get_velocity()
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6

        control = carla.VehicleControl()

        # 分级控制逻辑
        if distance < 10:
            # 距离过近：全力刹停
            control.throttle = 0.0
            control.brake = 1.0
        elif distance < 20:
            # 距离中等：减速（限制在10km/h以内）
            control.throttle = 0.1
            control.brake = 0.6
        else:
            # 距离远：正常加速（最高25km/h）
            if speed < 25:
                control.throttle = 0.3
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = 0.2

        ego_vehicle.apply_control(control)
        time.sleep(0.02)

except KeyboardInterrupt:
    ego_vehicle.destroy()
    obs_vehicle.destroy()
    print("\n程序已退出")
