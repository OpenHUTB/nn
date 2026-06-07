# run_scenario_0.py
import carla
import time
import math

# 把所有逻辑放进 run() 函数
def run():
    # 连接CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 清空旧车
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()

    bp_lib = world.get_blueprint_library()
    spawn_point = world.get_map().get_spawn_points()[12]

    # 主车（左车道）
    ego_vehicle = world.spawn_actor(bp_lib.find('vehicle.tesla.model3'), spawn_point)
    ego_vehicle.set_autopilot(False)

    # 障碍车（右车道，前方65米）
    forward = spawn_point.get_forward_vector()
    obs_loc = spawn_point.location + forward * 65 + carla.Location(y=3.2)
    obs_tf = carla.Transform(obs_loc, spawn_point.rotation)
    obs_vehicle = world.spawn_actor(bp_lib.find('vehicle.audi.a2'), obs_tf)

    # 视角
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        spawn_point.location + carla.Location(x=-18, z=5),
        carla.Rotation(pitch=-22, yaw=spawn_point.rotation.yaw)
    ))

    print(" 平滑超车版启动：无猛打方向，绝不左转")

    is_overtaking = False
    overtake_t = 0.0

    try:
        while True:
            control = carla.VehicleControl()
            now = time.time()

            ego_tf = ego_vehicle.get_transform()
            vel = ego_vehicle.get_velocity()
            speed = math.hypot(vel.x, vel.y) * 3.6
            dist = math.hypot(ego_tf.location.x - obs_tf.location.x, ego_tf.location.y - obs_tf.location.y)

            # ----------------------
            # 核心：全程微角度，无猛打方向，自然回中
            # ----------------------
            if not is_overtaking:
                control.steer = 0.0
                control.throttle = 0.45 if speed < 25 else 0.2
                control.brake = 0.0

                if dist < 55:
                    is_overtaking = True
                    overtake_t = now

            else:
                dt = now - overtake_t

                # 阶段1：先加速，不打方向
                if dt < 0.6:
                    control.steer = 0.0
                    control.throttle = 0.7
                # 阶段2：微幅左转，只打一点点
                elif dt < 1.5:
                    control.steer = -0.02
                    control.throttle = 0.7
                # 阶段3：保持方向，继续超车
                elif dt < 2.2:
                    control.steer = -0.01
                    control.throttle = 0.65
                # 阶段4：缓慢回正，不猛拉
                elif dt < 3.0:
                    control.steer = 0.015
                    control.throttle = 0.6
                # 阶段5：完全回中，重置状态
                else:
                    control.steer = 0.0
                    control.throttle = 0.4
                    if dt > 3.5:
                        is_overtaking = False

            # 安全兜底：距离过近才刹车
            if dist < 8:
                control.throttle = 0.0
                control.brake = 1.0

            ego_vehicle.apply_control(control)
            time.sleep(0.02)

    except KeyboardInterrupt:
        ego_vehicle.destroy()
        obs_vehicle.destroy()
        print("\n 已退出")

# 只有直接运行这个文件才会执行
if __name__ == "__main__":
    run()
