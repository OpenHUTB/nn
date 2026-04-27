import carla
import time
import math
import random

def run():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 清空所有车辆、行人、控制器
    for actor in world.get_actors().filter('*'):
        if (actor.type_id.startswith('vehicle') or
            actor.type_id.startswith('walker') or
            actor.type_id.startswith('controller')):
            actor.destroy()

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[12]

    # 生成主车
    ego_vehicle = world.spawn_actor(
        bp_lib.find('vehicle.tesla.model3'),
        spawn_point
    )
    ego_vehicle.set_autopilot(False)

    # --- 行人从路边冲向车道 ---
    forward = spawn_point.get_forward_vector()
    right = spawn_point.get_right_vector()

    # 行人生成点：路边
    pedestrian_start = spawn_point.location + forward * 22 + right * 6.5
    # 目标点：车道内
    pedestrian_goal = spawn_point.location + forward * 22 + right * 1.5

    # 生成行人
    walker_bp = bp_lib.find('walker.pedestrian.0001')
    pedestrian = world.spawn_actor(
        walker_bp,
        carla.Transform(pedestrian_start, spawn_point.rotation)
    )

    # 行人AI：向车道内走
    controller = world.spawn_actor(
        bp_lib.find('controller.ai.walker'),
        carla.Transform(),
        attach_to=pedestrian
    )
    controller.start()
    controller.go_to_location(pedestrian_goal)
    controller.set_max_speed(1.8)  # 正常步速冲向车道

    # 视角
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        spawn_point.location + carla.Location(x=-15, z=4),
        carla.Rotation(pitch=-15, yaw=spawn_point.rotation.yaw)
    ))

    print("✅ 场景1：行人主动闯入车道，车前方停、侧方近刹远减、后方正常")

    try:
        while True:
            control = carla.VehicleControl()
            ego_tf = ego_vehicle.get_transform()
            ego_loc = ego_tf.location
            vel = ego_vehicle.get_velocity()
            speed = math.hypot(vel.x, vel.y) * 3.6

            # 默认正常行驶
            control.throttle = 0.4 if speed < 25 else 0.1
            control.brake = 0.0
            control.steer = 0.0

            # --- 核心判断：前/侧/后，以及侧面距离 ---
            ped_loc = pedestrian.get_location()
            dx = ped_loc.x - ego_loc.x
            dy = ped_loc.y - ego_loc.y
            distance = math.sqrt(dx**2 + dy**2)

            forward_vec = ego_tf.get_forward_vector()
            # 前后判断（点积）
            dot_prod = dx * forward_vec.x + dy * forward_vec.y
            is_front = dot_prod > 1.0       # 在车头前方
            is_back = dot_prod < -1.0       # 在车尾后方

            # 侧面距离判断（横向距离）
            cross = abs(dx * forward_vec.y - dy * forward_vec.x)
            is_side = not is_front and not is_back and cross < 4.0 and distance < 20

            # --- 控制逻辑 ---
            if is_front and distance < 15:
                # 前方 + 距离近 → 完全停车礼让
                control.throttle = 0.0
                control.brake = 1.0
            elif is_side:
                if cross < 1.5:
                    # 侧面距离非常近 → 紧急刹车
                    control.throttle = 0.0
                    control.brake = 1.0
                else:
                    # 侧面但距离较远 → 减速让行
                    control.throttle = 0.15
                    control.brake = 0.2
            elif is_back:
                # 后方 → 完全不礼让，正常行驶
                pass
            else:
                # 其他情况（远、不在关键区域）→ 正常行驶
                pass

            ego_vehicle.apply_control(control)
            time.sleep(0.02)

    except KeyboardInterrupt:
        ego_vehicle.destroy()
        pedestrian.destroy()
        controller.stop()
        controller.destroy()
        print("\n✅ 已退出")

if __name__ == "__main__":
    run()
