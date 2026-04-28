import carla
import time
import math
import random

def run():
    # 连接 CARLA
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

    # 主车
    ego_vehicle = world.spawn_actor(
        bp_lib.find('vehicle.tesla.model3'),
        spawn_point
    )
    ego_vehicle.set_autopilot(False)

    # --- 关键修复：行人固定在路肩，强制冲向车道 ---
    forward = spawn_point.get_forward_vector()
    # 行人生成点：路肩，主车前方 30 米
    pedestrian_spawn_loc = spawn_point.location + forward * 30 + carla.Location(y=6.5)
    # 目标点：车道内
    pedestrian_goal_loc = spawn_point.location + forward * 30 + carla.Location(y=1.5)

    # 选择固定的行人蓝图，避免随机加载失败
    walker_bp = bp_lib.find('walker.pedestrian.0001')
    pedestrian = world.spawn_actor(walker_bp, carla.Transform(pedestrian_spawn_loc))

    # 生成并启动AI控制器
    walker_controller = world.spawn_actor(
        bp_lib.find('controller.ai.walker'),
        carla.Transform(),
        attach_to=pedestrian
    )
    walker_controller.start()
    walker_controller.go_to_location(pedestrian_goal_loc)
    walker_controller.set_max_speed(3.0)  # 快速冲入车道

    # 视角（拉近一点，方便看行人）
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        spawn_point.location + carla.Location(x=-15, z=3),
        carla.Rotation(pitch=-15, yaw=spawn_point.rotation.yaw)
    ))

    print("✅ 场景1启动：【突发行人碰撞】")

    try:
        while True:
            control = carla.VehicleControl()
            ego_tf = ego_vehicle.get_transform()
            ego_loc = ego_tf.location
            vel = ego_vehicle.get_velocity()
            speed = math.hypot(vel.x, vel.y) * 3.6

            # 基础行驶（降低初始速度，给刹车反应时间）
            if speed < 20:
                control.throttle = 0.4
            else:
                control.throttle = 0.1
            control.steer = 0.0

            # 安全刹车逻辑
            ped_loc = pedestrian.get_location()
            distance = math.sqrt(
                (ego_loc.x - ped_loc.x) ** 2 + 
                (ego_loc.y - ped_loc.y) ** 2
            )

            # 距离小于15米时开始刹车
            if distance < 15:
                control.throttle = 0.0
                control.brake = 1.0

            ego_vehicle.apply_control(control)
            time.sleep(0.02)

    except KeyboardInterrupt:
        ego_vehicle.destroy()
        walker_controller.stop()
        walker_controller.destroy()
        pedestrian.destroy()
        print("\n✅ 已退出")

if __name__ == "__main__":
    run()
