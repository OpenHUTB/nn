import carla
import time
import math

def run():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 清空所有车辆、行人
    for actor in world.get_actors().filter('*'):
        if actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker') or actor.type_id.startswith('controller'):
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

    # 视角
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        spawn_point.location + carla.Location(x=-20, z=5),
        carla.Rotation(pitch=-22, yaw=spawn_point.rotation.yaw)
    ))

    print("✅ 场景3：车辆闯红灯")

    try:
        while True:
            control = carla.VehicleControl()
            vel = ego_vehicle.get_velocity()
            speed = math.hypot(vel.x, vel.y) * 3.6

            # 强制前进，无视红绿灯
            if speed < 30:
                control.throttle = 0.5
            else:
                control.throttle = 0.2
            control.brake = 0.0
            control.steer = 0.0

            ego_vehicle.apply_control(control)
            time.sleep(0.02)

    except KeyboardInterrupt:
        ego_vehicle.destroy()
        print("\n✅ 场景3已退出")

if __name__ == "__main__":
    run()
