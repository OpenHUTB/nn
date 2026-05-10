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

    print("✅ 场景5：穿越无信号灯路口（自动减速）")

    # 找到最近的路口中心
    ego_start = spawn_point.location
    junctions = []
    for sp in spawn_points:
        if abs(sp.location.x - ego_start.x) < 50 and abs(sp.location.y - ego_start.y) < 50:
            junctions.append(sp.location)
    if not junctions:
        junctions.append(ego_start + carla.Location(x=30, y=0))
    junction_center = junctions[0]

    try:
        while True:
            control = carla.VehicleControl()
            vel = ego_vehicle.get_velocity()
            speed = math.hypot(vel.x, vel.y) * 3.6
            ego_loc = ego_vehicle.get_location()

            # 计算到路口中心的距离
            dx = junction_center.x - ego_loc.x
            dy = junction_center.y - ego_loc.y
            dist_to_junction = math.sqrt(dx**2 + dy**2)

            # 路口减速逻辑：距离路口20米内自动减速
            if dist_to_junction < 20:
                control.throttle = 0.1
                control.brake = 0.3
            else:
                control.throttle = 0.4 if speed < 30 else 0.2
                control.brake = 0.0

            control.steer = 0.0
            ego_vehicle.apply_control(control)
            time.sleep(0.02)

    except KeyboardInterrupt:
        ego_vehicle.destroy()
        print("\n✅ 场景5已退出")

if __name__ == "__main__":
    run()
