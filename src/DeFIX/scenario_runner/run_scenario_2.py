import carla
import time
import math

def run():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 清空所有车辆
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

    # ==============================
    # 前方生成 4 辆拥堵车（排队堵车）
    # ==============================
    forward = spawn_point.get_forward_vector()
    obstacle_vehicles = []
    
    # 生成一排堵车车辆
    for i in range(4):
        dist = 15 + i * 8
        loc = spawn_point.location + forward * dist
        bp = bp_lib.filter('vehicle.*')[0]
        car = world.spawn_actor(bp, carla.Transform(loc, spawn_point.rotation))
        obstacle_vehicles.append(car)

    # 视角
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        spawn_point.location + carla.Location(x=-25, z=6),
        carla.Rotation(pitch=-25, yaw=spawn_point.rotation.yaw)
    ))

    print("✅ 场景2：前方多车拥堵排队 → 自动跟车刹车")

    try:
        while True:
            control = carla.VehicleControl()
            ego_tf = ego_vehicle.get_transform()
            ego_loc = ego_tf.location
            vel = ego_vehicle.get_velocity()
            speed = math.hypot(vel.x, vel.y) * 3.6

            # 默认行驶
            control.throttle = 0.4
            control.brake = 0.0
            control.steer = 0.0

            # ==============================
            # 检测前方最近车辆（拥堵刹车）
            # ==============================
            min_dist = 999
            forward_vec = ego_tf.get_forward_vector()

            for car in obstacle_vehicles:
                car_loc = car.get_location()
                dx = car_loc.x - ego_loc.x
                dy = car_loc.y - ego_loc.y
                dist = math.sqrt(dx**2 + dy**2)
                dot = dx * forward_vec.x + dy * forward_vec.y

                if dot > 0 and dist < min_dist:
                    min_dist = dist

            # 距离近 = 刹车停车
            if min_dist < 12:
                control.throttle = 0.0
                control.brake = 1.0
            elif min_dist < 20:
                # 距离中等 = 减速
                control.throttle = 0.15

            ego_vehicle.apply_control(control)
            time.sleep(0.02)

    except KeyboardInterrupt:
        ego_vehicle.destroy()
        for car in obstacle_vehicles:
            car.destroy()
        print("\n✅ 场景2已退出")

if __name__ == "__main__":
    run()
