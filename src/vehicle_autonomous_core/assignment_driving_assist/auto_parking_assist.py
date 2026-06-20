import carla
import numpy as np
import math

class AutoParkingAssist:
    def __init__(self, ego_car, lidar_sensor):
        self.ego = ego_car
        self.lidar = lidar_sensor
        self.park_target_dist = 2.2
        self.is_parking = False

    def lidar_callback(self, point_cloud):
        # 兼容旧版CARLA LidarMeasurement 正确取坐标
        raw = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        points = raw.reshape((-1, 4))  # x,y,z,intensity
        dist_list = []
        for p in points:
            x, y, z, _ = p
            # 筛选车身右侧区域点云
            if 1 < y < 4 and -2 < x < 2:
                dist = math.sqrt(x**2 + y**2)
                dist_list.append(dist)
        if not dist_list:
            return
        avg_dist = sum(dist_list) / len(dist_list)
        if abs(avg_dist - self.park_target_dist) < 0.6 and not self.is_parking:
            self.is_parking = True
            print("🅿️ 检测到空闲车位，开始自动泊车")
            self.park_backward()

    def park_backward(self):
        control = carla.VehicleControl()
        control.reverse = True
        control.throttle = 0.12  # 油门不能负数，负数无效
        control.steer = -0.32
        self.ego.apply_control(control)

if __name__ == "__main__":
    client = carla.Client("localhost",2000)
    client.set_timeout(8)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    # 生成主车
    car = world.spawn_actor(bp_lib.filter("model3")[0], world.get_map().get_spawn_points()[12])
    # 挂载侧方激光雷达
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_tf = carla.Transform(carla.Location(x=1,y=1,z=1))
    lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=car)

    park_helper = AutoParkingAssist(car, lidar)
    lidar.listen(park_helper.lidar_callback)

    try:
        while True:
            world.tick()
    except KeyboardInterrupt:
        lidar.destroy()
        car.destroy()