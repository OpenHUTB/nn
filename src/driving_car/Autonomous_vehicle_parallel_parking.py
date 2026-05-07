import carla
import time
import numpy as np
import math
import pygame

# 初始化pygame用于键盘控制（可选）
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Carla Parking Control")

class CarlaParking:
    def __init__(self):
        # 连接Carla服务器
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        # 初始化车辆、传感器
        self.vehicle = None
        self.camera = None
        self.lidar = None
        self.spectator = self.world.get_spectator()

        # 车辆控制参数
        self.control = carla.VehicleControl()
        self.max_speed = 5.0  # 倒车入库最大速度（km/h）
        self.target_parking_transform = None  # 目标车位变换

        # 初始化场景
        self._init_scene()

    def _init_scene(self):
        """初始化场景：生成车辆、传感器、目标车位"""
        # 清除现有车辆和传感器
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        for actor in self.world.get_actors().filter('*sensor*'):
            actor.destroy()

        # 选择车辆蓝图（特斯拉Model3）
        vehicle_bp = self.blueprint_library.filter('model3')[0]
        vehicle_bp.set_attribute('color', '255,0,0')  # 红色车辆

        # 车辆初始位置（可根据地图调整，这里用Town03的道路位置）
        spawn_points = self.map.get_spawn_points()
        # 初始位置：离车位有一定距离的道路上
        start_transform = spawn_points[30]  # 可根据地图调整索引
        self.vehicle = self.world.spawn_actor(vehicle_bp, start_transform)
        print(f"车辆生成成功：{self.vehicle.id}")

        # 生成目标车位（这里手动设置车位位置，可替换为自动检测）
        # 车位位置：在初始位置的后方偏右（倒车入库的目标位置）
        parking_location = start_transform.location + carla.Location(x=-5.0, y=2.0, z=0.0)
        parking_rotation = start_transform.rotation + carla.Rotation(yaw=180.0)  # 车位朝向与初始方向相反
        self.target_parking_transform = carla.Transform(parking_location, parking_rotation)

        # 生成摄像头传感器（用于可视化）
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        # 摄像头安装在车辆顶部
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda image: self._process_camera_image(image))

        # 生成LIDAR传感器（可选，用于车位检测）
        # lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        # lidar_bp.set_attribute('range', '50')
        # lidar_transform = carla.Transform(carla.Location(z=2.0))
        # self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        # self.lidar.listen(lambda point_cloud: self._process_lidar(point_cloud))

    def _process_camera_image(self, image):
        """处理摄像头图像并显示"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # 去掉alpha通道
        array = array.swapaxes(0, 1)
        surface = pygame.surfarray.make_surface(array)
        screen.blit(surface, (0, 0))
        pygame.display.update()

    def _process_lidar(self, point_cloud):
        """处理LIDAR点云（车位检测用，这里仅占位）"""
        pass

    def get_vehicle_pose(self):
        """获取车辆当前位姿（位置+朝向）"""
        transform = self.vehicle.get_transform()
        x = transform.location.x
        y = transform.location.y
        yaw = math.radians(transform.rotation.yaw)
        return np.array([x, y, yaw])

    def calculate_parking_error(self):
        """计算车辆当前位姿与目标车位的误差"""
        current_pose = self.get_vehicle_pose()
        target_pose = np.array([
            self.target_parking_transform.location.x,
            self.target_parking_transform.location.y,
            math.radians(self.target_parking_transform.rotation.yaw)
        ])

        # 位置误差（x, y）
        pos_error = target_pose[:2] - current_pose[:2]
        # 朝向误差（yaw）
        yaw_error = target_pose[2] - current_pose[2]
        # 归一化朝向误差到[-pi, pi]
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

        return pos_error, yaw_error

    def parking_control(self):
        """倒车入库控制算法（纯追踪+PID控制）"""
        pos_error, yaw_error = self.calculate_parking_error()
        distance = np.linalg.norm(pos_error)  # 距离误差

        # 控制参数
        k_p_pos = 0.1  # 位置比例系数
        k_p_yaw = 0.5  # 朝向比例系数
        max_steer = math.radians(30)  # 最大转向角（30度）

        # 1. 朝向控制：优先调整朝向
        steer = k_p_yaw * yaw_error
        steer = np.clip(steer, -max_steer, max_steer)

        # 2. 速度控制：倒车（负速度），距离越近速度越慢
        speed = -min(self.max_speed * (distance / 5.0), self.max_speed)  # 距离5m时满速，近距离减速
        if distance < 0.2:  # 误差小于0.2m时停止
            speed = 0.0
            steer = 0.0
            print("倒车入库完成！")

        # 转换为Carla的控制指令
        self.control.steer = float(steer / max_steer)  # Carla转向范围[-1, 1]
        self.control.throttle = 0.0 if speed < 0 else 0.5  # 倒车时油门为0，前进时油门0.5
        self.control.brake = 0.0
        self.control.reverse = speed < 0  # 倒车标志

        # 应用控制
        self.vehicle.apply_control(self.control)

        return distance < 0.2  # 返回是否完成入库

    def run(self):
        """主循环"""
        try:
            finished = False
            while not finished:
                # 处理pygame事件（退出控制）
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        finished = True

                # 执行倒车入库控制
                finished = self.parking_control() or finished

                # 更新 spectator 视角（跟随车辆）
                vehicle_transform = self.vehicle.get_transform()
                spectator_transform = carla.Transform(
                    vehicle_transform.location + carla.Location(x=-10, z=5),
                    carla.Rotation(yaw=vehicle_transform.rotation.yaw, pitch=-15)
                )
                self.spectator.set_transform(spectator_transform)

                time.sleep(0.05)  # 控制循环频率（20Hz）

        finally:
            # 清理资源
            if self.vehicle:
                self.vehicle.destroy()
            if self.camera:
                self.camera.destroy()
            if self.lidar:
                self.lidar.destroy()
            pygame.quit()
            print("资源清理完成")

if __name__ == '__main__':
    parking = CarlaParking()
    parking.run()