import carla
import numpy as np
import gym
from queue import Queue
import time
import sys

class CarlaEnvironment(gym.Env):
    def __init__(self):
        super(CarlaEnvironment, self).__init__()
        self.client = None
        self.world = None
        self.blueprint_library = None
        self._connect_carla()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
        )

        self.vehicle = None
        self.camera = None
        self.image_queue = Queue(maxsize=1)
        self.spawn_points = self.world.get_map().get_spawn_points()
        print(f"[CARLA场景] 检测到 {len(self.spawn_points)} 个车辆生成点")
        sys.stdout.flush()

    def _connect_carla(self):
        retry_count = 3
        for i in range(retry_count):
            try:
                print(f"[CARLA连接] 尝试第{i+1}次连接（localhost:2000）...")
                sys.stdout.flush()
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(15.0)
                self.world = self.client.get_world()
                self.blueprint_library = self.world.get_blueprint_library()
                print("[CARLA连接] 成功连接到模拟器")
                sys.stdout.flush()
                return
            except Exception as e:
                print(f"[CARLA连接失败] 第{i+1}次尝试：{str(e)}")
                sys.stdout.flush()
                if i == retry_count - 1:
                    raise RuntimeError("超过最大重试次数，无法连接CARLA，请检查模拟器是否启动")
                time.sleep(2)

    def process_image(self, image):
        """修复负步长和数组不可写问题"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # 移除alpha通道
            array = array[:, :, ::-1]  # BGR转RGB
            array = array.copy()  # 消除负步长
            if self.image_queue.full():
                self.image_queue.get()
            self.image_queue.put(array)
        except Exception as e:
            print(f"[图像处理错误] {str(e)}")
            sys.stdout.flush()

    def reset(self):
        self.close()
        time.sleep(0.5)
        self._spawn_vehicle()
        if self.vehicle:
            self._spawn_camera()
        time.sleep(1.0)  # 等待传感器就绪
        return self.get_observation()

    def _spawn_vehicle(self):
        # 选择稳定车型（特斯拉Model3）
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('color', '255,0,0')  # 红色，便于观察
        vehicle_bp.set_attribute('role_name', 'drone')

        # 关键调整：使用第10个生成点（通常在主路中央，避免障碍物）
        spawn_index = 10  # 可根据场景调整（0~264）
        for i in range(3):
            # 优先用指定生成点，失败则重试
            spawn_point = self.spawn_points[(spawn_index + i) % len(self.spawn_points)]
            print(f"[车辆生成] 尝试在生成点 {spawn_index + i} 生成车辆（主路中央）...")
            sys.stdout.flush()
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle:
                self.vehicle.set_autopilot(False)
                self.vehicle.set_simulate_physics(True)  # 强制启用物理引擎
                print(f"[车辆生成] 成功（ID: {self.vehicle.id}）- 位置：主路中央")
                sys.stdout.flush()
                return
            time.sleep(0.5)

        raise RuntimeError("所有生成点尝试失败，无法生成车辆（请重启CARLA或更换场景）")

    def _spawn_camera(self):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '128')
        camera_bp.set_attribute('image_size_y', '128')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.05')
        # 摄像头位置：车辆前方1.5米，高度2.4米（驾驶员视角）
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera.listen(self.process_image)
        print("[传感器] 摄像头初始化成功")
        sys.stdout.flush()

    def get_observation(self):
        if not self.image_queue.empty():
            return self.image_queue.get()
        print("[观测数据] 暂无图像，返回空帧")
        sys.stdout.flush()
        return np.zeros((128, 128, 3), dtype=np.uint8)

    def close(self):
        if self.camera and self.camera.is_alive:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None
            print("[资源清理] 摄像头已销毁")
            sys.stdout.flush()
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
            self.vehicle = None
            print("[资源清理] 车辆已销毁")
            sys.stdout.flush()
        while not self.image_queue.empty():
            self.image_queue.get()