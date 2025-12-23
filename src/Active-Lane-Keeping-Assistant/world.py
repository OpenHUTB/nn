from __future__ import annotations

import carla
import cv2
import numpy as np
import queue
import random  # <--- 新增：为了随机选择出生点

from lane import Lane


class World():
    """Environment that wraps around the Carla-API
    """

    def __init__(self, server_ip: str = '127.0.0.1', port: int = 2000,
                 timeout: float = 5.0, map: str = 'Town04', image_height: int = 480,
                 image_width: int = 640, fov: int = 110,
                 time_difference: float = 0.01) -> None:
        """Constructor
        """

        self.image_height = image_height
        self.image_width = image_width

        self.lane = Lane(height=self.image_height, width=self.image_width)

        self.fov = fov

        # 1. 连接客户端
        self.client = carla.Client(server_ip, port)

        # === 修复 1: 设置 30 秒超时，防止加载地图太慢报错 ===
        self.client.set_timeout(30.0)
        # ===============================================

        # 2. 加载地图
        self.world = self.client.load_world(map)

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = time_difference
        # Client and server work synchronously.
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.blueprint_library = self.world.get_blueprint_library()

        # Use Tesla Model 3 as Car
        self.bp = self.blueprint_library.filter('model3')[0]

        # === 修改: 将出生点设置为当前上帝视角相机的位置 ===
        # 1. 获取上帝视角的观察者对象
        spectator = self.world.get_spectator()

        # 2. 获取观察者现在的坐标和朝向
        transform = spectator.get_transform()

        # 3.稍微把位置抬高一点（+2米），防止相机贴地导致车卡在地里
        transform.location.z += 2.0

        # 4. 稍微把位置往前挪一点（+5米），让你能直接看到车出现在眼前
        # (利用朝向向量计算前方的位置)
        forward_vector = transform.get_forward_vector()
        transform.location.x += forward_vector.x * 5.0
        transform.location.y += forward_vector.y * 5.0

        self.spawn_point = transform
        # ====================================================

        self.vehicle = None
        self.sensor = None
        self.collision_sensor = None

        self.image_queue = queue.Queue()
        self.collision_detected = False
        self.initialized = False

    def close(self) -> None:
        """Destroys all currently used Actors
        """
        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.sensor is not None:
            self.sensor.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()

    def reset(self) -> tuple[float, float, np.ndarray, bool]:
        """Resets the Actors

        Returns:
            tuple[float, float, np.ndarray, bool]:
                [0]: Difference to the center of the detected lane.
                [1]: Detected surface area.
                [2]: Image consisting including the detected surface area.
                [3]: Whether a collision has been detected.
        """
        self.image_queue = queue.Queue()
        self.close()

        # 尝试生成车辆，如果失败（比如还是有碰撞），这行可能会报错，但有了随机点后概率大大降低
        self.vehicle = self.world.spawn_actor(self.bp, self.spawn_point)

        # Spawn camera
        self._spawn_camera()

        # Spawn collision sensor
        blueprint = self.world.get_blueprint_library() \
            .find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(blueprint,
                                                       carla.Transform(), attach_to=self.vehicle)
        self.collision_detected = False
        # Remove redundant set brackets
        self.collision_sensor.listen(lambda e: self._set_collision())
        self.initialized = True

        return 0.0, 0.0, np.zeros((self.image_height, self.image_width, 3)), False

    def _spawn_camera(self) -> None:
        """Helper method to spawn the camera sensor"""
        blueprint = self.blueprint_library.find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', f'{self.image_width}')
        blueprint.set_attribute('image_size_y', f'{self.image_height}')
        blueprint.set_attribute('fov', f'{self.fov}')

        # Relative location to car
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(blueprint, spawn_point,
                                             attach_to=self.vehicle)
        self.sensor.listen(self.image_queue.put)
    def _set_collision(self) -> None:
        """Helper to set collision_detected to True.
        """
        self.collision_detected = True

    def _change_to_left_lane(self) -> tuple[float, float, np.ndarray, bool]:
        """Hard Code to make the Car start in the leftmost Lane

        Returns:
            tuple[float, float, np.ndarray, bool]:
                [0]: Difference to the center of the detected lane.
                [1]: Detected surface area.
                [2]: Image consisting including the detected surface area.
                [3]: Whether a collision has been detected.
        """
        for throttle, steer, steps in [
            (0.2, -0.11, 850), (0.2, 0.17, 250), (-0.2, 0.17, 50)]:
            for _ in range(steps):
                error, detection_surface_area, transformed_image, _ = self.step(
                    throttle=throttle, steer=steer)

        return error, detection_surface_area, transformed_image, \
            self.collision_detected

    def get_image(self) -> np.ndarray:
        """Retrieve the Image in RGB

        Returns:
            np.ndarray: Image in RGB.
        """
        image = self.image_queue.get()
        image = np.array(image.raw_data)
        image = image.reshape((self.image_height, self.image_width, 4))
        image = image[:, :, :3]
        return image

    @staticmethod
    def show_image(image: np.ndarray) -> None:
        """Display the Image

        Args:
            image (np.ndarray): Image to display.
        """
        cv2.imshow("", image)
        cv2.waitKey(1)

    def step(self, show: bool = True, throttle: float = 0,
             steer: float = 0) -> tuple[float, float, np.ndarray, bool]:
        """Simulate one Step

        Args:
            show (bool, optional): Whether to show the image of the car driving.
                Defaults to True.
            throttle (float, optional): Which throttle to apply. Defaults to 0.
            steer (float, optional): Which steering to apply. Defaults to 0.

        Raises:
            Exception: Requires the reset method to be called before step.

        Returns:
            tuple[float, float, np.ndarray, bool]:
                [0]: Difference to the center of the detected lane.
                [1]: Detected surface area.
                [2]: Image consisting including the detected surface area.
                [3]: Whether a collision has been detected.
        """

        if not self.initialized:
            raise Exception('Reset must be called before step.')

        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle,
                                                        steer=steer))
        self.world.tick()
        image = self.get_image()
        transformed_image, error, detection_surface_area = self.lane \
            .pipe(img=image)

        if show:
            World.show_image(image=transformed_image)

        return error, detection_surface_area, transformed_image, \
            self.collision_detected