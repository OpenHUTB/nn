from __future__ import annotations

import carla
import cv2
import numpy as np
import queue

from lane import Lane

class World():
    """Environment that wraps around the Carla-API
    """

    def __init__(self, server_ip:str='127.0.0.1', port:int=2000,
        timeout:float=5.0, map:str='Town04', image_height:int=480,
        image_width:int=640, fov:int=110,
        time_difference:float=0.01) -> None:
        """Constructor

        Args:
            server_ip (str, optional): IP address of the server running Carla.
                Defaults to '127.0.0.1'.
            port (int, optional): Port used to communicate with the server.
                Defaults to 2000.
            timeout (float, optional): Time difference after which the
                connection attempt with the server is aborted. Note that several
                attempts may be needed. Defaults to 5.0.
            map (str, optional): Map to be used. The available options depend on
                the Carla simulator used. Defaults to 'Town04'.
            image_height (int, optional): Height of the image to return.
                Defaults to 480.
            image_width (int, optional): Width of the image to return. Defaults
                to 640.
            fov (int, optional): Field of view of the camera. Defaults to 110.
            time_difference (float, optional): Difference of time simulated at
                each step. Defaults to 0.01.
        """

        self.image_height = image_height
        self.image_width = image_width

        self.lane = Lane(height=self.image_height, width=self.image_width)

        self.fov = fov
        self.client = carla.Client(server_ip, port)
        self.client.set_timeout(timeout)
        self.world = self.client.load_world(map)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = time_difference
        # Client and server work synchronously.
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.blueprint_library = self.world.get_blueprint_library()
        # Use Tesla Model 3 as Car
        self.bp = self.blueprint_library.filter('model3')[0]
        # Fixed spawnpoint.
        self.spawn_point = self.world.get_map().get_spawn_points()[0]

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
        self.vehicle = self.world.spawn_actor(self.bp, self.spawn_point)

        # Spawn camera
        blueprint = self.blueprint_library.find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', f'{self.image_width}')
        blueprint.set_attribute('image_size_y', f'{self.image_height}')
        blueprint.set_attribute('fov', f'{self.fov}')

        # Relative location to car
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(blueprint, spawn_point,
                                        attach_to=self.vehicle)
        self.sensor.listen(self.image_queue.put)

        # Spawn collision sensor
        blueprint = self.world.get_blueprint_library() \
            .find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(blueprint,
            carla.Transform(), attach_to=self.vehicle)
        self.collision_detected = False
        self.collision_sensor.listen(lambda e: {self._set_collision()})
        self.initialized = True

        return self._change_to_left_lane()

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
    def show_image(image:np.ndarray) -> None:
        """Display the Image

        Args:
            image (np.ndarray): Image to display.
        """
        cv2.imshow("", image)
        cv2.waitKey(1)

    def step(self, show:bool=True, throttle:float=0,
        steer:float=0) -> tuple[float, float, np.ndarray, bool]:
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