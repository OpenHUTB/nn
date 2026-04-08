
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q
from pygame import key
import numpy as np

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla

#*******************************************************************************
IM_WIDTH = 640
IM_HEIGHT = 480
IM_FOV = 100
IM_SHUTTER = 30
TIMEOUT = 20
CAM_LOC_X = 3
CAM_LOC_Z = 1.3


class sensor_platform():
    """Class to create a car with RGB sensors and pass the image to the visualizer"""
    def __init__(self):
        self.width = IM_WIDTH
        self.height = IM_HEIGHT
        self.fov = IM_FOV
        self.shutter = IM_SHUTTER
        self.sensor = None
        self.actor_list = []
        self.sensor_list = []
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(TIMEOUT)
        self.world = self.client.get_world()
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        
        # synchronous mode:
        settings = self.world.get_settings()
        print("Is client in synchrony mode? ",settings.synchronous_mode)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        bp = self.blueprint_library.filter("mustang")[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        # To solve spawn collision for the sensor platform and avoid execution crash
        for spawn_collision in range(1, 10):
            try:
                self.vehicle = self.world.spawn_actor(bp, spawn_point)
                self.vehicle.set_autopilot(True) #initiate with autopilot
                self.actor_list.append(self.vehicle)
            except:
                print(f'Sensor platform spawn failed. collision counter: {spawn_collision} ')
                spawn_point = random.choice(self.world.get_map().get_spawn_points())
                time.sleep(0.5)
                pass

    def __del__(self):
        for actor in self.actor_list:
            print(f'actor {actor}, destroyed')
            actor.destroy()
        for sensor in self.sensor_list:
            print(f'sensor {sensor}, destroyed')
            sensor.destroy()
        print("all actors and sensors destroyed")

    def set_sensor(self, bp_name = "sensor.camera.rgb"):
        cam_bp = self.blueprint_library.find(bp_name)
        cam_bp.set_attribute("image_size_x", f"{self.width}")
        cam_bp.set_attribute("image_size_y", f"{self.height}")
        cam_bp.set_attribute("fov", f"{self.fov}")
        cam_bp.set_attribute("shutter_speed", f"{self.shutter}")
        spawn_point = carla.Transform(carla.Location(x=CAM_LOC_X, z=CAM_LOC_Z))
        self.sensor = self.world.spawn_actor(cam_bp, spawn_point, attach_to=self.vehicle)
        self.sensor_list.append(self.sensor)
        return self.sensor


# to RGB avoiding alpha
    def carla_to_cv(self, image, visualize=False):
        i = np.array(image.raw_data) # one dimension array
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:,:,:3] #only get rgb avoids alpha 
        if visualize:
            cv2.imshow("RGB Sensor: Front Camera",i3)
            c = cv2.waitKey(1) # ASCII 'Esc' value
            if c == 27:
                print('Closing simulator camera, shutting down application...')
                cv2.destroyAllWindows()
                exit()
        return i3

def main():
    platform = sensor_platform()
    sensor = platform.set_sensor()
    sensor.listen(lambda data: platform.carla_to_cv(data))
    while 1:
        time.sleep(20)

if __name__ == "__main__":
    main()