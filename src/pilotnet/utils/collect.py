# The data collector. It connects to the carla server and records driving data

from utils.screen import clear, message, warn
from utils.piloterror import PilotError
import numpy as np
import carla, datetime, pygame, os

class Collector:
    def __init__(self, world, time):
        self.start_time = datetime.datetime.now()
        self.world = world
        self.vehicle = None
        try:
            pygame.init()
        except: pass
        try:
            self.display = pygame.display.set_mode((950, 500))
        except:
            warn("Failed to spawn live feed view for data collector. If you're on WSL, this happens as the OS doesn't have a display device yet. Otherwise, check your pygame installation.")
            pass
        self.directory = f'recordings/{datetime.datetime.now().strftime("%Y-%m-%d@%H.%M.%S" if os.name is "nt" else "%Y-%m-%d@%H:%M:%S" )}'
        self.start(time)
    
    def record(self, image):
        control = self.vehicle.get_control()        
        image.save_to_disk(f'{self.directory}/{[int((datetime.datetime.now() - self.start_time).total_seconds()), control.steer, control.throttle, control.brake]}.png')
        
        # we now convert image into a raw image to show in our display
        image.convert(carla.ColorConverter.Raw)
        
        # convert the image into an array using standard procedure
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        
        # show the frame in display & update it
        try:
            surf = pygame.surfarray.make_surface(np.rot90(array, 1))
            self.display.blit(surf, (0, 0))
            pygame.display.update()
        except:
            warn('Display is not working')

    def start(self, time):

        try:
            # get a list of spawn points & vehicles then randomly choose from one to use
            message('Spawning vehicle')
            vehicle_blueprints = self.world.get_blueprint_library().filter('*vehicle*')
            spawn_points = self.world.get_map().get_spawn_points()
            self.vehicle = self.world.spawn_actor(np.random.choice(vehicle_blueprints), np.random.choice(spawn_points))
            message('OK')
        except:
            raise PilotError('Failed to spawn vehicle. Check start() in utils/collect.py for more info')

        try:
            # make a camera and configure it
            message('Spawning camera and attaching to vehicle')
            camera_init_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
            camera_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_blueprint.set_attribute('image_size_x', '950')
            camera_blueprint.set_attribute('image_size_y', '500')
            camera_blueprint.set_attribute('fov', '110') # sets field of view (FOV)
            message('OK')
        except:
            raise PilotError('Failed to attach camera to vehicle. Check start() in utils/collect.py for more info')

        # attach camera to vehicle and start recording
        self.camera = self.world.spawn_actor(camera_blueprint, camera_init_trans, attach_to=self.vehicle)
        self.camera.listen(lambda image: self.record(image))

        # autopilot obviously
        self.vehicle.set_autopilot(True)

        try:
            elapsed = 0
            # in Carla, we have to call tick() or wait_for_tick() after altering anything in order to reflect change
            while elapsed <= time*60:
                self.world.tick()
                if elapsed != int((datetime.datetime.now() - self.start_time).total_seconds()):
                    elapsed = int((datetime.datetime.now() - self.start_time).total_seconds())
                    clear()
                    message(f'Time elapsed: {int(((datetime.datetime.now() - self.start_time).total_seconds())/60.0)}m {int((datetime.datetime.now() - self.start_time).total_seconds())}s')
            self.stop()
        except KeyboardInterrupt:
            self.stop()
            raise PilotError('You stopped the recording manually. Cleaning up and returning to main menu')
    
    def stop(self):
        message('Quitting recorder')
        try:
            self.camera.stop() # destroy sensor in main smiulation (server)
            self.vehicle.destroy()
        except:
            pass
        message("Vehicle destroyed")
        try:
            pygame.display.quit()
        except: pass