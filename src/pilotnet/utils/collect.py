# utils/collect.py
# 修改后的数据采集器：移除 pygame 显示，仅连接 CARLA 并记录驾驶数据

from utils.screen import clear, message, warn
from utils.piloterror import PilotError
import datetime
import os
import carla

class Collector:
    def __init__(self, world, time):
        self.start_time = datetime.datetime.now()
        self.world = world
        self.vehicle = None

        # 不再初始化 pygame 显示
        self.directory = f'recordings/{datetime.datetime.now().strftime("%Y-%m-%d@%H.%M.%S" if os.name == "nt" else "%Y-%m-%d@%H:%M:%S")}'
        self.start(time)

    def record(self, image):
        control = self.vehicle.get_control()
        # 保存图像到磁盘，文件名包含遥测数据
        image.save_to_disk(f'{self.directory}/{[int((datetime.datetime.now() - self.start_time).total_seconds()), control.steer, control.throttle, control.brake]}.png')
        # 移除所有 pygame 显示代码，不再转换和渲染图像

    def start(self, time):
        try:
            message('Spawning vehicle')
            vehicle_blueprints = self.world.get_blueprint_library().filter('*vehicle*')
            spawn_points = self.world.get_map().get_spawn_points()
            self.vehicle = self.world.spawn_actor(vehicle_blueprints[0], spawn_points[0])  # 使用第一个可用点，避免随机失败
            message('OK')
        except Exception as e:
            raise PilotError(f'Failed to spawn vehicle: {e}')

        try:
            message('Spawning camera and attaching to vehicle')
            camera_init_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
            camera_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_blueprint.set_attribute('image_size_x', '950')
            camera_blueprint.set_attribute('image_size_y', '500')
            camera_blueprint.set_attribute('fov', '110')
            message('OK')
        except Exception as e:
            raise PilotError(f'Failed to attach camera to vehicle: {e}')

        self.camera = self.world.spawn_actor(camera_blueprint, camera_init_trans, attach_to=self.vehicle)
        self.camera.listen(lambda image: self.record(image))
        self.vehicle.set_autopilot(True)

        try:
            elapsed = 0
            while elapsed <= time * 60:
                self.world.tick()
                if elapsed != int((datetime.datetime.now() - self.start_time).total_seconds()):
                    elapsed = int((datetime.datetime.now() - self.start_time).total_seconds())
                    clear()
                    message(f'Time elapsed: {int(elapsed / 60.0)}m {elapsed % 60}s')
            self.stop()
        except KeyboardInterrupt:
            self.stop()
            raise PilotError('You stopped the recording manually. Cleaning up and returning to main menu')

    def stop(self):
        message('Quitting recorder')
        try:
            self.camera.stop()
            self.vehicle.destroy()
        except:
            pass
        message("Vehicle destroyed")
        # 不再调用 pygame.display.quit()