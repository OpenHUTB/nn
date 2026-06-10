# utils/collect.py
# 修改后的数据采集器：移除 pygame 显示，仅连接 CARLA 并记录驾驶数据

from utils.screen import clear, message, warn
from utils.piloterror import PilotError
from utils.visualizer import CarlaVisualizer
from utils.logger import logger
import datetime
import os
import carla
import numpy as np

class Collector:
    def __init__(self, world, time, enable_visualization=True):
        self.start_time = datetime.datetime.now()
        self.world = world
        self.vehicle = None
        self.enable_visualization = enable_visualization
        self.visualizer = None

        self.directory = f'recordings/{datetime.datetime.now().strftime("%Y-%m-%d@%H.%M.%S" if os.name == "nt" else "%Y-%m-%d@%H:%M:%S")}'
        logger.info(f'Created data collection directory: {self.directory}')
        self.start(time)

    def record(self, image):
        control = self.vehicle.get_control()
        # 保存图像到磁盘，文件名包含遥测数据
        image.save_to_disk(f'{self.directory}/{[int((datetime.datetime.now() - self.start_time).total_seconds()), control.steer, control.throttle, control.brake]}.png')
        # 移除所有 pygame 显示代码，不再转换和渲染图像

    def update_spectator(self):
        '更新 spectator 相机位置，实现镜头跟随'
        if not self.vehicle:
            return
        
        # 获取车辆位置和旋转
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation
        
        # 计算 spectator 相机位置（在车辆后方和上方）
        # 保持与车辆相同的旋转，但位置在车辆后方
        camera_location = carla.Location(
            x=vehicle_location.x - 10.0 * vehicle_rotation.get_forward_vector().x,
            y=vehicle_location.y - 10.0 * vehicle_rotation.get_forward_vector().y,
            z=vehicle_location.z + 5.0
        )
        
        # 计算相机朝向（看向车辆）
        camera_rotation = vehicle_rotation
        
        # 更新 spectator 相机位置
        spectator_transform = carla.Transform(camera_location, camera_rotation)
        self.spectator.set_transform(spectator_transform)

    def start(self, time):
        try:
            message('Spawning vehicle')
            vehicle_blueprints = self.world.get_blueprint_library().filter('*vehicle*')
            spawn_points = self.world.get_map().get_spawn_points()
            self.vehicle = self.world.spawn_actor(vehicle_blueprints[0], spawn_points[0])
            message('OK')
            logger.info('Vehicle spawned successfully')
        except Exception as e:
            logger.error(f'Failed to spawn vehicle: {e}')
            raise PilotError(f'Failed to spawn vehicle: {e}')

        # 获取 spectator 相机用于镜头跟随
        self.spectator = self.world.get_spectator()
        logger.info('Spectator camera initialized for follow mode')

        if self.enable_visualization:
            self.visualizer = CarlaVisualizer(self.world, self.vehicle)
            message('Visualization enabled')
            logger.info('Visualization system initialized')

        try:
            message('Spawning camera and attaching to vehicle')
            camera_init_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
            camera_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_blueprint.set_attribute('image_size_x', '950')
            camera_blueprint.set_attribute('image_size_y', '500')
            camera_blueprint.set_attribute('fov', '110')
            message('OK')
            logger.info('Camera blueprint configured: 950x500 resolution, 110 FOV')
        except Exception as e:
            logger.error(f'Failed to configure camera: {e}')
            raise PilotError(f'Failed to attach camera to vehicle: {e}')

        self.camera = self.world.spawn_actor(camera_blueprint, camera_init_trans, attach_to=self.vehicle)
        self.camera.listen(lambda image: self.record(image))
        self.vehicle.set_autopilot(True)
        logger.info('Camera attached and autopilot enabled')

        try:
            elapsed = 0
            while elapsed <= time * 60:
                self.world.tick()
                
                # 更新 spectator 相机位置，实现镜头跟随
                self.update_spectator()
                
                if self.enable_visualization and self.visualizer:
                    camera_location = self.vehicle.get_transform().transform(camera_init_trans.location)
                    self.visualizer.draw_all(camera_location)
                
                if elapsed != int((datetime.datetime.now() - self.start_time).total_seconds()):
                    elapsed = int((datetime.datetime.now() - self.start_time).total_seconds())
                    clear()
                    message(f'Time elapsed: {int(elapsed / 60.0)}m {elapsed % 60}s')
                    
                    if self.enable_visualization and self.visualizer:
                        # 获取实时控制数据
                        control = self.vehicle.get_control()
                        velocity = self.vehicle.get_velocity()
                        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                        transform = self.vehicle.get_transform()
                        
                        # 显示实时数据
                        message(f'\n实时数据:')
                        message(f'当前速度: {speed:.1f} km/h')
                        message(f'转向角度: {control.steer:.3f}')
                        message(f'油门压力: {control.throttle:.3f}')
                        message(f'刹车压力: {control.brake:.3f}')
                        message(f'车辆位置: X={transform.location.x:.1f}, Y={transform.location.y:.1f}, Z={transform.location.z:.1f}')
                        message(f'车辆朝向: Yaw={transform.rotation.yaw:.1f}°')
                        
                        # 显示统计数据
                        stats = self.visualizer.get_statistics()
                        if stats:
                            message(f'\n统计数据:')
                            message(f'平均速度: {stats["avg_speed"]:.1f} km/h (最高: {stats["max_speed"]:.1f} km/h)')
                            message(f'平均控制 - 转向: {stats["avg_steer"]:.3f}, 油门: {stats["avg_throttle"]:.3f}, 刹车: {stats["avg_brake"]:.3f}')
                            message(f'录制帧数: {stats["total_frames"]}')
                            message(f'轨迹点数量: {len(self.visualizer.trajectory_points)}')
                            
            self.stop()
        except KeyboardInterrupt:
            self.stop()
            raise PilotError('You stopped the recording manually. Cleaning up and returning to main menu')

    def stop(self):
        message('Quitting recorder')
        logger.info('Stopping data collection')
        try:
            self.camera.stop()
            self.vehicle.destroy()
            logger.info('Camera stopped and vehicle destroyed')
        except Exception as e:
            logger.warning(f'Error during cleanup: {e}')
        message("Vehicle destroyed")
        
        total_time = int((datetime.datetime.now() - self.start_time).total_seconds())
        logger.info(f'Data collection completed - total time: {total_time} seconds')
        
        if self.enable_visualization and self.visualizer:
            stats = self.visualizer.get_statistics()
            if stats:
                message(f'\n=== 录制统计报告 ===')
                message(f'总录制时间: {total_time} 秒')
                message(f'平均速度: {stats["avg_speed"]:.1f} km/h')
                message(f'最高速度: {stats["max_speed"]:.1f} km/h')
                message(f'平均转向角度: {stats["avg_steer"]:.3f}')
                message(f'平均油门压力: {stats["avg_throttle"]:.3f}')
                message(f'平均刹车压力: {stats["avg_brake"]:.3f}')
                message(f'总录制帧数: {stats["total_frames"]}')
                message(f'轨迹点数量: {len(self.visualizer.trajectory_points)}')
                message(f'数据保存目录: {self.directory}')
                
                logger.info(f'Recording statistics - avg_speed: {stats["avg_speed"]:.1f} km/h, max_speed: {stats["max_speed"]:.1f} km/h, total_frames: {stats["total_frames"]}')