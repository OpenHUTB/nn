#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Original script from https://github.com/carla-simulator

Script that render multiple sensors in the same pygame window.

Oriented towards creating visual explanations for deep learning models.

By default, it renders four cameras, one LiDAR and one Semantic LiDAR.

So far only RGB camera feed is analyzed due to the CAM algorithm compatibilities.

@author: Pablo Roca - github.com/RocaPiedra

"""
import glob
import os
import sys
import traceback
import math
# 将自定义模块路径添加到系统路径，以便导入
sys.path.append('./visualizer')
sys.path.append('./carlacomms')

import roc_functions
import parameters
from gui_CAM import gui_CAM

try:
    # 查找并添加 CARLA 的 Python API 库 (.egg 文件)
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import random
import time
import numpy as np
import pygame

# 自定义计时器类，用于计算传感器处理延迟
class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()
# 界面显示管理器：负责 Pygame 窗口的初始化、多网格布局及渲染
class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        # 初始化显示窗口，使用硬件加速和双缓冲
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        try:
            roc_functions.blip_logo(self.display, parameters.carla_logo)
        except Exception as e:
            print(f'Logo blip failed with exception:\n{e}')
        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]
    # 计算单个网格单元的大小
    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]
    # 根据网格坐标计算在窗口中的像素偏移量
    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list
    # 循环渲染所有传感器的画面并刷新屏幕
    def render(self):
        if not self.render_enabled():
            return
        for s in self.sensor_list:
            s.render()
        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None
    
    def get_display(self):
        return self.display
    # 根据点击位置选择主传感器（仅限 RGB 相机）
    def select_main_sensor(self, location):
        for s in self.sensor_list:
            if s.is_clicked(location):
                if s.sensor_type == 'RGBCamera':
                    print(f'the sensor selected is {s.sensor_type}')
                    return s
                else:
                    print(f'the sensor selected is {s.sensor_type}, please select a compatible sensor [RGBCamera]')
# 传感器管理器：负责传感器的创建、数据监听、预处理及渲染准备
class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos):
        self.surface = None   # 用于存储准备渲染到 Pygame 的表面
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos 
        self.display_offset = display_man.get_display_offset(display_pos) 
        self.display_size = display_man.get_display_size() 
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()
        self.sensor_type = sensor_type

        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)

    # 初始化不同类型的传感器（RGB、LiDAR、语义LiDAR、雷达）
    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)
            return camera

        elif sensor_type == 'LiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)
            lidar.listen(self.save_lidar_image)
            return lidar
        
        elif sensor_type == 'SemanticLiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', '100')

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)
            lidar.listen(self.save_semanticlidar_image)
            return lidar
        
        elif sensor_type == "Radar":
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
            radar.listen(self.save_radar_image)
            return radar
        else:
            return None

    def get_sensor(self):
        return self.sensor
    # 处理 RGB 图像数据并转换为 Pygame Surface 
    def save_rgb_image(self, image, return_image = False):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
        
        if return_image:
            return image
    # 处理 LiDAR 点云并投影为 2D 俯视图图像
    def save_lidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data) 
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        # 在黑色背景上绘制白色点
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
    # 处理语义 LiDAR 数据
    def save_semanticlidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data) 
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
    # 处理雷达数据
    def save_radar_image(self, radar_data):
        t_start = self.timer.time()
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
    # 将生成的 Surface 绘制到窗口对应位置
    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()

    def return_surface(self):
        if self.surface is not None:
            return self.surface
    # 检查鼠标点击是否在该传感器的显示区域内
    def is_clicked(self, location):
        if ((self.display_offset[0]<location[0]<self.display_offset[0]+self.display_size[0]) and
            (self.display_offset[1]<location[1]<self.display_offset[1]+self.display_size[1])):
            return True


def run_carla_CAM(args, client):
    display_manager = None
    vehicle = None
    vehicle_list = []
    timer = CustomTimer()
    class_menu = None
    
    if args.gpu:
        use_cuda = args.gpu
    
    try:
        world = client.get_world()
        original_settings = world.get_settings()

        if args.sync:
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

        bp = world.get_blueprint_library().filter('charger_2020')[0]
        vehicle = world.spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
        vehicle_list.append(vehicle)
        vehicle.set_autopilot(True)

        # --- 动态计算所需的网格尺寸和传感器数量 ---
        total_sensors = 1 # 至少有前置主摄
        if args.side_cams: total_sensors += 3
        if args.lidar: total_sensors += 1
        if args.semantic_lidar: total_sensors += 1

        if total_sensors == 1:
            grid_size = [1, 1]
        elif total_sensors == 2:
            grid_size = [1, 2]
        elif total_sensors <= 4:
            grid_size = [2, 2]
        else:
            grid_size = [2, 3]

        display_manager = DisplayManager(grid_size=grid_size, window_size=[args.width, args.height])

        # 获取下一个可用的网格坐标的辅助函数
        def get_grid_pos(index):
            return [index // grid_size[1], index % grid_size[1]]
        
        current_idx = 0

        # 前置主摄像机 (始终开启)
        main_sensor = SensorManager(world, display_manager, 'RGBCamera', 
                                    carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)), 
                                    vehicle, {}, display_pos=get_grid_pos(current_idx))
        current_idx += 1

        # 侧边和后置摄像机开关
        if args.side_cams:
            SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=-90)), vehicle, {}, display_pos=get_grid_pos(current_idx))
            current_idx += 1
            SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+90)), vehicle, {}, display_pos=get_grid_pos(current_idx))
            current_idx += 1
            SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=180)), vehicle, {}, display_pos=get_grid_pos(current_idx))
            current_idx += 1

        # LiDAR开关
        if args.lidar:
            SensorManager(world, display_manager, 'LiDAR', carla.Transform(carla.Location(x=0, z=2.4)), 
                          vehicle, {'channels' : '64', 'range' : '100',  'points_per_second': '250000', 'rotation_frequency': '20'}, display_pos=get_grid_pos(current_idx))
            current_idx += 1

        # 语义LiDAR开关
        if args.semantic_lidar:
            SensorManager(world, display_manager, 'SemanticLiDAR', carla.Transform(carla.Location(x=0, z=2.4)), 
                          vehicle, {'channels' : '64', 'range' : '100', 'points_per_second': '100000', 'rotation_frequency': '20'}, display_pos=get_grid_pos(current_idx))
            current_idx += 1

        class_menu = gui_CAM(display_manager.display, use_cuda, display_manager)
        weather_manager = roc_functions.WeatherManager(world)
        weather_manager.list_weather_presets()
        
        call_exit = False
        parameters.paused = False
        
        # 将 CAM 分析框绑定到第一个位置 (0,0)
        cam_offset = display_manager.get_display_offset([0,0])
        
        display_manager.display.fill((0,0,0))
        pygame.display.update() 
        
        while True:
            if args.sync and not parameters.paused:
                world.tick()
                display_manager.render()
            elif args.sync and parameters.paused:
                client.set_timeout(100.0)
            else:
                world.wait_for_tick()
                display_manager.render()

            for event in pygame.event.get():
                input_surface = main_sensor.return_surface()
                call_exit = class_menu.run_menu_no_loop(event, call_exit, input_surface, cam_offset)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_presses = pygame.mouse.get_pressed()
                    if mouse_presses[0]:
                        selected_sensor = display_manager.select_main_sensor(location = pygame.mouse.get_pos())
                        if selected_sensor:
                            main_sensor = selected_sensor
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        weather_manager.next_weather()
                        
            if call_exit:
                break
                
    except Exception as e:
        print(f'Is the simulation running?')
        print(e)
        
    finally:
        if display_manager:
            display_manager.destroy()
        if class_menu:
            del class_menu
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
        if world:
            world.apply_settings(original_settings)

def main():
    argparser = argparse.ArgumentParser(description='CARLA CAM Visualizer')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--sync', action='store_true', help='Synchronous mode execution')
    argparser.add_argument('--async', dest='sync', action='store_false', help='Asynchronous mode execution')
    argparser.set_defaults(sync=True)
    
    # 分辨率设回了 800x600，保证轻量化
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='800x600', help='window resolution (default: 800x600)')
    argparser.add_argument('--keepsim', dest='keepsim', action='store_true', help='Maintain simulation execution in the background')
    argparser.set_defaults(keepsim=False)
    argparser.add_argument('--gpu', dest='gpu', action='store_true', help='GPU acceleration mode execution')
    argparser.set_defaults(gpu=True)

    # --- 新增的模块开关 ---
    argparser.add_argument('--lidar', action='store_true', help='Enable standard LiDAR sensor')
    argparser.add_argument('--semantic-lidar', action='store_true', help='Enable Semantic LiDAR sensor')
    argparser.add_argument('--side-cams', action='store_true', help='Enable side and rear RGB cameras')

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        run_carla_CAM(args, client)

    except (RuntimeError, TypeError, NameError):
        _, generate_traffic = roc_functions.launch_carla_simulator_locally()
        client = carla.Client(args.host, args.port)
        client.set_timeout(15.0)
        run_carla_CAM(args, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        
    finally:
        print('terminating traffic...')
        try:
            generate_traffic.terminate()
        except:
            pass
        if not args.keepsim:
            roc_functions.close_carla_simulator()
            sys.exit(0)
        else:
            sys.exit(0)

if __name__ == '__main__':
    main()
