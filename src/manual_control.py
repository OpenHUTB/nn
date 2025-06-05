#!/usr/bin/env python
# 参考：https://github.com/marcgpuig/carla_py_clients/blob/master/imu_plot.py

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

# 如果要自己操控，新开一个命令行，进入WindowsNoEditor\PythonAPI\examples目录，在cmd中输入：
# python manual_control.py

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    O            : open/close all doors of vehicle
    T            : toggle vehicle's telemetry

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

# 获取当前进程ID（便于调试）
process_id = os.getpid()
print("Current process id is: ", process_id)

try:
    # 动态添加 CARLA Python API 的路径
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    # 如果找不到 CARLA egg 文件，则不执行任何操作
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla # CARLA 核心库

from carla import ColorConverter as cc # CARLA 颜色转换工具


import argparse # 用于解析命令行参数
import collections # 提供额外的数据结构，如 defaultdict
import datetime # 用于处理日期和时间
import logging # 用于日志记录
import math # 数学运算库
import random # 用于生成随机数
import re # 正则表达式库
import weakref # 用于创建弱引用，避免循环引用

try:
    import pygame # Pygame 库，用于创建窗口、处理输入和图形显示
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed') # Pygame 未安装则抛出异常

try:
    import numpy as np # Numpy 库，用于数值计算，特别是数组操作
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed') # Numpy 未安装则抛出异常


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets(): # 查找可用的天气预设
    # 定义一个正则表达式，用于将 PascalCase 格式的字符串拆分成单词
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    # 定义一个 lambda 函数，将类名如 "ClearNoon" 拆分为 "Clear Noon"
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    # 从 carla.WeatherParameters 中提取所有以大写字母开头的属性名 (这些是天气预设)
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    # 返回包含 (天气参数对象, 格式化后的名称) 的元组列表
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250): # 获取 Actor 的显示名称
    # 提取 actor 的类型标识符，并将其格式化为更易读的名称（例如 vehicle.tesla.model3 -> Tesla Model3）
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    # 如果名称过长，则进行截断，并在末尾加上省略号（…）
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def get_actor_blueprints(world, filter, generation): # 根据筛选条件和代数获取 Actor 蓝图
    bps = world.get_blueprint_library().filter(filter) # 根据 filter 筛选蓝图

    if generation.lower() == "all": # 如果指定 "all"，则返回所有筛选到的蓝图
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    # 如果筛选结果只有一个蓝图，则直接返回该蓝图，忽略代数参数
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation) # 将代数参数转换为整数
        # Check if generation is in available generations
        # 检查代数是否在可用范围内 (1, 2, 3)
        if int_generation in [1, 2, 3]:
            # 筛选出指定代数的蓝图
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            # 如果代数无效，则打印警告并返回空列表
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        # 如果代数转换失败，则打印警告并返回空列表
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


#  ==============================================================================
#  -- World ---------------------------------------------------------------------
#  ==============================================================================


class World(object): # 定义 World 类，管理仿真世界中的所有元素和逻辑
    def __init__(self, carla_world, hud, args): # World 类的构造函数
        self.world = carla_world # CARLA 世界对象
        self.sync = args.sync # 是否启用同步模式
        self.actor_role_name = args.rolename # 主角 Actor 的角色名
        try:
            self.map = self.world.get_map() # 获取当前地图
        except RuntimeError as error:
            # 如果无法获取地图 (例如 .xodr 文件问题)，则打印错误并退出
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud # HUD (Heads-Up Display) 对象，用于显示信息
        self.player = None # 玩家 Actor (车辆或行人)
        self.collision_sensor = None # 碰撞传感器
        self.lane_invasion_sensor = None # 车道入侵传感器
        self.gnss_sensor = None # GNSS (全球导航卫星系统) 传感器
        self.imu_sensor = None # IMU (惯性测量单元) 传感器
        self.radar_sensor = None # 雷达传感器
        self.camera_manager = None # 摄像头管理器
        self._weather_presets = find_weather_presets() # 获取所有天气预设
        self._weather_index = 0 # 当前天气预设的索引
        self._actor_filter = args.filter # Actor 筛选条件 (例如 "vehicle.*")
        self._actor_generation = args.generation # Actor 代数
        self._gamma = args.gamma # 摄像头的 Gamma 校正值
        self.restart() # 初始化或重置玩家和相关传感器
        self.world.on_tick(hud.on_world_tick) # 注册 HUD 的 on_world_tick 方法，在每个仿真 tick 时调用
        self.recording_enabled = False # 录制功能是否启用
        self.recording_start = 0 # 录制开始时间 (用于回放)
        self.constant_velocity_enabled = False # 恒定速度模式是否启用
        self.show_vehicle_telemetry = False # 是否显示车辆遥测数据
        self.doors_are_open = False # 车辆门是否打开
        self.current_map_layer = 0 # 当前选择的地图图层索引
        self.map_layer_names = [ # 可用的地图图层列表
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def restart(self): # 重置玩家角色和相关设置
        self.player_max_speed = 1.589 # 玩家默认最大速度 (m/s)
        self.player_max_speed_fast = 3.713 # 玩家快速行走时的最大速度 (m/s)
        # Keep same camera config if the camera manager exists.
        # 如果摄像头管理器已存在，则保留当前的摄像头配置
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        # 获取符合条件的随机蓝图
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("Couldn't find any blueprints with the specified filters") # 如果找不到蓝图，则抛出异常
        blueprint = random.choice(blueprint_list) # 从列表中随机选择一个蓝图
        blueprint.set_attribute('role_name', self.actor_role_name) # 设置蓝图的角色名
        if blueprint.has_attribute('terramechanics'): # 如果蓝图有地形力学属性，则启用它
            blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'): # 如果蓝图有颜色属性，则随机选择一个推荐颜色
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'): # 如果蓝图有驾驶员ID属性，则随机选择一个推荐ID
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'): # 如果蓝图有无敌属性，则设置为true
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        # 如果蓝图有速度属性，则设置玩家的最大速度
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        # Spawn the player.
        # 生成玩家 Actor
        if self.player is not None: # 如果玩家已存在，则销毁旧玩家并在其位置上方生成新玩家
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0 # 在原位置上方2米处生成，避免碰撞
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy() # 销毁当前玩家及传感器
            self.player = self.world.try_spawn_actor(blueprint, spawn_point) # 尝试生成新玩家
            self.show_vehicle_telemetry = False # 重置遥测显示状态
            self.modify_vehicle_physics(self.player) # 应用车辆物理设置
        while self.player is None: # 如果玩家不存在 (例如首次生成或上次生成失败)
            if not self.map.get_spawn_points(): # 如果地图中没有可用的生成点
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1) # 退出程序
            spawn_points = self.map.get_spawn_points() # 获取所有可用生成点
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform() # 随机选择一个生成点
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)  # 尝试使用蓝图从生成点生成玩家
            self.show_vehicle_telemetry = False # 重置遥测显示状态
            self.modify_vehicle_physics(self.player) # 应用车辆物理设置
        # Set up the sensors.
        # 设置传感器
        self.collision_sensor = CollisionSensor(self.player, self.hud) # 初始化碰撞传感器
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud) # 初始化车道入侵传感器
        self.gnss_sensor = GnssSensor(self.player) # 初始化GNSS传感器
        self.imu_sensor = IMUSensor(self.player) # 初始化IMU传感器
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma) # 初始化摄像头管理器
        self.camera_manager.transform_index = cam_pos_index # 设置摄像头变换索引
        self.camera_manager.set_sensor(cam_index, notify=False) # 设置当前摄像头，不发送通知
        actor_type = get_actor_display_name(self.player) # 获取玩家Actor的显示名称
        self.hud.notification(actor_type) # 在HUD上显示Actor类型通知

        if self.sync: # 如果是同步模式
            self.world.tick() # 手动触发一次仿真tick
        else:
            self.world.wait_for_tick() # 异步模式下等待下一次tick

    def next_weather(self, reverse=False): # 切换到下一个天气预设
        self._weather_index += -1 if reverse else 1 # 根据 reverse 参数决定是向前还是向后切换
        self._weather_index %= len(self._weather_presets) # 循环选择天气索引
        preset = self._weather_presets[self._weather_index] # 获取选定的天气预设
        self.hud.notification('Weather: %s' % preset[1]) # 在HUD上显示天气通知
        self.player.get_world().set_weather(preset[0]) # 应用选定的天气

    def next_map_layer(self, reverse=False): # 切换到下一个地图图层
        self.current_map_layer += -1 if reverse else 1 # 根据 reverse 参数决定是向前还是向后切换
        self.current_map_layer %= len(self.map_layer_names) # 循环选择图层索引
        selected = self.map_layer_names[self.current_map_layer] # 获取选定的地图图层
        self.hud.notification('LayerMap selected: %s' % selected) # 在HUD上显示图层通知

    def load_map_layer(self, unload=False): # 加载或卸载当前选定的地图图层
        selected = self.map_layer_names[self.current_map_layer] # 获取选定的地图图层
        if unload: # 如果是卸载操作
            self.hud.notification('Unloading map layer: %s' % selected) # 显示卸载通知
            self.world.unload_map_layer(selected) # 卸载图层
        else: # 如果是加载操作
            self.hud.notification('Loading map layer: %s' % selected) # 显示加载通知
            self.world.load_map_layer(selected) # 加载图层

    def toggle_radar(self): # 切换雷达传感器的开启/关闭状态
        if self.radar_sensor is None: # 如果雷达传感器未初始化
            self.radar_sensor = RadarSensor(self.player) # 创建雷达传感器实例
        elif self.radar_sensor.sensor is not None: # 如果雷达传感器已存在且其内部sensor也存在
            self.radar_sensor.sensor.destroy() # 销毁雷达传感器
            self.radar_sensor = None # 将雷达传感器实例置为None

    def modify_vehicle_physics(self, actor): # 修改车辆的物理属性
        #If actor is not a vehicle, we cannot use the physics control
        # 如果Actor不是车辆，则无法使用物理控制
        try:
            physics_control = actor.get_physics_control() # 获取车辆的物理控制对象
            physics_control.use_sweep_wheel_collision = True # 启用车轮扫描碰撞检测，可以提高碰撞检测的真实性
            actor.apply_physics_control(physics_control) # 应用修改后的物理控制
        except Exception:
            # 如果actor不是车辆，或者其他原因导致获取物理控制失败，则忽略
            pass

    def tick(self, clock): # 每个仿真tick时调用的方法
        self.hud.tick(self, clock) # 更新HUD的显示内容

    def render(self, display): # 渲染方法，在Pygame窗口上绘制内容
        self.camera_manager.render(display) # 渲染摄像头视图
        self.hud.render(display) # 渲染HUD

    def destroy_sensors(self): # 销毁所有传感器
        self.camera_manager.sensor.destroy() # 销毁当前活动的摄像头传感器
        self.camera_manager.sensor = None # 将摄像头传感器对象置为None
        self.camera_manager.index = None # 重置摄像头索引

    def destroy(self): # 销毁世界中的主要对象
        if self.radar_sensor is not None: # 如果雷达传感器存在，则关闭它
            self.toggle_radar()
        sensors = [ # 列出所有需要销毁的传感器
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors: # 遍历并销毁每个传感器
            if sensor is not None:
                sensor.stop() # 停止传感器数据监听
                sensor.destroy() # 销毁传感器对象
        if self.player is not None: # 如果玩家Actor存在
            self.player.destroy() # 销毁玩家Actor


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object): # 定义键盘控制类
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot): # 构造函数
        # 初始化控制状态标志
        self._autopilot_enabled = start_in_autopilot # 自动驾驶是否启用
        self._ackermann_enabled = False # Ackermann 控制模式是否启用
        self._ackermann_reverse = 1 # Ackermann 控制模式下的倒车标记 (1:前进, -1:倒车)
        # 根据角色类型(车辆/行人)初始化不同的控制器
        if isinstance(world.player, carla.Vehicle):
             # 车辆控制初始化
            self._control = carla.VehicleControl() # 创建车辆控制对象
            self._ackermann_control = carla.VehicleAckermannControl() # 创建Ackermann车辆控制对象
            self._lights = carla.VehicleLightState.NONE # 初始化车辆灯光状态
            # 设置初始自动驾驶状态和灯光状态
            world.player.set_autopilot(self._autopilot_enabled) # 应用自动驾驶设置
            world.player.set_light_state(self._lights) # 应用灯光设置
        elif isinstance(world.player, carla.Walker):
            # 行人控制初始化
            self._control = carla.WalkerControl() # 创建行人控制对象
            self._autopilot_enabled = False # 行人没有自动驾驶模式
            self._rotation = world.player.get_transform().rotation # 获取行人当前朝向
        else:
            # 不支持的角色类型抛出异常
            raise NotImplementedError("Actor type not supported")
        # 初始化转向缓存值(用于平滑转向控制)
        self._steer_cache = 0.0
         # 在HUD上显示帮助提示信息(4秒)
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock, sync_mode): # 解析Pygame事件 (主要是键盘事件)
        if isinstance(self._control, carla.VehicleControl): # 如果是车辆控制
            current_lights = self._lights # 获取当前灯光状态

        for event in pygame.event.get(): # 遍历所有Pygame事件
            if event.type == pygame.QUIT: # 如果是退出事件
                return True # 返回True，表示程序应退出
            elif event.type == pygame.KEYUP: # 如果是键盘按键松开事件
                if self._is_quit_shortcut(event.key): # 检查是否是退出快捷键 (ESC或Ctrl+Q)
                    return True # 返回True，表示程序应退出

                # 以下是各种按键的功能处理
                elif event.key == K_BACKSPACE: # 退格键：重置车辆/世界
                    if self._autopilot_enabled: # 如果自动驾驶已启用
                        world.player.set_autopilot(False) # 关闭自动驾驶
                        world.restart() # 重启世界 (重新生成车辆等)
                        world.player.set_autopilot(True) # 重新开启自动驾驶
                    else:
                        world.restart() # 重启世界
                elif event.key == K_F1: # F1键：切换HUD信息显示
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT: # Shift + V：反向切换地图图层
                    world.next_map_layer(reverse=True)
                elif event.key == K_v: # V键：正向切换地图图层
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT: # Shift + B：卸载当前地图图层
                    world.load_map_layer(unload=True)
                elif event.key == K_b: # B键：加载当前地图图层
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT): # H或?键：切换帮助信息显示
                    world.hud.help.toggle()
                elif event.key == K_TAB: # TAB键：切换摄像头位置/视角
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT: # Shift + C：反向切换天气
                    world.next_weather(reverse=True)
                elif event.key == K_c: # C键：正向切换天气
                    world.next_weather()
                elif event.key == K_g: # G键：切换雷达可视化显示
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE: # `(反引号)键：切换到下一个传感器
                    world.camera_manager.next_sensor()
                elif event.key == K_n: # N键：切换到下一个传感器 (同上)
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL): # Ctrl + W：切换恒定速度模式 (60 km/h)
                    if world.constant_velocity_enabled: # 如果已启用
                        world.player.disable_constant_velocity() # 关闭恒定速度
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else: # 如果未启用
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0)) # 启用恒定速度 (17 m/s ≈ 60 km/h)
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == K_o: # O键：打开/关闭所有车门
                    try:
                        if world.doors_are_open: # 如果车门已打开
                            world.hud.notification("Closing Doors")
                            world.doors_are_open = False
                            world.player.close_door(carla.VehicleDoor.All) # 关闭所有门
                        else: # 如果车门已关闭
                            world.hud.notification("Opening doors")
                            world.doors_are_open = True
                            world.player.open_door(carla.VehicleDoor.All) # 打开所有门
                    except Exception: # 捕获可能发生的异常 (例如，非车辆Actor没有门)
                        pass
                elif event.key == K_t: # T键：切换车辆遥测信息显示
                    if world.show_vehicle_telemetry: # 如果已显示
                        world.player.show_debug_telemetry(False) # 关闭遥测
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else: # 如果未显示
                        try:
                            world.player.show_debug_telemetry(True) # 开启遥测
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception: # 捕获可能发生的异常
                            pass
                elif event.key > K_0 and event.key <= K_9: # 数字键1-9：切换到指定索引的传感器
                    index_ctrl = 0
                    if pygame.key.get_mods() & KMOD_CTRL: # 如果同时按下Ctrl键，则索引增加9 (用于更多传感器)
                        index_ctrl = 9
                    world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl) # 设置传感器
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL): # R键：切换图像录制到磁盘
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL): # Ctrl + R：切换仿真录制
                    if (world.recording_enabled): # 如果正在录制
                        client.stop_recorder() # 停止录制
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else: # 如果未录制
                        client.start_recorder("manual_recording.rec") # 开始录制到文件 "manual_recording.rec"
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL): # Ctrl + P：开始回放上次录制的仿真
                    # stop recorder
                    client.stop_recorder() # 停止任何正在进行的录制
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    # 解决回放开始时摄像头位置问题的临时方案
                    current_index = world.camera_manager.index # 记录当前摄像头索引
                    world.destroy_sensors() # 销毁当前传感器
                    # disable autopilot
                    self._autopilot_enabled = False # 关闭自动驾驶
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'") # 显示回放通知
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0) # 开始回放文件
                    world.camera_manager.set_sensor(current_index) # 恢复之前的摄像头
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL): # Ctrl + - (减号)：减少回放开始时间
                    if pygame.key.get_mods() & KMOD_SHIFT: # 如果同时按下Shift，则减少10秒
                        world.recording_start -= 10
                    else: # 否则减少1秒
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL): # Ctrl + = (等于号)：增加回放开始时间
                    if pygame.key.get_mods() & KMOD_SHIFT: # 如果同时按下Shift，则增加10秒
                        world.recording_start += 10
                    else: # 否则增加1秒
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))

                if isinstance(self._control, carla.VehicleControl): # 如果是车辆控制模式
                    if event.key == K_f: # F键：切换Ackermann控制器
                        self._ackermann_enabled = not self._ackermann_enabled # 切换状态
                        world.hud.show_ackermann_info(self._ackermann_enabled) # 在HUD上显示Ackermann控制器信息
                        world.hud.notification("Ackermann Controller %s" %
                                               ("Enabled" if self._ackermann_enabled else "Disabled"))
                    if event.key == K_q: # Q键：切换倒车档 / Ackermann倒车方向
                        if not self._ackermann_enabled: # 如果不是Ackermann模式
                            self._control.gear = 1 if self._control.reverse else -1 # 切换前进/倒车档
                        else: # 如果是Ackermann模式
                            self._ackermann_reverse *= -1 # 切换Ackermann控制的行驶方向
                            # Reset ackermann control
                            self._ackermann_control = carla.VehicleAckermannControl() # 重置Ackermann控制参数
                    elif event.key == K_m: # M键：切换手动/自动变速器
                        self._control.manual_gear_shift = not self._control.manual_gear_shift # 切换手动换挡状态
                        self._control.gear = world.player.get_control().gear # 获取当前车辆档位
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA: # , (逗号)键 (手动模式下)：降档
                        self._control.gear = max(-1, self._control.gear - 1) # 最低到倒车档(-1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD: # . (句号)键 (手动模式下)：升档
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL: # P键 (非Ctrl组合)：切换自动驾驶
                        if not self._autopilot_enabled and not sync_mode: # 如果在异步模式下启用自动驾驶，发出警告
                            print("WARNING: You are currently in asynchronous mode and could "
                                  "experience some issues with the traffic simulation")
                        self._autopilot_enabled = not self._autopilot_enabled # 切换自动驾驶状态
                        world.player.set_autopilot(self._autopilot_enabled) # 应用设置
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    # 灯光控制
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL: # Ctrl + L：切换特殊灯1 (例如警灯)
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT: # Shift + L：切换远光灯
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l: # L键：循环切换灯光 (关 -> 位置灯 -> 近光灯 -> 雾灯 -> 关)
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position: # 如果位置灯未亮
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position # 打开位置灯
                        else: # 如果位置灯已亮
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam # 打开近光灯
                        if self._lights & carla.VehicleLightState.LowBeam: # 如果近光灯已亮 (意味着之前已是近光灯或雾灯)
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog # 打开雾灯
                        if self._lights & carla.VehicleLightState.Fog: # 如果雾灯已亮 (意味着之前是雾灯，现在要关闭所有)
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position # 关闭位置灯
                            current_lights ^= carla.VehicleLightState.LowBeam # 关闭近光灯
                            current_lights ^= carla.VehicleLightState.Fog # 关闭雾灯
                    elif event.key == K_i: # I键：切换车内灯
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z: # Z键：切换左转向灯
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x: # X键：切换右转向灯
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled: # 如果未启用自动驾驶，则处理手动控制输入
            if isinstance(self._control, carla.VehicleControl): # 如果是车辆控制
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time()) # 解析持续按下的车辆控制键
                self._control.reverse = self._control.gear < 0 # 根据档位设置倒车状态
                # Set automatic control-related vehicle lights
                # 设置与控制相关的自动灯光 (刹车灯、倒车灯)
                if self._control.brake: # 如果踩了刹车
                    current_lights |= carla.VehicleLightState.Brake # 打开刹车灯
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake # 关闭刹车灯
                if self._control.reverse: # 如果是倒车状态
                    current_lights |= carla.VehicleLightState.Reverse # 打开倒车灯
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse # 关闭倒车灯
                if current_lights != self._lights: # Change the light state only if necessary
                    # 仅在灯光状态发生变化时才更新
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights)) # 应用灯光状态
                # Apply control
                # 应用控制指令
                if not self._ackermann_enabled: # 如果不是Ackermann模式
                    world.player.apply_control(self._control) # 应用标准车辆控制
                else: # 如果是Ackermann模式
                    world.player.apply_ackermann_control(self._ackermann_control) # 应用Ackermann控制
                    # Update control to the last one applied by the ackermann controller.
                    # 将标准控制对象更新为Ackermann控制器最后应用的控制状态 (用于HUD显示等)
                    self._control = world.player.get_control()
                    # Update hud with the newest ackermann control
                    # 更新HUD显示的Ackermann控制信息
                    world.hud.update_ackermann_control(self._ackermann_control)

            elif isinstance(self._control, carla.WalkerControl): # 如果是行人控制
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world) # 解析持续按下的行人控制键
                world.player.apply_control(self._control) # 应用行人控制

    def _parse_vehicle_keys(self, keys, milliseconds): # 解析车辆的持续按键输入
        # 处理加速/前进控制 (W键或上箭头)
        if keys[K_UP] or keys[K_w]:
            if not self._ackermann_enabled:
                # 普通控制模式: 增加油门(最大1.0)
                self._control.throttle = min(self._control.throttle + 0.1, 1.00)
            else:
                # Ackermann控制模式: 根据时间增量增加速度, _ackermann_reverse 控制方向
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                # 未按下加速键时重置油门
                self._control.throttle = 0.0
        # 处理减速/后退控制 (S键或下箭头)
        if keys[K_DOWN] or keys[K_s]:
            if not self._ackermann_enabled:
                # 普通控制模式: 增加刹车(最大1.0)
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                # Ackermann控制模式: 根据时间增量减少速度
                # 减去速度的绝对值乘以系数，确保减速行为在前进和后退时一致
                self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                # 确保速度不为负 (或在倒车时不为正)
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                # 未按下减速键时重置刹车
                self._control.brake = 0
        # 处理转向控制 (A/D键或左右箭头)
        steer_increment = 5e-4 * milliseconds # 转向增量，与帧时间相关以实现平滑转向
        if keys[K_LEFT] or keys[K_a]:
             # 左转: 如果是向右转状态则重置转向缓存，否则减少转向缓存 (向左)
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            # 右转: 如果是向左转状态则重置转向缓存，否则增加转向缓存 (向右)
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            # 没有转向输入时逐渐重置转向缓存 (使方向盘回正)
            self._steer_cache = 0.0 # 这里原版是直接置零，也可以改成缓慢回正
        # 限制转向幅度在[-0.7, 0.7]范围内
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        # 应用转向控制
        if not self._ackermann_enabled:
            # 普通控制模式: 设置转向值(四舍五入到小数点后1位)和手刹状态
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE] # 空格键控制手刹
        else:
            # Ackermann控制模式: 只设置转向值 (Ackermann通常不直接控制手刹)
            self._ackermann_control.steer = round(self._steer_cache, 1)

    def _parse_walker_keys(self, keys, milliseconds, world): # 解析行人的持续按键输入
        # 初始化速度为0
        self._control.speed = 0.0
        # 处理停止/减速 (S键或下箭头) - 行人通常是立即停止
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        # 处理左转 (A键或左箭头)
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01 # 设置一个微小的速度，使得转向时有视觉反馈
            # 根据时间增量减少偏航角(左转)
            self._rotation.yaw -= 0.08 * milliseconds
        # 处理右转 (D键或右箭头)
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01 # 同上
            # 根据时间增量增加偏航角(右转)
            self._rotation.yaw += 0.08 * milliseconds
        # 处理前进 (W键或上箭头)
        if keys[K_UP] or keys[K_w]:
            # 按下Shift键时使用更快的速度 (跑)
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE] # 空格键控制跳跃
        self._rotation.yaw = round(self._rotation.yaw, 1) # 偏航角四舍五入
        self._control.direction = self._rotation.get_forward_vector() # 根据旋转计算前进方向向量

    @staticmethod
    def _is_quit_shortcut(key): # 检查是否是退出快捷键
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object): # 定义HUD类，用于在屏幕上显示各种信息
    def __init__(self, width, height): # 构造函数
        self.dim = (width, height) # HUD的尺寸 (通常与屏幕尺寸相同)
        font = pygame.font.Font(pygame.font.get_default_font(), 20) # 获取默认字体，字号20
        font_name = 'courier' if os.name == 'nt' else 'mono' # 根据操作系统选择等宽字体名称
        fonts = [x for x in pygame.font.get_fonts() if font_name in x] # 查找包含该名称的字体
        default_font = 'ubuntumono' # 默认等宽字体
        mono = default_font if default_font in fonts else fonts[0] # 选择最终的等宽字体
        mono = pygame.font.match_font(mono) # 匹配字体
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14) # 创建等宽字体对象
        self._notifications = FadingText(font, (width, 40), (0, height - 40)) # 创建用于显示通知的FadingText对象
        self.help = HelpText(pygame.font.Font(mono, 16), width, height) # 创建用于显示帮助信息的HelpText对象
        self.server_fps = 0 # 服务器FPS
        self.frame = 0 # 当前帧号
        self.simulation_time = 0 # 仿真时间
        self._show_info = True # 是否显示详细信息
        self._info_text = [] # 存储详细信息的文本列表
        self._server_clock = pygame.time.Clock() # 用于计算服务器FPS的时钟

        self._show_ackermann_info = False # 是否显示Ackermann控制器信息
        self._ackermann_control = carla.VehicleAckermannControl() # Ackermann控制器状态 (用于显示)

    def on_world_tick(self, timestamp): # 当CARLA世界tick时调用的回调函数
        self._server_clock.tick() # 更新服务器时钟
        self.server_fps = self._server_clock.get_fps() # 计算服务器FPS
        self.frame = timestamp.frame # 获取当前帧号
        self.simulation_time = timestamp.elapsed_seconds # 获取仿真总运行时间

    def tick(self, world, clock): # HUD的tick方法，在每帧调用以更新显示内容
        self._notifications.tick(world, clock) # 更新通知文本的淡出效果
        if not self._show_info: # 如果不显示详细信息，则直接返回
            return
        t = world.player.get_transform() # 获取玩家的位置和姿态
        v = world.player.get_velocity() # 获取玩家的速度
        c = world.player.get_control() # 获取玩家的控制状态
        compass = world.imu_sensor.compass # 获取IMU传感器的指南针数据 (航向角)
        heading = 'N' if compass > 270.5 or compass < 89.5 else '' # 根据航向角判断基本方向 (北)
        heading += 'S' if 90.5 < compass < 269.5 else '' # (南)
        heading += 'E' if 0.5 < compass < 179.5 else '' # (东)
        heading += 'W' if 180.5 < compass < 359.5 else '' # (西)
        colhist = world.collision_sensor.get_collision_history() # 获取碰撞历史记录
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)] # 获取最近200帧的碰撞数据
        max_col = max(1.0, max(collision)) # 计算最大碰撞强度 (至少为1.0，避免除零)
        collision = [x / max_col for x in collision] # 归一化碰撞数据
        vehicles = world.world.get_actors().filter('vehicle.*') # 获取场景中所有车辆Actor
        self._info_text = [ # 构建要显示的文本信息列表
            'Server:  % 16.0f FPS' % self.server_fps, # 服务器FPS
            'Client:  % 16.0f FPS' % clock.get_fps(), # 客户端FPS (Pygame渲染帧率)
            '', # 空行
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20), # 玩家车辆名称
            'Map:     % 20s' % world.map.name.split('/')[-1], # 当前地图名称
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)), # 仿真运行时间
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)), # 车辆速度 (km/h)
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading), # 指南针读数和方向
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer), # 加速度计读数
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope), # 陀螺仪读数
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)), # 车辆位置 (X, Y)
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)), # GNSS经纬度
            'Height:  % 18.0f m' % t.location.z, # 车辆高度 (Z)
            '']
        if isinstance(c, carla.VehicleControl): # 如果玩家是车辆
            self._info_text += [ # 添加车辆控制相关信息
                ('Throttle:', c.throttle, 0.0, 1.0), # 油门 (值, 最小值, 最大值) - 用于绘制进度条
                ('Steer:', c.steer, -1.0, 1.0), # 转向
                ('Brake:', c.brake, 0.0, 1.0), # 刹车
                ('Reverse:', c.reverse), # 倒车状态 (布尔值) - 用于绘制复选框
                ('Hand brake:', c.hand_brake), # 手刹状态
                ('Manual:', c.manual_gear_shift), # 手动换挡状态
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)] # 当前档位
            if self._show_ackermann_info: # 如果启用了Ackermann信息显示
                self._info_text += [ # 添加Ackermann控制器信息
                    '',
                    'Ackermann Controller:',
                    '  Target speed: % 8.0f km/h' % (3.6*self._ackermann_control.speed), # 目标速度
                ]
        elif isinstance(c, carla.WalkerControl): # 如果玩家是行人
            self._info_text += [ # 添加行人控制相关信息
                ('Speed:', c.speed, 0.0, 5.556), # 速度
                ('Jump:', c.jump)] # 跳跃状态
        self._info_text += [ # 添加碰撞和车辆列表信息
            '',
            'Collision:', # 碰撞信息标题
            collision, # 碰撞历史数据 (用于绘制图表)
            '',
            'Number of vehicles: % 8d' % len(vehicles)] # 场景中车辆总数
        if len(vehicles) > 1: # 如果场景中有多于一辆车 (包括玩家)
            self._info_text += ['Nearby vehicles:'] # 附近车辆信息标题
            # 计算其他车辆与玩家的距离
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id] # 排除玩家自身
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]): # 按距离排序
                if d > 200.0: # 只显示200米范围内的车辆
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22) # 获取车辆类型名称
                self._info_text.append('% 4dm %s' % (d, vehicle_type)) # 添加到显示列表

    def show_ackermann_info(self, enabled): # 设置是否显示Ackermann控制器信息
        self._show_ackermann_info = enabled

    def update_ackermann_control(self, ackermann_control): # 更新用于显示的Ackermann控制器状态
        self._ackermann_control = ackermann_control

    def toggle_info(self): # 切换详细信息的显示/隐藏
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0): # 显示一条通知信息
        self._notifications.set_text(text, seconds=seconds) # 设置通知文本和持续时间

    def error(self, text): # 显示一条错误信息 (红色文本)
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display): # 渲染HUD到Pygame显示表面
        if self._show_info: # 如果需要显示详细信息
            info_surface = pygame.Surface((220, self.dim[1])) # 创建信息面板的表面
            info_surface.set_alpha(100) # 设置半透明效果
            display.blit(info_surface, (0, 0)) # 将信息面板绘制到主显示表面
            v_offset = 4 # 垂直偏移量，用于逐行绘制文本
            bar_h_offset = 100 # 进度条的水平偏移量
            bar_width = 106 # 进度条的宽度
            for item in self._info_text: # 遍历信息列表
                if v_offset + 18 > self.dim[1]: # 如果超出屏幕高度，则停止绘制
                    break
                if isinstance(item, list): # 如果是列表 (通常是碰撞数据)
                    if len(item) > 1: # 绘制碰撞历史图表
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2) # 橙色线条
                    item = None # 清空item，避免后续作为文本处理
                    v_offset += 18 # 增加垂直偏移 (图表占用的空间)
                elif isinstance(item, tuple): # 如果是元组 (通常是带进度条或复选框的信息)
                    if isinstance(item[1], bool): # 如果第二个元素是布尔值 (复选框)
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1) # 选中则填充，否则只画边框
                    else: # 否则是带范围的值 (进度条)
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1) # 绘制进度条边框
                        f = (item[1] - item[2]) / (item[3] - item[2]) # 计算当前值在范围内的比例
                        if item[2] < 0.0: # 如果范围包含负值 (例如转向)
                            # 进度条从中点开始绘制
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else: # 如果范围从0开始 (例如油门)
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect) # 绘制进度条填充部分
                    item = item[0] # 取元组的第一个元素作为要显示的文本
                if item:  # At this point has to be a str.
                    # 此时item应该是字符串，渲染并绘制文本
                    surface = self._font_mono.render(item, True, (255, 255, 255)) # 白色文本
                    display.blit(surface, (8, v_offset)) # 绘制到屏幕
                v_offset += 18 # 增加垂直偏移，准备绘制下一行
        self._notifications.render(display) # 渲染通知文本
        self.help.render(display) # 渲染帮助文本 (如果已启用)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object): # 定义淡出文本类
    def __init__(self, font, dim, pos): # 构造函数
        self.font = font # Pygame字体对象
        self.dim = dim # 文本区域的尺寸 (宽度, 高度)
        self.pos = pos # 文本区域在屏幕上的位置 (x, y)
        self.seconds_left = 0 # 文本剩余显示时间 (秒)
        self.surface = pygame.Surface(self.dim) # 创建用于渲染文本的Pygame表面

    def set_text(self, text, color=(255, 255, 255), seconds=2.0): # 设置要显示的文本
        text_texture = self.font.render(text, True, color) # 使用指定颜色渲染文本
        self.surface = pygame.Surface(self.dim) # 重新创建表面 (清空旧内容)
        self.seconds_left = seconds # 设置文本显示时长
        self.surface.fill((0, 0, 0, 0)) # 用透明颜色填充表面 (RGBA)
        self.surface.blit(text_texture, (10, 11)) # 将渲染好的文本绘制到表面上 (带有一些内边距)

    def tick(self, _, clock): # FadingText的tick方法，在每帧调用以更新淡出效果
        delta_seconds = 1e-3 * clock.get_time() # 获取自上一帧以来的时间差 (秒)
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds) # 更新剩余显示时间，不小于0
        self.surface.set_alpha(500.0 * self.seconds_left) # 根据剩余时间设置表面的透明度，实现淡出效果

    def render(self, display): # 渲染FadingText到主显示表面
        display.blit(self.surface, self.pos) # 将文本表面绘制到指定位置


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object): # 定义帮助文本类
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height): # 构造函数
        lines = __doc__.split('\n') # 从脚本的文档字符串 (docstring) 中获取帮助文本内容，按行分割
        self.font = font # Pygame字体对象
        self.line_space = 18 # 行间距
        self.dim = (780, len(lines) * self.line_space + 12) # 计算帮助文本区域的尺寸
        # 计算帮助文本在屏幕上的居中位置
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0 # (未使用) 兼容性或未来扩展
        self.surface = pygame.Surface(self.dim) # 创建用于渲染帮助文本的Pygame表面
        self.surface.fill((0, 0, 0, 0)) # 用透明颜色填充表面
        for n, line in enumerate(lines): # 遍历每一行帮助文本
            text_texture = self.font.render(line, True, (255, 255, 255)) # 白色渲染文本
            self.surface.blit(text_texture, (22, n * self.line_space)) # 将渲染好的文本行绘制到表面上
            self._render = False # 初始状态下不显示帮助文本
        self.surface.set_alpha(220) # 设置帮助文本表面的透明度 (轻微透明)

    def toggle(self): # 切换帮助文本的显示/隐藏状态
        self._render = not self._render

    def render(self, display): # 渲染帮助文本到主显示表面
        if self._render: # 如果需要显示帮助文本
            display.blit(self.surface, self.pos) # 将帮助文本表面绘制到计算好的居中位置


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object): # 定义碰撞传感器类
    def __init__(self, parent_actor, hud): # 构造函数
        self.sensor = None # 传感器对象
        self.history = [] # 存储碰撞历史记录的列表
        self._parent = parent_actor # 传感器的父Actor (通常是玩家车辆)
        self.hud = hud # HUD对象，用于显示碰撞通知
        world = self._parent.get_world() # 获取父Actor所在的世界
        bp = world.get_blueprint_library().find('sensor.other.collision') # 查找碰撞传感器的蓝图
        # 在父Actor的位置生成碰撞传感器，并附加到父Actor上
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        # 我们需要给lambda回调函数传递self的弱引用，以避免循环引用
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event)) # 监听碰撞事件

    def get_collision_history(self): # 获取碰撞历史数据 (用于HUD显示)
        history = collections.defaultdict(int) # 使用defaultdict存储每帧的碰撞强度总和
        for frame, intensity in self.history: # 遍历记录的碰撞事件
            history[frame] += intensity # 累加同一帧的碰撞强度
        return history

    @staticmethod
    def _on_collision(weak_self, event): # 碰撞事件的回调函数 (静态方法)
        self = weak_self() # 获取CollisionSensor的实例 (从弱引用)
        if not self: # 如果实例已被销毁，则返回
            return
        actor_type = get_actor_display_name(event.other_actor) # 获取与之碰撞的Actor的类型名称
        self.hud.notification('Collision with %r' % actor_type) # 在HUD上显示碰撞通知
        impulse = event.normal_impulse # 获取碰撞的法向冲量
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2) # 计算碰撞强度 (冲量大小)
        self.history.append((event.frame, intensity)) # 将碰撞事件 (帧号, 强度) 添加到历史记录
        if len(self.history) > 4000: # 限制历史记录的长度，防止内存无限增长
            self.history.pop(0) # 移除最早的记录


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object): # 定义车道入侵传感器类
    def __init__(self, parent_actor, hud): # 构造函数
        self.sensor = None # 传感器对象

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        # 如果生成对象不是车辆，则无法使用车道入侵传感器
        if parent_actor.type_id.startswith("vehicle."): # 仅对车辆类型的Actor设置此传感器
            self._parent = parent_actor # 传感器的父Actor
            self.hud = hud # HUD对象，用于显示车道入侵通知
            world = self._parent.get_world() # 获取父Actor所在的世界
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion') # 查找车道入侵传感器的蓝图
            # 生成传感器并附加到父Actor
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            # 传递弱引用以避免循环引用
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event)) # 监听车道入侵事件

    @staticmethod
    def _on_invasion(weak_self, event): # 车道入侵事件的回调函数 (静态方法)
        self = weak_self() # 获取LaneInvasionSensor的实例
        if not self: # 如果实例已被销毁，则返回
            return
        lane_types = set(x.type for x in event.crossed_lane_markings) # 获取所有压过的车道线类型
        text = ['%r' % str(x).split()[-1] for x in lane_types] # 格式化车道线类型名称 (例如 'Solid' 而不是 'carla.LaneMarkingType.Solid')
        self.hud.notification('Crossed line %s' % ' and '.join(text)) # 在HUD上显示通知


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object): # 定义GNSS传感器类
    def __init__(self, parent_actor): # 构造函数
        self.sensor = None # 传感器对象
        self._parent = parent_actor # 传感器的父Actor
        self.lat = 0.0 # 纬度
        self.lon = 0.0 # 经度
        world = self._parent.get_world() # 获取父Actor所在的世界
        bp = world.get_blueprint_library().find('sensor.other.gnss') # 查找GNSS传感器的蓝图
        # 生成GNSS传感器，设置其相对父Actor的位置 (x=1.0, z=2.8)，并附加
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        # 传递弱引用以避免循环引用
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event)) # 监听GNSS数据事件

    @staticmethod
    def _on_gnss_event(weak_self, event): # GNSS数据事件的回调函数 (静态方法)
        self = weak_self() # 获取GnssSensor的实例
        if not self: # 如果实例已被销毁，则返回
            return
        self.lat = event.latitude # 更新纬度值
        self.lon = event.longitude # 更新经度值


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object): # 定义IMU传感器类
    def __init__(self, parent_actor): # 构造函数
        self.sensor = None # 传感器对象
        self._parent = parent_actor # 传感器的父Actor
        self.accelerometer = (0.0, 0.0, 0.0) # 加速度计读数 (x, y, z)
        self.gyroscope = (0.0, 0.0, 0.0) # 陀螺仪读数 (x, y, z)
        self.compass = 0.0 # 指南针读数 (航向角，单位：度)
        world = self._parent.get_world() # 获取父Actor所在的世界
        bp = world.get_blueprint_library().find('sensor.other.imu') # 查找IMU传感器的蓝图
        # 生成IMU传感器并附加到父Actor
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # 我们需要给self传递一个lambda弱引用，以避免循环引用。
        # 弱引用并不会增加引用数（不会阻止对象被垃圾回收），但是它会关联目标对象
        # 避免因为缓存导致对象无法回收，从而导致内存泄漏（既能拿到这个对象，又不影响它的内存回收）
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data)) # 监听IMU数据事件

    @staticmethod
    def _IMU_callback(weak_self, sensor_data): # IMU数据事件的回调函数 (静态方法)
        self = weak_self() # 获取IMUSensor的实例
        if not self: # 如果实例已被销毁，则返回
            return
        limits = (-99.9, 99.9) # IMU读数的显示限制范围
        # 更新加速度计读数，并限制在设定范围内
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        # 更新陀螺仪读数 (原始数据是弧度/秒，转换为度/秒)，并限制在设定范围内
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass) # 更新指南针读数 (原始数据是弧度，转换为度)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object): # 定义雷达传感器类
    def __init__(self, parent_actor): # 构造函数
        self.sensor = None # 传感器对象
        self._parent = parent_actor # 传感器的父Actor
        # 计算雷达传感器的安装位置，使其在车辆包围盒的前方和上方一点
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y # (未使用)
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s, 雷达检测的速度范围，用于颜色编码
        world = self._parent.get_world() # 获取父Actor所在的世界
        self.debug = world.debug # 获取世界的调试绘图接口
        bp = world.get_blueprint_library().find('sensor.other.radar') # 查找雷达传感器的蓝图
        bp.set_attribute('horizontal_fov', str(35)) # 设置水平视场角 (度)
        bp.set_attribute('vertical_fov', str(20)) # 设置垂直视场角 (度)
        # 生成雷达传感器，设置其位置和姿态 (稍微向上倾斜5度)，并附加到父Actor
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05), # 位于车辆前方和上方
                carla.Rotation(pitch=5)), # 向上倾斜5度
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        # 传递弱引用以避免循环引用
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data)) # 监听雷达数据事件

    @staticmethod
    def _Radar_callback(weak_self, radar_data): # 雷达数据事件的回调函数 (静态方法)
        self = weak_self() # 获取RadarSensor的实例
        if not self: # 如果实例已被销毁，则返回
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))
        # 以上注释演示了如何将原始雷达数据转换为Numpy数组

        current_rot = radar_data.transform.rotation # 获取雷达传感器当前的旋转姿态
        for detect in radar_data: # 遍历雷达检测到的每个点
            azi = math.degrees(detect.azimuth) # 方位角 (度)
            alt = math.degrees(detect.altitude) # 高度角 (度)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            # -0.25是为了调整距离，使得绘制的点更容易被看到
            fw_vec = carla.Vector3D(x=detect.depth - 0.25) # 创建一个表示检测点深度方向的向量
            # 将检测点从雷达的局部坐标系转换到世界坐标系
            carla.Transform(
                carla.Location(), # 相对位置为原点
                carla.Rotation( # 旋转基于雷达的当前姿态和检测点的方位角、高度角
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec) # 应用变换

            def clamp(min_v, max_v, value): # 辅助函数：将值限制在最小和最大值之间
                return max(min_v, min(value, max_v))

            # 根据检测点的相对速度进行颜色编码
            norm_velocity = detect.velocity / self.velocity_range # 归一化速度到 [-1, 1] 范围
            # 计算RGB颜色值：红色表示远离，蓝色表示靠近，绿色表示速度接近零
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            # 使用调试接口在世界中绘制一个点来表示雷达检测
            self.debug.draw_point(
                radar_data.transform.location + fw_vec, # 点在世界中的位置
                size=0.075, # 点的大小
                life_time=0.06, # 点的持续时间 (秒)
                persistent_lines=False, # 是否持久显示 (否)
                color=carla.Color(r, g, b)) # 点的颜色

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object): # 定义摄像头管理器类
    def __init__(self, parent_actor, hud, gamma_correction): # 构造函数
        self.sensor = None # 当前活动的传感器对象 (摄像头或Lidar)
        self.surface = None # 用于在Pygame上显示的表面
        self._parent = parent_actor # 传感器的父Actor
        self.hud = hud # HUD对象
        self.recording = False # 是否正在录制图像
        # 计算摄像头安装位置的参考点 (基于父Actor的包围盒)
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType # CARLA的附加类型枚举

        if not self._parent.type_id.startswith("walker.pedestrian"): # 如果父Actor不是行人 (即是车辆)
            self._camera_transforms = [ # 定义车辆的摄像头预设位置和附加类型
                # (变换, 附加类型)
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost), # 车后追随视角 (弹性臂)
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid), # 驾驶员视角 (刚性)
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArmGhost), # 车侧视角
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost), # 更远的追随视角
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)] # 侧下方视角
        else: # 如果父Actor是行人
            self._camera_transforms = [ # 定义行人的摄像头预设位置和附加类型
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost), # 行人后方追随
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid), # 行人第一人称视角
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost), # 行人侧后方
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost), # 行人鸟瞰追随
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)] # 行人侧面固定

        self.transform_index = 1 # 当前使用的摄像头变换索引 (默认为驾驶员/第一人称视角)
        self.sensors = [ # 定义可用的传感器列表
            # [蓝图名称, 颜色转换类型, 显示名称, 额外属性字典]
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}], # 深度图 (灰度)
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}], # 对数深度图 (灰度)
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}], # 语义分割 (原始数据)
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}], # 语义分割 (CityScapes调色板)
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette, 'Camera Instance Segmentation (CityScapes Palette)', {}], # 实例分割 (CityScapes调色板)
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}], # 实例分割 (原始数据)
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}], # Lidar传感器，范围50米
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}], # DVS动态视觉传感器
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted', # 带畸变的RGB摄像头
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}], # 光流传感器
            ['sensor.camera.normals', cc.Raw, 'Camera Normals', {}], #法线相机
        ]
        world = self._parent.get_world() # 获取世界对象
        bp_library = world.get_blueprint_library() # 获取蓝图库
        for item in self.sensors: # 遍历传感器列表，配置蓝图属性
            bp = bp_library.find(item[0]) # 查找传感器蓝图
            if item[0].startswith('sensor.camera'): # 如果是摄像头类传感器
                bp.set_attribute('image_size_x', str(hud.dim[0])) # 设置图像宽度
                bp.set_attribute('image_size_y', str(hud.dim[1])) # 设置图像高度
                if bp.has_attribute('gamma'): # 如果有gamma属性
                    bp.set_attribute('gamma', str(gamma_correction)) # 设置gamma校正值
                for attr_name, attr_value in item[3].items(): # 设置其他额外属性
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'): # 如果是Lidar传感器
                self.lidar_range = 50 # 默认Lidar范围
                for attr_name, attr_value in item[3].items(): # 设置Lidar属性
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range': # 如果是范围属性
                        self.lidar_range = float(attr_value) # 更新Lidar范围值

            item.append(bp) # 将配置好的蓝图对象添加到传感器信息列表中
        self.index = None # 当前选中的传感器在self.sensors列表中的索引，初始为None

    def toggle_camera(self): # 切换摄像头的变换位置/视角
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms) # 循环选择下一个变换索引
        # 重新设置传感器，强制重新生成，因为附加类型可能改变
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False): # 设置并激活一个传感器
        index = index % len(self.sensors) # 确保索引在有效范围内
        # 判断是否需要重新生成传感器：
        # 1. 当前没有传感器 (self.index is None)
        # 2. 强制重新生成 (force_respawn is True)
        # 3. 新旧传感器的显示名称不同 (意味着传感器类型改变)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn: # 如果需要重新生成传感器
            if self.sensor is not None: # 如果已存在传感器，则销毁它
                self.sensor.destroy()
                self.surface = None # 清空显示表面
            # 生成新的传感器实例
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1], # 使用预配置的蓝图
                self._camera_transforms[self.transform_index][0], # 使用当前选定的变换
                attach_to=self._parent, # 附加到父Actor
                attachment_type=self._camera_transforms[self.transform_index][1]) # 使用当前选定的附加类型
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            # 传递弱引用以避免循环引用
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image)) # 监听传感器数据
        if notify: # 如果需要通知
            self.hud.notification(self.sensors[index][2]) # 在HUD上显示当前传感器名称
        self.index = index # 更新当前传感器的索引

    def next_sensor(self): # 切换到下一个传感器类型
        self.set_sensor(self.index + 1) # 调用set_sensor，索引加1

    def toggle_recording(self): # 切换图像录制状态
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off')) # 显示录制状态通知

    def render(self, display): # 将传感器数据显示到Pygame窗口
        if self.surface is not None: # 如果有可供显示的表面数据
            display.blit(self.surface, (0, 0)) # 将表面绘制到屏幕左上角

    @staticmethod
    def _parse_image(weak_self, image): # 处理传感器数据的回调函数 (静态方法)
        self = weak_self() # 获取CameraManager实例
        if not self: # 如果实例已被销毁，则返回
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'): # 如果是Lidar传感器数据
            # 将Lidar原始数据转换为点云，并投影到2D平面进行可视化
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4')) # 从原始数据创建Numpy数组 (float32)
            points = np.reshape(points, (int(points.shape[0] / 4), 4)) # 重塑为 N x 4 数组 (x, y, z, intensity)
            lidar_data = np.array(points[:, :2]) # 只取 x, y 坐标
            # 将Lidar点缩放到HUD的尺寸，并居中显示
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111 (忽略fabs相关的pylint警告)
            lidar_data = lidar_data.astype(np.int32) # 转换为整数坐标
            lidar_data = np.reshape(lidar_data, (-1, 2)) # 确保是 N x 2 数组
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3) # Lidar可视化图像的尺寸
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8) # 创建黑色背景图像
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255) # 将Lidar点绘制为白色
            self.surface = pygame.surfarray.make_surface(lidar_img) # 创建Pygame表面
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'): # 如果是DVS（动态视觉传感器）数据
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            # 将DVS事件数据转换为图像进行可视化
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([ # 从原始数据创建结构化Numpy数组
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)])) # (x, y, timestamp, polarity)
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8) # 创建黑色背景图像
            # Blue is positive, red is negative
            # 蓝色表示正极性事件，红色表示负极性事件
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1)) # 创建Pygame表面 (需要交换轴)
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'): # 如果是光流传感器数据
            image = image.get_color_coded_flow() # 获取颜色编码的光流图像
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) # 转换为Numpy数组
            array = np.reshape(array, (image.height, image.width, 4)) # BGRA格式
            array = array[:, :, :3] # 取BGR通道
            array = array[:, :, ::-1] # 转换为RGB格式
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1)) # 创建Pygame表面 (需要交换轴)
        else: # 如果是其他摄像头类型 (RGB, Depth, Segmentation等)
            image.convert(self.sensors[self.index][1]) # 使用预设的颜色转换方法处理图像
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) # 转换为Numpy数组
            array = np.reshape(array, (image.height, image.width, 4)) # BGRA格式
            array = array[:, :, :3] # 取BGR通道
            array = array[:, :, ::-1] # 转换为RGB格式 (Pygame使用RGB)
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1)) # 创建Pygame表面 (Pygame的surfarray通常需要交换x和y轴)
        if self.recording: # 如果启用了录制
            image.save_to_disk('_out/%08d' % image.frame) # 将图像保存到磁盘，文件名格式为帧号


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args): # 主游戏循环函数
    # 初始化 pygame 和字体模块
    pygame.init()
    pygame.font.init()
    world = None # 初始化world变量
    original_settings = None  # 用于记录原始仿真设置（以便退出时恢复）

    try:
        # 创建 CARLA 客户端，连接指定主机和端口
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)  # 设置连接超时时间为 2 秒

        sim_world = client.get_world()  # 获取仿真世界

        # 如果启用了同步模式（同步仿真帧和主循环）
        if args.sync:
            original_settings = sim_world.get_settings()  # 记录原始设置
            settings = sim_world.get_settings()  # 获取当前设置
            if not settings.synchronous_mode: # 如果当前不是同步模式
                # 启用同步模式，并设置固定步长（即每帧为 0.05 秒 = 20 FPS）
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)  # 应用设置

            # 设置交通管理器也为同步模式，确保交通控制和仿真帧同步
            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        # 警告：如果启用自动驾驶但不在同步模式下，可能会导致交通系统问题
        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        # 设置渲染窗口（HWSURFACE：硬件加速，DOUBLEBUF：双缓冲）
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0, 0, 0))  # 背景填充为黑色
        pygame.display.flip()    # 显示更新后的画面

        # 初始化 HUD（用于显示信息的图层）
        hud = HUD(args.width, args.height)

        # 创建 World 实例，封装了车辆、摄像头等逻辑
        world = World(sim_world, hud, args)

        # 初始化键盘控制器（可选：控制车辆或切换模式等）
        controller = KeyboardControl(world, args.autopilot)
        # 注：在某些场景（如“湖工商”）中可能注释此行以禁用用户交互 (此注释来自用户提供的原始文件)

        # 在同步模式下，需要手动推进仿真一帧，确保所有Actor都已生成并准备就绪
        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()  # 异步模式下等待下一帧自动到来

        clock = pygame.time.Clock()  # 用于控制帧率的时钟对象

        # 主循环：每帧处理输入、更新世界状态、渲染画面
        while True:
            if args.sync: # 如果是同步模式
                sim_world.tick()  # 手动推进仿真一帧

            clock.tick_busy_loop(60)  # 尝试保持 60 FPS 的运行频率 (忙等待)

            # 解析并处理用户输入事件（如退出、键盘操作等）
            if controller.parse_events(client, world, clock, args.sync):
                return  # 如果parse_events返回True (例如按下退出键)，则退出主循环

            world.tick(clock)         # 更新世界状态（如 HUD 信息、传感器数据处理等）
            world.render(display)     # 渲染当前帧到 pygame 显示窗口 (摄像头视图和HUD)
            pygame.display.flip()     # 刷新屏幕显示最新画面

    finally: # 无论循环如何结束 (正常退出或异常)，都会执行finally块中的代码

        if original_settings: # 如果记录了原始设置 (通常在同步模式下)
            sim_world.apply_settings(original_settings) # 恢复原始仿真设置

        if (world and world.recording_enabled): # 如果正在录制仿真
            client.stop_recorder() # 停止录制

        if world is not None: # 如果world对象已创建
            world.destroy() # 清理world对象 (销毁Actor和传感器)

        pygame.quit() # 退出Pygame


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main(): # 主函数，程序入口
    argparser = argparse.ArgumentParser( # 创建参数解析器
        description='CARLA Manual Control Client')
    # 定义命令行参数
    argparser.add_argument( # 详细模式
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument( # CARLA服务器主机IP
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument( # CARLA服务器端口
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument( # 启用自动驾驶模式
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument( # 窗口分辨率
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument( # Actor筛选器 (例如 'vehicle.*', 'walker.pedestrian.*')
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument( # Actor代数 (用于选择特定版本的蓝图)
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument( # 主角Actor的角色名
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument( # 摄像头的Gamma校正值
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument( # 启用同步模式
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    args = argparser.parse_args() # 解析命令行参数

    args.width, args.height = [int(x) for x in args.res.split('x')] # 从分辨率字符串解析宽度和高度

    # 设置日志级别
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port) # 记录连接信息

    print(__doc__) # 打印脚本开头的文档字符串 (帮助信息)

    try:
        game_loop(args) # 调用主游戏循环

    except KeyboardInterrupt: # 捕获Ctrl+C中断
        print('\nCancelled by user. Bye!')


if __name__ == '__main__': # 如果脚本作为主程序运行
    main() # 调用main函数