import carla
import random

import time
import logging
from typing import Optional

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carla 连接配置
CARLA_CONNECT_TIMEOUT = 30  # 连接超时时间（秒）
CARLA_RECONNECT_RETRIES = 3  # 重连重试次数
MAP_LOAD_RETRIES = 2  # 地图加载重试次数

class MatrixWorld(object):
    """Builds the world, cars, sensors etc."""

    # hardcoded 2d view properties
    _X = 3
    _Y = 0
    _Z = 10

    def __init__(self, client, im_width=480.0, im_height=480.0, render=True,
                 weather=None, fast=False, town='Town02'):
        """The MatrixWorld class manages the building properties of the Carla world. """
        self.im_width = im_width
        self.im_height = im_height
        self.client = client
        self.world = None  # 初始化world为None
        self.weather = weather
        self.bp_lib = None
        self.spawn_points = []
        self.actor_list = []
        self.yaw = 0

        try:
            # 加载地图（增加重试）
            for retry in range(MAP_LOAD_RETRIES):
                try:
                    self.world = self.client.load_world(town)
                    logger.info(f"成功加载地图: {town}")
                    break
                except Exception as e:
                    logger.warning(f"加载地图 {town} 失败 (重试 {retry + 1}/{MAP_LOAD_RETRIES}): {e}")
                    time.sleep(1)
                    if retry == MAP_LOAD_RETRIES - 1:
                        raise RuntimeError(f"地图 {town} 加载失败，已重试 {MAP_LOAD_RETRIES} 次")

            # 设置天气（增加异常捕获）
            if self.weather is not None:
                try:
                    self.world.set_weather(self.weather)
                    logger.info(f"成功设置天气: {self.weather}")
                except Exception as e:
                    logger.warning(f"设置天气失败: {e}，使用默认天气")

            # 渲染/帧率设置
            if not render:
                try:
                    settings = self.world.get_settings()
                    settings.no_rendering_mode = True
                    self.world.apply_settings(settings)
                except Exception as e:
                    logger.warning(f"关闭渲染模式失败: {e}")

            if fast:
                try:
                    settings = self.world.get_settings()
                    settings.fixed_delta_seconds = 0.05
                    self.world.apply_settings(settings)
                    logger.info("设置快速仿真模式（fixed_delta_seconds=0.05）")
                except Exception as e:
                    logger.warning(f"设置快速仿真模式失败: {e}")

            self.bp_lib = self.world.get_blueprint_library()

        except Exception as e:
            logger.error(f"MatrixWorld 初始化失败: {e}")
            # 清理已创建的actor
            self.clean_world()
            raise

    def change_map(self, map_name=None):
        """切换地图（增加异常处理）"""
        if map_name is None:
            map_id = random.choice([7, 2])
            map_name = "Town0{}".format(map_id)

        try:
            # 切换前先清理actor
            self.clean_world()
            # 加载新地图（重试机制）
            for retry in range(MAP_LOAD_RETRIES):
                try:
                    self.world = self.client.load_world(map_name)
                    logger.info(f"成功切换到地图: {map_name}")
                    break
                except Exception as e:
                    logger.warning(f"切换地图 {map_name} 失败 (重试 {retry + 1}/{MAP_LOAD_RETRIES}): {e}")
                    time.sleep(1)
                    if retry == MAP_LOAD_RETRIES - 1:
                        raise RuntimeError(f"切换地图 {map_name} 失败，已重试 {MAP_LOAD_RETRIES} 次")

            self.bp_lib = self.world.get_blueprint_library()
            self.spawn_points = []
            return self.world
        except Exception as e:
            logger.error(f"切换地图失败: {e}")
            raise

    def is_close_to_junction(self, location, max_d):
        """Checks the next waypoints to given locations are junction."""
        for d in range(5, max_d):
            wp = self.world.get_map().get_waypoint(location, carla.LaneType.Driving)
            target_wps = wp.next(d)
            for target_wp in target_wps:
                if target_wp.is_junction:
                    return True
        return False

    def generate_near_junction_points(self, max_d):
        """Generate spawn points to near junction."""
        spawn_points = self.world.get_map().get_spawn_points()
        near_junction_points = []
        for transform in spawn_points:
            if self.is_close_to_junction(transform.location, max_d):
                near_junction_points.append(transform)
        return near_junction_points

    def get_point_near_junction(self, max_d):
        """Returns a transform near junction waypoint."""
        if len(self.spawn_points) < 1:
            self.spawn_points = self.generate_near_junction_points(max_d)
        transform = random.choice(self.spawn_points)
        return transform

    def spawn_vehicle(self, transform=None, tr_spectator=True, near_junction=False):
        """Spawns a vehicle on transform point. If the transform
        not setted spawns on random spawn point. Optionally transfroms
        the spectator to spawn point."""
        # vehicle_bp = self.bp_lib.filter('vehicle.audi.tt')[0]


        vehicle_bp = self.bp_lib.filter('vehicle.tesla.model3')[0]
        #vehicle_bp = self.bp_lib.filter('vehicle.mini.cooperst')[0]
        if not transform and not near_junction:
            transform = random.choice(self.world.get_map().get_spawn_points())
        elif near_junction:
            transform = self.get_point_near_junction(max_d=50)

        vehicle = self.world.spawn_actor(vehicle_bp, transform)
        self.actor_list.append(vehicle)

        if tr_spectator:
            # Wait for world to get the vehicle actor
            self.world.tick()
            world_snapshot = self.world.wait_for_tick()
            actor_snapshot = world_snapshot.find(vehicle.id)
            # Set spectator at given transform (vehicle transform)
            spectator = self.world.get_spectator()
            actor_trans = actor_snapshot.get_transform()
            spectator.set_transform(actor_trans)
        return vehicle

    def spawn_sensor(self, sensor, vehicle, location, args=None):
        """Spawns image sensors (rgb or semantic) in 2d view."""
        sensor_bp = self.bp_lib.find(sensor)
        sensor_bp.set_attribute("image_size_x", str(self.im_width))
        sensor_bp.set_attribute("image_size_y", str(self.im_width))
        sensor_bp.set_attribute('sensor_tick', '0.1')

        actor_trans = vehicle.get_transform()

        # apply 2d view
        yaw = actor_trans.rotation.yaw
        roll = actor_trans.rotation.roll

        sensor_transform = carla.Transform(
            carla.Location(self._X, self._Y, self._Z),
            carla.Rotation(pitch=-90, roll=roll, yaw=yaw)
        )
        # store yaw to use it for rotate image
        # to make car always bottom center of image
        self.yaw = yaw

        sensor_actor = self.world.spawn_actor(sensor_bp, sensor_transform,
                                              attach_to=vehicle)
        self.actor_list.append(sensor_actor)
        return sensor_actor

    def spawn_collision_sensor(self, vehicle, location):
        col_sensor = self.bp_lib.find("sensor.other.collision")
        col_sensor = self.world.spawn_actor(col_sensor, carla.Transform(),
                                            attach_to=vehicle)
        self.actor_list.append(col_sensor)
        return col_sensor

    def spawn_lane_sensor(self, vehicle):
        lane_sensor_bp = self.bp_lib.find('sensor.other.lane_invasion')
        lane_sensor = self.world.spawn_actor(lane_sensor_bp, carla.Transform(),
                                             attach_to=vehicle)
        self.actor_list.append(lane_sensor)
        return lane_sensor

    def clean_world(self):
        """彻底清理所有actor（传感器、车辆），停止传感器监听"""
        if not self.actor_list:
            logger.info("无需要清理的actor")
            return

        logger.info(f"开始清理 {len(self.actor_list)} 个actor...")
        for actor in self.actor_list:
            try:
                if actor is None:
                    continue
                # 停止传感器监听（针对camera/collision等传感器）
                if hasattr(actor, 'is_listening') and actor.is_listening:
                    actor.stop()
                    logger.debug(f"停止传感器监听: {actor.type_id}")
                # 销毁actor
                if actor.is_alive:
                    actor.destroy()
                    logger.debug(f"销毁actor: {actor.type_id}")
            except Exception as e:
                logger.error(f"清理actor {actor.id if actor else '未知'} 失败: {e}")
        # 清空列表
        self.actor_list = []
        self.spawn_points = []
        logger.info("Actor清理完成")
