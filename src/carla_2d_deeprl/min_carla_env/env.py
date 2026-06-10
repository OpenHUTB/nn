import cv2
import gym
import time
import math
import carla
import imutils
import numpy as np
import logging
from typing import Optional

from min_carla_env.matrix_world import MatrixWorld
from min_carla_env.config import (
    REWARD_CONFIG,
    ACTIONS,
    CONFIG,  # re-exported for backward compatibility
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Carla 客户端重连工具
def reconnect_carla_client(original_client, host='localhost', port=2000,
                           timeout=30, retries=3) -> Optional[carla.Client]:
    """Carla客户端重连工具"""
    for retry in range(retries):
        try:
            # 关闭原有连接（如果存在）
            if original_client:
                try:
                    original_client.close()
                except Exception:
                    pass
            # 新建连接
            client = carla.Client(host, port)
            client.set_timeout(timeout)
            # 验证连接（获取服务器版本）
            client.get_server_version()
            logger.info(f"Carla客户端重连成功 (重试 {retry + 1}/{retries})")
            return client
        except Exception as e:
            logger.warning(f"Carla客户端重连失败 (重试 {retry + 1}/{retries}): {e}")
            time.sleep(2)
    logger.error(f"Carla客户端重连失败，已重试 {retries} 次")
    return None


class CarlaEnv(gym.Env):
    """Simple gym wrapper for Carla.
    Unfortunately it only uses the gym environment interface.
    Its not that much compatible with gym."""

    def __init__(self, client, config, world_config=None, reward_config=None,
                 debug=False, demo=False):
        self.debug = debug
        self.done = False
        self.rgb_data = None
        self.semantic_data = None
        self.steps = 0
        self.stuck_count = 0
        self.started = False
        self.collision_hist = []
        self.crossed_lane_hist = []
        self.config = config
        self.max_step = config.get("max_step", 90000)
        self.demo = demo
        self.client = client  # 保存客户端引用
        # 奖励系数配置（可单独传入微调）
        self.reward_config = reward_config if reward_config is not None else REWARD_CONFIG
        self.mw = None  # 初始化MatrixWorld为None
        self.world = None
        self.vehicle = None
        self.rgb_sensor = None
        self.semantic_sensor = None
        self.col_sensor = None
        self.lane_sensor = None
        self.measurements = {
            "kmh": 0.0,
            "prev_loc": None
        }
        self.hist_wp = None

        try:
            # 初始化MatrixWorld（增加重连机制）
            # 将观测尺寸从 config 注入到 world_config，确保传感器分辨率一致
            wcfg = dict(world_config) if world_config else {}
            wcfg.setdefault("im_width", self.config.get("width", 480))
            wcfg.setdefault("im_height", self.config.get("height", 480))
            self.mw = MatrixWorld(self.client, **wcfg)
            self.world = self.mw.world
            # 生成actor
            self.spawn_actors()
        except Exception as e:
            logger.error(f"CarlaEnv初始化失败: {e}")
            # 初始化失败时清理资源
            if self.mw:
                self.mw.clean_world()
            # 尝试重连客户端
            new_client = reconnect_carla_client(self.client)
            if new_client:
                self.client = new_client
                self.mw = MatrixWorld(self.client, **wcfg)
                self.world = self.mw.world
                self.spawn_actors()
                logger.info("重连后初始化成功")
            else:
                raise RuntimeError("CarlaEnv初始化失败，且重连客户端失败")

    def spawn_actors(self):
        """Spawns agent car and sensors."""
        self.vehicle = self.mw.spawn_vehicle()

        self.rgb_sensor = self.mw.spawn_sensor('sensor.camera.rgb',
                                               self.vehicle,
                                               carla.Location(x=2.5, z=0.7))
        self.rgb_sensor.listen(lambda image: self.process_img(image))
        self.semantic_sensor = self.mw.spawn_sensor(
            'sensor.camera.semantic_segmentation',
            self.vehicle, carla.Location(x=2.5, z=0.7))
        self.semantic_sensor.listen(lambda image: self.process_semantic(image))
        self.col_sensor = self.mw.spawn_collision_sensor(
            self.vehicle, carla.Location(x=2.5, z=0.7))
        self.col_sensor.listen(lambda event: self.collision_data(event))
        self.lane_sensor = self.mw.spawn_lane_sensor(self.vehicle)
        self.lane_sensor.listen(lambda event: self.lane_data(event))

    def collision_data(self, event):
        self.collision_hist.append(event)

    def lane_data(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        # text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.crossed_lane_hist.extend(lane_types)

    def process_img(self, image):
        """Convert rgb image to array."""
        i = np.array(image.raw_data)
        i2 = i.reshape((self.config['width'], self.config['height'], 4))
        i3 = i2[:, :, :3]
        # rotate to make car bottom center
        i3 = imutils.rotate_bound(i3, self.mw.yaw)
        if self.debug:
            cv2.imwrite("i.png", i3)
        self.rgb_data = i3

    def semantic_mask(self, data):
        """Masks the current semantic data to desired labels
        classes I want to show:
        0: none
        1: road
        2: roadlines
        3: poles
        4: sidewalks
        5: vehicles
        """
        # replace sidewalks with others
        data[data == 1] = 4  # buildings
        data[data == 2] = 4  # fences
        data[data == 3] = 4  # other
        data[data == 4] = 4  # pedesterians
        data[data == 9] = 4  # vegetation
        data[data == 11] = 4  # walls
        data[data == 12] = 4  # TrafficSigns

        data[data == 5] = 3  # change poles
        data[data == 6] = 2  # change roadline
        data[data == 7] = 1  # change road
        data[data == 8] = 4  # change sidewalks
        data[data == 10] = 5  # change vehicles
        return data

    def process_semantic(self, image):
        """Convert a convert semantic image to array."""
        # if not isinstance(image, sensor.Image):
        #     raise ValueError("Argument must be a carla.sensor.Image")
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))

        # rotate to make car bottom center
        array = imutils.rotate_bound(array, self.mw.yaw)

        self.semantic_data = array[:, :, 2].copy()
        self.semantic_data = self.semantic_mask(self.semantic_data)
        # self.semantic_data = self.semantic_mask_simple(self.semantic_data)
        if self.debug:
            cc_semantic_data = self.labels_to_cityscapes_palette(
                self.semantic_data)
            cv2.imwrite("s.png", cc_semantic_data)

    def labels_to_cityscapes_palette(self, array):
        """
        Convert an image containing CARLA semantic segmentation labels to
        Cityscapes palette.
        """
        classes = {
            0: [0, 0, 0],  # None
            1: [70, 70, 70],  # Buildings
            2: [190, 153, 153],  # Fences
            3: [72, 0, 90],  # Other
            4: [220, 20, 60],  # Pedestrians
            5: [153, 153, 153],  # Poles
            6: [157, 234, 50],  # RoadLines
            7: [128, 64, 128],  # Roads
            8: [244, 35, 232],  # Sidewalks
            9: [107, 142, 35],  # Vegetation
            10: [0, 0, 255],  # Vehicles
            11: [102, 102, 156],  # Walls
            12: [220, 220, 0]  # TrafficSigns
        }
        result = np.zeros((array.shape[0], array.shape[1], 3))
        for key, value in classes.items():
            result[np.where(array == key)] = value
        return result

    def get_measurements(self):
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        measurements = {
            "kmh": kmh,
            "prev_loc": self.vehicle.get_transform().location
        }
        return measurements

    def reset(self):
        self.done = False
        self.steps = 0
        self.started = False
        self.rgb_data = None
        self.semantic_data = None
        self.collision_hist = []
        self.crossed_lane_hist = []
        self.hist_wp = None
        self.stuck_count = 0
        self.update_spectator_follow()  # 重置时立刻设置视角，不等待

        try:
            self.mw.clean_world()
            # 重新初始化 MatrixWorld（复用初始化时的配置，而非 __dict__）
            world_config = {
                "im_width": self.config["width"],
                "im_height": self.config["height"],
                "render": self.config["render"],
                "weather": self.mw.weather,
                "fast": getattr(self.mw, 'fast', False),
                "town": self.mw.world.get_map().name.split('/')[-1]  # 保留当前地图
            }
            # 重试生成actor
            spawn_success = False
            for _ in range(5):
                try:
                    self.mw = MatrixWorld(self.client, **world_config)  # 重新初始化
                    self.world = self.mw.world
                    self.spawn_actors()
                    spawn_success = True
                    logger.info("Actor生成成功")
                    break
                except Exception as e:
                    logger.warning(f"Actor生成失败 (重试 {_ + 1}/5): {e}")
                    self.mw.clean_world()
                    time.sleep(1)

            if not spawn_success:
                logger.warning("Actor生成多次失败，尝试重连Carla客户端")
                new_client = reconnect_carla_client(self.client)
                if new_client:
                    self.client = new_client
                    self.mw = MatrixWorld(self.client, **world_config)
                    self.world = self.mw.world
                    self.spawn_actors()
                    logger.info("重连后Actor生成成功")
                else:
                    raise RuntimeError("Actor生成失败，且重连客户端失败")

            self.measurements = {"kmh": 0.0, "prev_loc": None}
            time.sleep(1)
            self.update_spectator_follow()  # 车辆已生成，设置俯视视角
            return self.semantic_data
        except Exception as e:
            logger.error(f"Reset失败: {e}")
            self.mw.clean_world()
            raise
        finally:
            logger.debug("Reset流程完成，资源已清理")

    def __euclid_dist(self, loc1, loc2):
        """Calc euclid distance of carla Locations."""
        dist = math.sqrt(
            (loc1.x - loc2.x) ** 2 +
            (loc1.y - loc2.y) ** 2 +
            (loc1.z - loc2.z) ** 2
        )
        return dist

    def simple_loc_reward(self, map: carla.Map, location: carla.Location):
        """Calculates lane-center reward for given location.
        调优版本：使用可配置的奖励系数，裁剪惩罚上限，训练更稳定。"""
        reward = 0.0
        rc = self.reward_config
        wp = map.get_waypoint(location, carla.LaneType.Driving)
        wp_location = wp.transform.location
        dist = self.__euclid_dist(wp_location, location)
        threshold = rc["lane_center_threshold"]
        if dist < threshold:
            reward += rc["lane_center_reward"]
        else:
            # 缩放后的指数惩罚，并裁剪上限，防止梯度爆炸
            penalty = rc["dist_penalty_scale"] * np.exp(dist)
            penalty = min(penalty, rc["dist_penalty_clip"])
            reward -= penalty

        return reward

    def is_stuck(self, location: carla.Location):
        prev_loc = self.measurements["prev_loc"]
        if prev_loc is not None and self.started:
            dist = self.__euclid_dist(prev_loc, location)
            if dist <= 0.05:
                return True
        return False

    def step(self, action):
        """Apply action, calculate reward, return observation."""
        # interpret actions
        action = ACTIONS[int(action)]
        steer = float(np.clip(action[1], -1, 1))
        reverse = False
        hand_brake = False

        measurements = self.get_measurements()
        kmh = measurements["kmh"]
        if kmh >= 1 and not self.started:
            self.started = True
        else:
            self.collision_hist = []

        # make the car always in stable velocity
        brake = 0.0
        throttle = 0.3
        if kmh >= 20:
            brake = 0.2
            throttle = 0.0
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, brake=brake, steer=steer,
            reverse=reverse, hand_brake=hand_brake))

        # calculate reward
        rc = self.reward_config
        reward = 0.0
        vehicle_location = self.vehicle.get_transform().location

        # 1) 车道中心奖励（核心）
        reward += self.simple_loc_reward(self.mw.world.get_map(), vehicle_location)

        # 2) 速度奖励（新增，鼓励维持目标车速，让训练更稳定）
        speed_reward = rc["speed_reward_scale"] * kmh
        reward += speed_reward

        # count stuck to be able to stop running
        self.steps += 1
        if self.is_stuck(vehicle_location):
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        self.measurements = measurements
        if self.steps >= self.max_step:
            self.done = True

        current_w = self.mw.world.get_map().get_waypoint(vehicle_location)
        if reward <= -rc["dist_penalty_clip"]:  # 使用配置的惩罚裁剪值
            self.done = True
        if len(self.collision_hist) != 0:  # stop on collision
            self.done = True
            reward *= rc["collision_penalty_mult"]
        if len(self.crossed_lane_hist) != 0:  # stop on crossed lane
            for lane_marking in self.crossed_lane_hist:
                if lane_marking == carla.LaneMarkingType.Solid or \
                        lane_marking == carla.LaneMarkingType.NONE:
                    self.done = True
                    reward *= rc["lane_violation_penalty_mult"]
                    break
            self.crossed_lane_hist = []
        if current_w.lane_type == carla.LaneType.Sidewalk:  # stop on out of road
            self.done = True
            reward *= rc["sidewalk_penalty_mult"]
        if self.stuck_count > self.config.get("stuck_max_count", 20):  # stop on stuck
            self.done = True
            reward -= rc["stuck_penalty"]

        if self.demo and self.stuck_count < 20:
            self.done = False

        self.update_spectator_follow()  # 每一步都自动跟随，丝滑不卡顿

        return self.semantic_data, reward, self.done, {}
        # return (self.rgb_data, self.semantic_data), reward, self.done, {}

    def close(self):
        """手动关闭环境，释放所有资源"""
        logger.info("开始关闭CarlaEnv环境...")
        try:
            # 停止传感器监听
            if self.rgb_sensor:
                self.rgb_sensor.stop()
            if self.semantic_sensor:
                self.semantic_sensor.stop()
            if self.col_sensor:
                self.col_sensor.stop()
            if self.lane_sensor:
                self.lane_sensor.stop()
            # 再清理MatrixWorld资源
            if self.mw:
                self.mw.clean_world()
            logger.info("CarlaEnv环境关闭成功")
        except Exception as e:
            logger.error(f"关闭CarlaEnv环境失败: {e}")

    def update_spectator_follow(self):
        """
        【无卡顿、永久、车辆正上方俯视视角】
        一启动就固定在车辆上方俯视，全程自动跟随，录视频完美
        """
        if not self.vehicle:
            return
        spectator = self.world.get_spectator()
        trans = self.vehicle.get_transform()

        # 车辆正上方 18 米，pitch=-90 纯俯视，yaw 跟随车辆朝向
        location = carla.Location(
            x=trans.location.x,
            y=trans.location.y,
            z=trans.location.z + 18.0
        )
        rotation = carla.Rotation(pitch=-90, yaw=trans.rotation.yaw, roll=0)

        spectator.set_transform(carla.Transform(location, rotation))
