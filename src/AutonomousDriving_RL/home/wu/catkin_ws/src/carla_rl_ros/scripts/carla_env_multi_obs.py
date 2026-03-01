# carla_env/carla_env_multi_obs.py
"""
CARLA 强化学习环境（4D 观测增强版）
- 观测: [x, y, vx, vy]
- 动作: [throttle, steer, brake]
- 新增: 车道保持奖励、合理速度区间、轨迹日志、参数化配置、抗崩溃机制
- 【本次更新】新增 get_forward_waypoint() 用于高层导航
"""

import carla
import numpy as np
import random
import time
import os
import json
from gymnasium import Env, spaces

VEHICLE_ID_FILE = ".last_vehicle_id.json"


class CarlaEnvMultiObs(Env):
    def __init__(
        self,
        keep_alive_after_exit=False,
        log_trajectory=True,
        trajectory_log_file="trajectory.csv",
        target_speed=8.0,          # 目标速度 (m/s)
        max_episode_steps=1000,    # 最大步数
        map_name=None,             # 指定地图（如 'Town10HD'）
        spawn_point_index=0,       # spawn 点索引
        random_spawn=False,        # 是否随机 spawn
        reward_weights=None        # 奖励权重配置
    ):
        super().__init__()
        self.client = None
        self.world = None
        self.vehicle = None
        self._current_vehicle_id = None
        self.frame_count = 0
        self.max_episode_steps = max_episode_steps
        self.spectator = None
        self.keep_alive = keep_alive_after_exit
        self.log_trajectory = log_trajectory
        self.trajectory_log_file = trajectory_log_file
        self.trajectory_data = []
        self._collision_sensor = None
        self._collision_hist = []

        # 奖励配置
        self.target_speed = target_speed
        self.reward_weights = {
            'forward': 0.1,
            'speed_match': 0.5,
            'lane_center': 1.0,
            'steer_smooth': 0.05,
            'brake_penalty': 0.1,
            'collision': -50.0,
            'time_bonus': 0.01
        }
        if reward_weights:
            self.reward_weights.update(reward_weights)

        # 地图与 spawn 配置
        self.map_name = map_name
        self.spawn_point_index = spawn_point_index
        self.random_spawn = random_spawn

        # 固定 4D 观测空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

    def _connect_carla(self, max_retries=3, timeout=10.0):
        for attempt in range(max_retries):
            try:
                print(f"🔄 尝试连接 CARLA 服务器 (第 {attempt + 1} 次)...")
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(timeout)
                self.world = self.client.get_world()
                if self.map_name and self.map_name not in self.world.get_map().name:
                    print(f"🔄 加载指定地图: {self.map_name}")
                    self.world = self.client.load_world(self.map_name)
                print(f"✅ 成功连接！地图: {self.world.get_map().name}")
                return True
            except Exception as e:
                print(f"⚠️ 连接失败: {e}")
                time.sleep(2)
        raise RuntimeError("❌ 无法连接 CARLA，请确保已启动 `CarlaUE4.sh`")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._connect_carla()
        self._destroy_last_run_vehicle()
        self.spawn_vehicle()

        # 初始化传感器
        self._collision_hist.clear()
        if self._collision_sensor:
            self._collision_sensor.destroy()
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self._collision_sensor.listen(lambda event: self._collision_hist.append(event))

        # 稳定物理
        for _ in range(5):
            self.world.tick()
            time.sleep(0.05)

        # 设置视角
        self.spectator = self.world.get_spectator()
        self._update_spectator_view()

        # 重置状态
        self.trajectory_data = []
        self.frame_count = 0

        obs = self.get_observation()
        return obs, {}

    def _destroy_last_run_vehicle(self):
        if not os.path.exists(VEHICLE_ID_FILE):
            return
        try:
            with open(VEHICLE_ID_FILE, 'r') as f:
                data = json.load(f)
                last_id = data.get("vehicle_id")
            if isinstance(last_id, int):
                self.client.apply_batch_sync([carla.command.DestroyActor(last_id)], do_tick=True)
        except Exception:
            pass
        try:
            os.remove(VEHICLE_ID_FILE)
        except OSError:
            pass

    def spawn_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        if not vehicle_bp or not vehicle_bp.has_attribute('number_of_wheels'):
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        if vehicle_bp.has_attribute('color'):
            color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
            vehicle_bp.set_attribute('color', color)

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("❌ 地图无可用 spawn 点！")

        if self.random_spawn:
            spawn_transform = random.choice(spawn_points)
        elif self.spawn_point_index < len(spawn_points):
            spawn_transform = spawn_points[self.spawn_point_index]
        else:
            spawn_transform = spawn_points[0]

        # 尝试生成
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
        if self.vehicle is None:
            # 备用：抬高 Z 轴
            sp = spawn_transform
            safe_sp = carla.Transform(
                carla.Location(x=sp.location.x, y=sp.location.y, z=max(sp.location.z, 0.0) + 0.5),
                sp.rotation
            )
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, safe_sp)

        if self.vehicle is None:
            raise RuntimeError("❌ 所有 spawn 方式均失败！")

        self._current_vehicle_id = self.vehicle.id
        loc = self.vehicle.get_location()
        print(f"✅ 车辆生成成功 | ID={self._current_vehicle_id} | ({loc.x:.1f}, {loc.y:.1f})")

        try:
            with open(VEHICLE_ID_FILE, 'w') as f:
                json.dump({"vehicle_id": self._current_vehicle_id}, f)
        except Exception as e:
            print(f"⚠️ 保存车辆ID失败: {e}")

    def _update_spectator_view(self):
        if not (self.vehicle and self.spectator):
            return
        try:
            v_transform = self.vehicle.get_transform()
            offset = carla.Location(x=-6.0, y=0.0, z=2.5)
            camera_loc = v_transform.transform(offset)
            rot = carla.Rotation(pitch=-15.0, yaw=v_transform.rotation.yaw)
            self.spectator.set_transform(carla.Transform(camera_loc, rot))
        except Exception:
            pass

    def get_observation(self):
        if not self.vehicle or not self.vehicle.is_alive:
            return np.zeros(4, dtype=np.float32)
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        # 防止 NaN
        x = float(loc.x) if np.isfinite(loc.x) else 0.0
        y = float(loc.y) if np.isfinite(loc.y) else 0.0
        vx = float(vel.x) if np.isfinite(vel.x) else 0.0
        vy = float(vel.y) if np.isfinite(vel.y) else 0.0
        return np.array([x, y, vx, vy], dtype=np.float32)

    def _get_lane_offset(self):
        """计算到最近车道中心的距离（仅用于奖励，不放入 obs）"""
        try:
            waypoint = self.world.get_map().get_waypoint(
                self.vehicle.get_location(), project_to_road=True
            )
            return self.vehicle.get_location().distance(waypoint.transform.location)
        except:
            return 5.0  # 默认远离车道

    def _compute_reward(self, speed, lane_offset, action):
        w = self.reward_weights
        reward = 0.0

        # 前进奖励
        if speed > 0.1:
            reward += w['forward'] * speed

        # 速度匹配
        speed_diff = abs(speed - self.target_speed)
        if speed_diff < 1.0:
            reward += w['speed_match']
        else:
            reward -= speed_diff * 0.05

        # 车道中心奖励（即使 4D 也鼓励 stay in lane）
        if lane_offset < 1.0:
            reward += w['lane_center'] * (1.0 - lane_offset)
        else:
            reward -= (lane_offset - 1.0) * 0.5

        # 控制平滑
        _, steer, brake = action
        reward -= w['steer_smooth'] * abs(steer)
        reward -= w['brake_penalty'] * brake

        # 时间奖励（鼓励存活）
        reward += w['time_bonus']

        return reward

    def step(self, action):
        # 安全钳位
        throttle = np.clip(action[0], 0.0, 1.0)
        steer = np.clip(action[1], -1.0, 1.0)
        brake = np.clip(action[2], 0.0, 1.0)
        control = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        self.vehicle.apply_control(control)
        self.world.tick()
        self.frame_count += 1
        self._update_spectator_view()

        # 车辆死亡
        if not self.vehicle or not self.vehicle.is_alive:
            obs = np.zeros(4, dtype=np.float32)
            return obs, self.reward_weights['collision'], True, False, {}

        # 状态
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(max(0.0, velocity.x**2 + velocity.y**2))
        lane_offset = self._get_lane_offset()
        reward = self._compute_reward(speed, lane_offset, [throttle, steer, brake])

        # 终止条件
        terminated = len(self._collision_hist) > 0
        if terminated:
            reward = self.reward_weights['collision']

        truncated = self.frame_count >= self.max_episode_steps

        # 记录轨迹
        if self.log_trajectory:
            loc = self.vehicle.get_location()
            self.trajectory_data.append((float(loc.x), float(loc.y), float(speed)))

        obs = self.get_observation()
        return obs, reward, terminated, truncated, {
            "speed": speed,
            "lane_offset": lane_offset,
            "collision": terminated
        }

    # ================================
    # 【新增功能】用于高层导航
    # ================================

    def get_vehicle_transform(self):
        """安全获取车辆当前位姿（Transform）"""
        if not self.vehicle or not self.vehicle.is_alive:
            return None
        try:
            return self.vehicle.get_transform()
        except:
            return None

    def get_forward_waypoint(self, distance=3.0):
        """
        获取车辆前方指定距离的车道中心点（世界坐标）
        :param distance: 前瞻距离（米），建议 2.0~5.0
        :return: carla.Location 对象，若失败返回 None
        """
        try:
            vehicle_tf = self.get_vehicle_transform()
            if vehicle_tf is None:
                return None
            # 沿车头方向前进
            forward = vehicle_tf.get_forward_vector()
            target_loc = vehicle_tf.location + carla.Location(
                x=forward.x * distance,
                y=forward.y * distance,
                z=0.0
            )
            # 投影到最近可行驶车道中心
            waypoint = self.world.get_map().get_waypoint(
                target_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            return waypoint.transform.location if waypoint else None
        except Exception as e:
            print(f"⚠️ get_forward_waypoint 失败: {e}")
            return None

    def close(self):
        # 保存轨迹
        if self.log_trajectory and self.trajectory_data:
            try:
                with open(self.trajectory_log_file, 'w') as f:
                    f.write("x,y,speed\n")
                    for x, y, speed in self.trajectory_data:
                        f.write(f"{x:.3f},{y:.3f},{speed:.3f}\n")
                print(f"📊 轨迹已保存至: {self.trajectory_log_file}")
            except Exception as e:
                print(f"⚠️ 轨迹保存失败: {e}")

        # 清理
        if self._collision_sensor:
            self._collision_sensor.destroy()
        if not self.keep_alive and self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
        elif self.keep_alive:
            print("ℹ️ 车辆已保留（下次运行将自动清理）")