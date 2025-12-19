# carla_env/carla_env_multi_obs.py

"""
CARLA 强化学习环境封装（支持多维观测）
- 支持两种观测模式：旧版 4 维（位置+速度） / 新版 9 维（增强感知）
- 集成碰撞检测、轨迹记录、自动车辆清理、视角跟随等功能
- 适配 Gymnasium 接口，可直接用于 Stable-Baselines3 等 RL 库
"""

import carla
import numpy as np
import random
import time
import os
import json
from gymnasium import Env, spaces

# 全局常量：用于跨运行清理上一次残留车辆
VEHICLE_ID_FILE = ".last_vehicle_id.json"
# 轨迹日志文件名（x, y, speed）
TRAJECTORY_LOG_FILE = "trajectory.csv"


class CarlaEnvMultiObs(Env):
    """
    基于 CARLA 的自动驾驶强化学习环境
    """

    def __init__(self, keep_alive_after_exit=True, log_trajectory=True, legacy_mode=False):
        """
        初始化环境

        参数:
            keep_alive_after_exit (bool):
                是否在环境关闭后保留车辆（便于手动观察或调试）
            log_trajectory (bool):
                是否记录车辆轨迹到 CSV 文件
            legacy_mode (bool):
                是否使用旧版 4 维观测空间（[x, y, vx, vy]）
                默认 False → 使用新版 9 维观测（含车道、障碍物、红灯等）
        """
        super(CarlaEnvMultiObs, self).__init__()

        # CARLA 客户端与世界对象
        self.client = None
        self.world = None
        self.vehicle = None
        self._current_vehicle_id = None

        # 训练控制
        self.frame_count = 0
        self.max_frames = 1000  # 单轮最大步数（防止无限运行）

        # 视角控制
        self.spectator = None

        # 行为控制标志
        self.keep_alive = keep_alive_after_exit
        self.log_trajectory = log_trajectory
        self.trajectory_data = []  # 存储 (x, y, speed) 轨迹点

        # 碰撞传感器
        self._collision_sensor = None
        self._collision_hist = []  # 存储碰撞事件

        # 观测模式开关
        self.legacy_mode = legacy_mode

        # 根据模式设置观测空间
        if self.legacy_mode:
            # 旧版：仅位置和速度（无方向、无环境感知）
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            )
        else:
            # 新版：9 维标准化观测
            # [speed_norm, lane_offset_norm, curvature_norm, obstacle_dist_norm,
            #  is_red_light, vx_norm, vy_norm, sin(yaw), cos(yaw)]
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(9,), dtype=np.float32
            )

        # 动作空间：[throttle, steer, brake]
        # throttle ∈ [0, 1], steer ∈ [-1, 1], brake ∈ [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

    def _connect_carla(self, max_retries=3):
        """
        尝试连接本地 CARLA 服务器（localhost:2000）
        """
        for attempt in range(max_retries):
            try:
                print(f"🔄 尝试连接 CARLA 服务器 (第 {attempt + 1} 次)...")
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(10.0)  # 超时 10 秒
                self.world = self.client.get_world()
                if self.world is not None:
                    print(f"✅ 成功连接到 CARLA！地图: {self.world.get_map().name}")
                    return True
            except Exception as e:
                print(f"⚠️ 连接失败: {e}")
                time.sleep(2)
        raise RuntimeError("❌ 无法连接到 CARLA 服务器，请确保 CARLA 已启动！")

    def reset(self, seed=None, options=None):
        """
        重置环境：清理旧车、生成新车、初始化传感器
        """
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 连接 CARLA 并销毁上次残留车辆
        self._connect_carla()
        self._destroy_last_run_vehicle()

        # 生成新车辆
        self.spawn_vehicle()

        # 清空碰撞历史并创建新碰撞传感器
        self._collision_hist.clear()
        if self._collision_sensor is not None:
            self._collision_sensor.destroy()
            self._collision_sensor = None

        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self._collision_sensor.listen(lambda event: self._collision_hist.append(event))

        # 等待几帧让物理稳定
        for _ in range(5):
            self.world.tick()
            time.sleep(0.05)

        # 设置第三人称视角跟随车辆
        self.spectator = self.world.get_spectator()
        self._update_spectator_view()

        # 重置轨迹与帧计数
        self.trajectory_data = []
        self.frame_count = 0

        # 返回初始观测
        obs = self.get_observation()
        return obs, {}

    def _destroy_last_run_vehicle(self):
        """
        从 .last_vehicle_id.json 读取上次车辆 ID 并尝试销毁
        避免多次运行导致车辆堆积
        """
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
        """
        在地图中生成一辆车（优先 Tesla Model 3）
        - Town10HD_Opt 使用固定 spawn 点
        - 其他地图使用第一个可用 spawn 点
        - 若失败则遍历所有点尝试
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        if not vehicle_bp or not vehicle_bp.has_attribute('number_of_wheels'):
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        if vehicle_bp.has_attribute('color'):
            color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
            vehicle_bp.set_attribute('color', color)

        map_name = self.world.get_map().name.lower()
        spawn_transform = None

        # Town10HD 特定 spawn 点（避免出生在空中/水里）
        if 'town10' in map_name:
            spawn_transform = carla.Transform(
                carla.Location(x=100.0, y=130.0, z=0.3),
                carla.Rotation(yaw=180.0)
            )
        else:
            spawn_points = self.world.get_map().get_spawn_points()
            if spawn_points:
                spawn_transform = spawn_points[0]
            else:
                spawn_transform = carla.Transform(carla.Location(x=0, y=0, z=1.0), carla.Rotation())

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
        if self.vehicle is None:
            print("⚠️ 主 spawn 点失败，尝试遍历所有可用点...")
            all_spawn_points = self.world.get_map().get_spawn_points()
            random.shuffle(all_spawn_points)
            for sp in all_spawn_points:
                safe_z = max(sp.location.z, 0.0) + 0.3  # 抬高一点避免穿模
                safe_sp = carla.Transform(
                    carla.Location(x=sp.location.x, y=sp.location.y, z=safe_z),
                    sp.rotation
                )
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, safe_sp)
                if self.vehicle is not None:
                    break

        if self.vehicle is None:
            raise RuntimeError("❌ 所有 spawn 点均无法生成车辆！请检查地图或 CARLA 状态。")

        self._current_vehicle_id = self.vehicle.id
        loc = self.vehicle.get_location()
        print(
            f"✅ 车辆生成成功: {self.vehicle.type_id} | ID={self._current_vehicle_id} | 位置: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

        # 保存 ID 供下次清理
        try:
            with open(VEHICLE_ID_FILE, 'w') as f:
                json.dump({"vehicle_id": self._current_vehicle_id}, f)
        except Exception as e:
            print(f"⚠️ 保存车辆ID失败: {e}")

    def _update_spectator_view(self):
        """
        更新第三人称摄像机视角，跟随车辆后上方
        """
        if not (self.vehicle and self.spectator):
            return
        try:
            v_transform = self.vehicle.get_transform()
            offset = carla.Location(x=-5.0, y=1.0, z=2.2)  # 相对偏移
            camera_loc = v_transform.transform(offset)
            spectator_rot = carla.Rotation(pitch=-10.0, yaw=v_transform.rotation.yaw, roll=0.0)
            self.spectator.set_transform(carla.Transform(camera_loc, spectator_rot))
        except Exception:
            pass

    def _traffic_light_ahead(self, dist=15.0):
        """
        检测前方 dist 米内是否有红灯
        利用向量点积判断是否在车辆前方
        """
        if not self.vehicle:
            return False
        lights = self.world.get_actors().filter('traffic.traffic_light*')
        vehicle_transform = self.vehicle.get_transform()
        forward = vehicle_transform.get_forward_vector()
        for light in lights:
            delta = light.get_transform().location - vehicle_transform.location
            dot = delta.x * forward.x + delta.y * forward.y  # 点积
            if 0 < dot < dist and delta.distance(vehicle_transform.location) < dist:
                if light.state == carla.TrafficLightState.Red:
                    return True
        return False

    def _log_trajectory(self, x, y, speed):
        """
        记录当前帧的轨迹点（若启用）
        """
        if self.log_trajectory:
            self.trajectory_data.append((x, y, speed))

    def get_observation(self):
        """
        获取当前环境观测值

        返回:
            np.ndarray: 形状为 (4,) 或 (9,) 的浮点数组
        """
        if not self.vehicle or not self.vehicle.is_alive:
            dim = 4 if self.legacy_mode else 9
            return np.zeros(dim, dtype=np.float32)

        if self.legacy_mode:
            # 旧模式：仅位置和速度
            loc = self.vehicle.get_location()
            vel = self.vehicle.get_velocity()
            return np.array([loc.x, loc.y, vel.x, vel.y], dtype=np.float32)

        # 新模式：9 维增强观测
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])  # 2D 速度大小
        forward_vec = transform.get_forward_vector()

        # 车道信息（通过 Waypoint 获取）
        try:
            waypoint = self.world.get_map().get_waypoint(transform.location, project_to_road=True)
            lane_offset = transform.location.distance(waypoint.transform.location)  # 到车道中心距离
            next_wp_list = waypoint.next(5.0)
            next_wp = next_wp_list[0] if next_wp_list else waypoint
            # 曲率：下一 waypoint 与当前航向的偏转角（反映弯道程度）
            curvature = abs(next_wp.transform.rotation.yaw - waypoint.transform.rotation.yaw) / 5.0
        except:
            lane_offset, curvature = 5.0, 0.5  # 异常时设为最差值

        # 障碍物检测：向前发射射线（ray-cast）
        obstacle_dist = 50.0
        try:
            start = transform.location + carla.Location(z=0.5)
            end = start + forward_vec * 20.0
            hits = self.world.cast_ray(start, end)
            if hits:
                obstacle_dist = min(h.distance for h in hits)
        except:
            pass

        # 红灯检测
        is_red_light = self._traffic_light_ahead()

        # 构建 9 维观测（全部归一化到 [-1, 1] 或 [0, 1]）
        obs = np.array([
            speed / 30.0,  # 速度归一化（假设 max=30 m/s）
            min(lane_offset, 3.0) / 3.0,  # 车道偏移（最大 3 米）
            min(curvature, 10.0) / 10.0,  # 曲率（最大 10 度/米）
            min(obstacle_dist, 50.0) / 50.0,  # 障碍物距离（最大 50 米）
            float(is_red_light),  # 红灯（0 或 1）
            np.clip(velocity.x / 30.0, -1, 1),  # vx 归一化
            np.clip(velocity.y / 30.0, -1, 1),  # vy 归一化
            np.sin(np.radians(transform.rotation.yaw)),  # 航向角正弦
            np.cos(np.radians(transform.rotation.yaw))  # 航向角余弦
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    def _compute_reward(self, speed, lane_offset, obstacle_dist, is_red_light, action):
        """
        计算每一步的奖励（当前实现较基础，可优化）
        """
        reward = 0.0
        target_speed = 10.0
        reward += -abs(speed - target_speed) * 0.1  # 速度惩罚

        if lane_offset < 1.0:
            reward += (1.0 - lane_offset) * 0.5  # 车道内奖励
        else:
            reward -= 1.0  # 车道外惩罚

        if obstacle_dist < 5.0:
            reward -= (5.0 - obstacle_dist) * 2.0
        if obstacle_dist < 2.0:
            reward -= 10.0  # 极近重罚

        if is_red_light and speed > 1.0:
            reward -= 5.0  # 闯红灯惩罚

        throttle, steer, brake = action
        reward -= (abs(steer) * 0.1 + abs(brake) * 0.05)  # 控制平滑性

        return reward

    def step(self, action):
        """
        执行一步动作，返回 (obs, reward, terminated, truncated, info)
        """
        throttle, steer, brake = action
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        )
        self.vehicle.apply_control(control)
        self.world.tick()
        self.frame_count += 1
        self._update_spectator_view()

        # 车辆死亡（如被销毁）
        if not self.vehicle or not self.vehicle.is_alive:
            dim = 4 if self.legacy_mode else 9
            obs = np.zeros(dim, dtype=np.float32)
            return obs, -10.0, True, False, {}

        # 获取状态
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])

        try:
            waypoint = self.world.get_map().get_waypoint(transform.location, project_to_road=True)
            lane_offset = transform.location.distance(waypoint.transform.location)
        except:
            lane_offset = 5.0

        obstacle_dist = 50.0
        try:
            forward_vec = transform.get_forward_vector()
            start = transform.location + carla.Location(z=0.5)
            end = start + forward_vec * 20.0
            hits = self.world.cast_ray(start, end)
            if hits:
                obstacle_dist = min(h.distance for h in hits)
        except:
            pass

        is_red_light = self._traffic_light_ahead()
        reward = self._compute_reward(speed, lane_offset, obstacle_dist, is_red_light, action)

        # 终止条件：碰撞
        terminated = len(self._collision_hist) > 0
        if terminated:
            reward -= 50.0

        # 截断条件：超时
        truncated = self.frame_count >= self.max_frames

        # 记录轨迹
        self._log_trajectory(transform.location.x, transform.location.y, speed)

        # 获取新观测
        obs = self.get_observation()
        return obs, reward, terminated, truncated, {}

    def close(self):
        """
        关闭环境：保存轨迹、清理传感器和车辆
        """
        # 保存轨迹
        if self.log_trajectory and self.trajectory_data:
            try:
                with open(TRAJECTORY_LOG_FILE, 'w') as f:
                    f.write("x,y,speed\n")
                    for x, y, speed in self.trajectory_data:
                        f.write(f"{x:.3f},{y:.3f},{speed:.3f}\n")
                print(f"📊 轨迹已保存至: {TRAJECTORY_LOG_FILE}")
            except Exception as e:
                print(f"⚠️ 轨迹保存失败: {e}")

        # 销毁传感器
        if self._collision_sensor is not None:
            self._collision_sensor.destroy()
            self._collision_sensor = None

        # 车辆处理
        if self.keep_alive:
            print("ℹ️ 车辆已保留（ID已记录，下次运行时将自动清理）")
            if self.vehicle:
                self.vehicle.apply_control(carla.VehicleControl())
                for i in range(30):
                    self.world.tick()
                    self._update_spectator_view()
                    time.sleep(0.1)
                print("✅ 现在你可以自由操作 CARLA 视角（按 F1~F4）！")
        else:
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.destroy()
