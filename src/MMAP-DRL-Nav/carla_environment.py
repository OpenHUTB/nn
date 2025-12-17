import carla
import numpy as np
import gym
import time
import socket
import random

class CarlaEnvironment(gym.Env):
    def __init__(self):
        super(CarlaEnvironment, self).__init__()
        # 初始化CARLA客户端
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)

        # ========== 1. 端口检测 + 重试连接 ==========
        def is_port_used(port):
            """检测端口是否被占用"""
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.connect(('localhost', port))
                return True
            except:
                return False
            finally:
                s.close()

        if not is_port_used(2000):
            raise RuntimeError("2000端口未被占用，请先启动CARLA模拟器")

        max_retry = 3
        retry_count = 0
        while retry_count < max_retry:
            try:
                self.world = self.client.get_world()
                break
            except RuntimeError as e:
                retry_count += 1
                print(f"连接失败，重试第{retry_count}次...")
                time.sleep(5)
        else:
            raise RuntimeError("CARLA连接超时（3次重试失败），请检查模拟器是否启动")

        self.blueprint_library = self.world.get_blueprint_library()

        # ========== 同步模式配置 ==========
        self.sync_settings = self.world.get_settings()
        self.sync_settings.synchronous_mode = True
        self.sync_settings.fixed_delta_seconds = 1.0 / 30
        self.sync_settings.no_rendering_mode = False
        self.world.apply_settings(self.sync_settings)

        # ========== NPC配置 ==========
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        self.npc_vehicle_list = []
        self.npc_pedestrian_list = []
        self.hit_vehicle = False
        self.hit_pedestrian = False
        self.hit_static = False  # 碰撞静态物体标记
        self.collision_penalty_applied = False  # 碰撞惩罚是否已执行

        # ========== 1. 红绿灯奖惩配置 ==========
        self.red_light_penalty = -8.0       # 闯红灯一次性重罚
        self.green_light_reward = 10.0      # 绿灯通过高奖励
        self.red_light_stop_reward = 0.3    # 红灯停车奖励（高于前进奖励）
        self.traffic_light_cooldown = 10.0
        self.last_traffic_light_time = 0
        self.traffic_light_trigger_distance = 10.0
        self.traffic_light_reset_distance = 15.0
        self.has_triggered_red = False
        self.has_triggered_green = False

        # ========== 2. 超速奖惩配置 ==========
        self.speed_limit_urban = 30.0       # 城区限速30km/h
        self.over_speed_light_penalty = -1.0 # 轻度超速（30-40km/h）/秒
        self.over_speed_heavy_penalty = -4.0 # 重度超速（>40km/h）/秒
        self.over_speed_cooldown = 1.0
        self.last_over_speed_time = 0

        # ========== 3. 车道偏离奖惩配置（核心调整） ==========
        self.lane_keep_reward = 0.05        # 保持车道内奖励/帧
        self.lane_light_penalty = -0.2      # 轻微偏离单次扣分
        self.lane_heavy_penalty = -1.0      # 严重偏离单次扣分
        self.lane_offset_light = 0.2        # 轻微偏离阈值（米）
        self.lane_offset_heavy = 0.4        # 严重偏离阈值（米）
        self.lane_check_interval = 3.0      # 车道检测间隔：3秒
        self.lane_log_interval = 10.0       # 车道日志输出间隔：10秒
        self.last_lane_check_time = 0       # 上次车道检测时间戳
        self.last_lane_log_time = 0         # 上次车道日志输出时间戳
        self.enable_lane_log = True         # 开启车道偏离日志

        # ========== 4. 碰撞奖惩配置 ==========
        self.collision_vehicle_penalty = -50.0  # 碰撞车辆惩罚
        self.collision_pedestrian_penalty = -100.0 # 碰撞行人惩罚
        self.collision_static_penalty = -15.0    # 碰撞静态物体惩罚

        # 基础配置
        self.spawn_retry_times = 20
        self.spawn_safe_radius = 2.0

        # 动作/观测空间
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
        )

        # 核心对象
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.image_data = None
        self.has_collision = False

        # 视角参数
        self.view_height = 6.0
        self.view_pitch = -15.0
        self.view_distance = 8.0
        self.z_offset = 0.5

    # ========== 红绿灯判定（移除特殊符号） ==========
    def _check_traffic_light(self):
        current_time = time.time()
        if current_time - self.last_traffic_light_time < self.traffic_light_cooldown:
            return 0.0

        if not self.vehicle or not self.vehicle.is_alive:
            return 0.0

        vehicle_loc = self.vehicle.get_transform().location
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        reward = 0.0
        has_near_light = False

        # 获取车辆速度（判断是否停车）
        velocity = self.vehicle.get_velocity()
        speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        is_stopped = speed_m_s < 0.1

        for light in traffic_lights:
            dist = vehicle_loc.distance(light.get_transform().location)
            if dist <= self.traffic_light_trigger_distance:
                has_near_light = True
                light_state = light.state

                # 红灯逻辑：停车奖励 / 闯灯惩罚
                if light_state == carla.TrafficLightState.Red:
                    if not is_stopped and not self.has_triggered_red:
                        reward = self.red_light_penalty
                        print(f"闯红灯！扣分{self.red_light_penalty}")
                        self.has_triggered_red = True
                        self.last_traffic_light_time = current_time
                    elif is_stopped:
                        reward = self.red_light_stop_reward
                    break

                # 绿灯逻辑：通过奖励
                elif light_state == carla.TrafficLightState.Green and not self.has_triggered_green:
                    reward = self.green_light_reward
                    print(f"绿灯合规通过！加分{self.green_light_reward}")
                    self.has_triggered_green = True
                    self.last_traffic_light_time = current_time
                    break

            elif dist > self.traffic_light_reset_distance:
                self.has_triggered_red = False
                self.has_triggered_green = False

        if not has_near_light:
            self.has_triggered_red = False
            self.has_triggered_green = False

        return reward

    # ========== 超速检测（无日志） ==========
    def _check_over_speed(self):
        current_time = time.time()
        if current_time - self.last_over_speed_time < self.over_speed_cooldown:
            return 0.0

        if not self.vehicle or not self.vehicle.is_alive:
            return 0.0

        # 速度计算（m/s → km/h）
        velocity = self.vehicle.get_velocity()
        speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_km_h = speed_m_s * 3.6

        # 超速判定
        over_speed = speed_km_h - self.speed_limit_urban
        reward = 0.0
        if over_speed > 0:
            if 0 < over_speed <= 10:
                reward = self.over_speed_light_penalty
            else:
                reward = self.over_speed_heavy_penalty
            self.last_over_speed_time = current_time

        return reward

    # ========== 车道偏离检测（3秒检测 + 10秒日志 + 无特殊符号） ==========
    def _check_lane_offset(self):
        current_time = time.time()
        # 仅每3秒执行一次检测
        if current_time - self.last_lane_check_time < self.lane_check_interval:
            return 0.0
        
        # 更新检测时间戳
        self.last_lane_check_time = current_time

        if not self.vehicle or not self.vehicle.is_alive:
            return 0.0

        # 获取当前车道waypoint和车辆位置
        vehicle_transform = self.vehicle.get_transform()
        waypoint = self.world.get_map().get_waypoint(vehicle_transform.location, project_to_road=True)
        
        # 完全偏离道路（无可用waypoint）
        if not waypoint:
            # 仅每10秒输出一次日志
            if self.enable_lane_log and current_time - self.last_lane_log_time >= self.lane_log_interval:
                print(f"完全偏离道路！扣分{self.lane_heavy_penalty}")
                self.last_lane_log_time = current_time
            return self.lane_heavy_penalty

        # 计算车辆与车道中心线的偏移量（投影到车道垂直方向）
        lane_center = waypoint.transform.location
        vehicle_loc = vehicle_transform.location
        yaw_rad = np.radians(waypoint.transform.rotation.yaw)
        
        # 偏移量计算：消除车道方向的影响，仅保留垂直偏移
        offset_x = vehicle_loc.x - lane_center.x
        offset_y = vehicle_loc.y - lane_center.y
        offset = np.abs(offset_x * np.sin(yaw_rad) - offset_y * np.cos(yaw_rad))

        # 按偏移量分级判定（3秒检测一次，10秒日志一次）
        reward = 0.0
        if offset < self.lane_offset_light:
            # 保持车道内：奖励（无日志）
            reward = self.lane_keep_reward
        elif offset < self.lane_offset_heavy:
            # 轻微偏离：单次扣0.2分
            reward = self.lane_light_penalty
            # 仅每10秒输出一次日志
            if self.enable_lane_log and current_time - self.last_lane_log_time >= self.lane_log_interval:
                print(f"轻微偏离车道（偏移{offset:.2f}m）！扣分{self.lane_light_penalty}")
                self.last_lane_log_time = current_time
        else:
            # 严重偏离：单次扣1.0分
            reward = self.lane_heavy_penalty
            # 仅每10秒输出一次日志
            if self.enable_lane_log and current_time - self.last_lane_log_time >= self.lane_log_interval:
                print(f"严重偏离车道（偏移{offset:.2f}m）！扣分{self.lane_heavy_penalty}")
                self.last_lane_log_time = current_time

        return reward

    # ========== 安全生成车辆（移除特殊符号） ==========
    def _spawn_vehicle_safely(self, vehicle_bp):
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            spawn_points = [carla.Transform(
                carla.Location(x=random.uniform(-50, 50), y=random.uniform(-50, 50), z=0.5),
                carla.Rotation(yaw=random.uniform(0, 360))
            )]

        for attempt in range(self.spawn_retry_times):
            spawn_point = random.choice(spawn_points)
            spawn_point.location.z += 0.2

            try:
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if vehicle is not None:
                    print(f"车辆生成成功（重试{attempt}次）")
                    return vehicle
            except RuntimeError as e:
                print(f"出生点碰撞，重试第{attempt+1}次...")
                continue

        raise RuntimeError("所有出生点都有碰撞，无法生成车辆！")

    # ========== 优化NPC生成 ==========
    def _spawn_small_npc(self):
        for v in self.npc_vehicle_list:
            if v.is_alive:
                v.destroy()
        self.npc_vehicle_list.clear()
        for p in self.npc_pedestrian_list:
            if p.is_alive:
                p.destroy()
        self.npc_pedestrian_list.clear()

        vehicle_bps = self.blueprint_library.filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()
        
        if len(spawn_points) < 50:
            for _ in range(50 - len(spawn_points)):
                random_x = np.random.uniform(-200, 200)
                random_y = np.random.uniform(-200, 200)
                spawn_points.append(carla.Transform(carla.Location(x=random_x, y=random_y, z=0.5)))
        
        if self.vehicle is not None:
            hero_loc = self.vehicle.get_transform().location
            valid_spawn = []
            for sp in spawn_points:
                dist = np.linalg.norm([sp.location.x - hero_loc.x, sp.location.y - hero_loc.y])
                if dist > 15.0:
                    valid_spawn.append(sp)
            spawn_points = valid_spawn[:50]
        else:
            spawn_points = spawn_points[:50]

        for sp in spawn_points:
            try:
                sp.location.z += 0.1
                npc_vehicle = self.world.try_spawn_actor(random.choice(vehicle_bps), sp)
                if npc_vehicle is not None:
                    self.npc_vehicle_list.append(npc_vehicle)
                    npc_vehicle.set_autopilot(True, self.traffic_manager.get_port())
            except:
                continue

        pedestrian_bps = self.blueprint_library.filter('walker.pedestrian.*')
        for _ in range(5):
            if self.vehicle is not None:
                hero_loc = self.vehicle.get_transform().location
                random_x = hero_loc.x + random.uniform(20, 50) * (1 if random.random()>0.5 else -1)
                random_y = hero_loc.y + random.uniform(20, 50) * (1 if random.random()>0.5 else -1)
            else:
                random_x = np.random.uniform(-100, 100)
                random_y = np.random.uniform(-100, 100)
            
            loc = carla.Location(x=random_x, y=random_y, z=0.1)
            try:
                pedestrian = self.world.try_spawn_actor(random.choice(pedestrian_bps), carla.Transform(loc))
                if pedestrian:
                    self.npc_pedestrian_list.append(pedestrian)
            except:
                continue

    # ========== 初始化碰撞传感器 ==========
    def _init_collision_sensor(self):
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, collision_transform, attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._collision_callback(event))

    # ========== 碰撞回调（仅标记状态） ==========
    def _collision_callback(self, event):
        self.has_collision = True
        other_actor = event.other_actor
        other_actor_type = other_actor.type_id
        if 'vehicle' in other_actor_type:
            self.hit_vehicle = True
        elif 'walker' in other_actor_type:
            self.hit_pedestrian = True
        elif 'static' in other_actor_type or 'building' in other_actor_type or 'guardrail' in other_actor_type:
            self.hit_static = True

    # ========== 初始化相机 ==========
    def _init_camera(self):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '128')
        camera_bp.set_attribute('image_size_y', '128')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.0')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda img: self._camera_callback(img))

    # ========== 重置环境（重置所有标记） ==========
    def reset(self):
        # 清理旧资源
        if self.vehicle is not None and self.vehicle.is_alive:
            self.vehicle.destroy()
        if self.camera is not None and self.camera.is_alive:
            self.camera.destroy()
        if self.collision_sensor is not None and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
        self.image_data = None
        
        # 重置所有碰撞标记
        self.has_collision = False
        self.hit_vehicle = False
        self.hit_pedestrian = False
        self.hit_static = False
        self.collision_penalty_applied = False

        # 重置红绿灯标记
        self.last_traffic_light_time = 0
        self.has_triggered_red = False
        self.has_triggered_green = False

        # 重置超速标记
        self.last_over_speed_time = 0

        # 重置车道检测/日志标记
        self.last_lane_check_time = 0
        self.last_lane_log_time = 0

        # 生成车辆
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        self.vehicle = self._spawn_vehicle_safely(vehicle_bp)

        # 初始化传感器
        self._init_camera()
        self._init_collision_sensor()

        # 生成NPC
        self._spawn_small_npc()

        # 等待传感器就绪
        timeout = 0
        while self.image_data is None and timeout < 30:
            self.world.tick()
            time.sleep(0.001)
            timeout += 1

        # 绑定视角
        self.follow_vehicle()
        self.world.tick()
        return self.image_data.copy() if self.image_data is not None else np.zeros((128,128,3), dtype=np.uint8)

    # ========== 视角跟随 ==========
    def follow_vehicle(self):
        spectator = self.world.get_spectator()
        if not spectator or not self.vehicle or not self.vehicle.is_alive:
            return

        vehicle_tf = self.vehicle.get_transform()
        vehicle_loc = vehicle_tf.location
        yaw_rad = np.radians(vehicle_tf.rotation.yaw)

        cam_x = vehicle_loc.x - (np.cos(yaw_rad) * self.view_distance)
        cam_y = vehicle_loc.y - (np.sin(yaw_rad) * self.view_distance)
        cam_z = vehicle_loc.z + self.view_height + self.z_offset

        spectator.set_transform(carla.Transform(
            carla.Location(x=cam_x, y=cam_y, z=cam_z),
            carla.Rotation(pitch=self.view_pitch, yaw=vehicle_tf.rotation.yaw, roll=0.0)
        ))

    # ========== 获取观测 ==========
    def get_observation(self):
        return self.image_data.copy() if self.image_data is not None else np.zeros((128, 128, 3), dtype=np.uint8)

    # ========== 相机回调 ==========
    def _camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.image_data = array.reshape((image.height, image.width, 4))[:, :, :3]

    # ========== 核心：奖励函数（碰撞仅扣一次分 + 移除特殊符号） ==========
    def step(self, action):
        if self.vehicle is None or not self.vehicle.is_alive:
            raise RuntimeError("车辆未初始化/已销毁，请先调用reset()")

        # 车辆控制逻辑
        throttle = 0.0
        steer = 0.0
        if action == 0:  # 前进
            throttle = 0.5
        elif action == 1:  # 左转
            throttle = 0.4
            steer = -0.1
        elif action == 2:  # 右转
            throttle = 0.4
            steer = 0.1
        elif action == 3:  # 后退
            throttle = -0.2

        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            hand_brake=False,
            reverse=(throttle < 0),
            gear=1,
            manual_gear_shift=True
        ))
        
        self.world.tick()
        self.follow_vehicle()

        # ========== 1. 基础行驶奖励 ==========
        base_reward = 0.1 if throttle > 0 else (-0.1 if throttle < 0 else 0.0)

        # ========== 2. 红绿灯奖惩 ==========
        traffic_light_reward = self._check_traffic_light()

        # ========== 3. 超速奖惩 ==========
        over_speed_reward = self._check_over_speed()

        # ========== 4. 车道偏离奖惩（3秒检测 + 10秒日志） ==========
        lane_reward = self._check_lane_offset()

        # ========== 5. 碰撞奖惩（仅扣一次分 + 移除特殊符号） ==========
        collision_reward = 0.0
        done = False
        if self.has_collision and not self.collision_penalty_applied:
            # 仅当碰撞发生且未执行过惩罚时，才扣分
            if self.hit_pedestrian:
                collision_reward = self.collision_pedestrian_penalty
                print(f"碰撞行人！扣分{self.collision_pedestrian_penalty}，终止训练")
                done = True
            elif self.hit_vehicle:
                collision_reward = self.collision_vehicle_penalty
                print(f"碰撞车辆！扣分{self.collision_vehicle_penalty}，终止训练")
                done = True
            elif self.hit_static:
                collision_reward = self.collision_static_penalty
                print(f"碰撞静态物体！扣分{self.collision_static_penalty}")
                done = False  # 碰撞静态物体不终止训练
            
            # 标记惩罚已执行，避免重复扣分
            self.collision_penalty_applied = True

        # ========== 总奖励计算 ==========
        total_reward = (
            base_reward          # 基础行驶
            + traffic_light_reward  # 红绿灯
            + over_speed_reward     # 超速
            + lane_reward           # 车道偏离
            + collision_reward      # 碰撞（仅一次）
        )

        next_state = self.get_observation()

        return next_state, total_reward, done, {
            "base_reward": base_reward,
            "traffic_light_reward": traffic_light_reward,
            "over_speed_reward": over_speed_reward,
            "lane_reward": lane_reward,
            "collision_reward": collision_reward,
            "total_reward": total_reward
        }

    # ========== 关闭环境（移除特殊符号） ==========
    def close(self):
        # 清理NPC
        for v in self.npc_vehicle_list:
            if v.is_alive:
                v.destroy()
        self.npc_vehicle_list.clear()
        for p in self.npc_pedestrian_list:
            if p.is_alive:
                p.destroy()
        self.npc_pedestrian_list.clear()
        self.traffic_manager.set_synchronous_mode(False)

        # 恢复同步设置
        try:
            self.sync_settings.synchronous_mode = False
            self.world.apply_settings(self.sync_settings)
        except Exception as e:
            print(f"恢复异步模式时警告：{e}")

        # 销毁核心对象
        try:
            if self.vehicle is not None and self.vehicle.is_alive:
                self.vehicle.destroy()
        except Exception as e:
            print(f"销毁车辆时警告：{e}")
        
        try:
            if self.camera is not None and self.camera.is_alive:
                self.camera.destroy()
        except Exception as e:
            print(f"销毁相机时警告：{e}")
        
        try:
            if self.collision_sensor is not None and self.collision_sensor.is_alive:
                self.collision_sensor.destroy()
        except Exception as e:
            print(f"销毁碰撞传感器时警告：{e}")

        time.sleep(0.5)
        print("CARLA环境已关闭（同步模式已恢复为异步）")