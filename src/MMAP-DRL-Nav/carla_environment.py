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
            raise RuntimeError("❌ 2000端口未被占用，请先启动CARLA模拟器")

        max_retry = 3
        retry_count = 0
        while retry_count < max_retry:
            try:
                self.world = self.client.get_world()
                break
            except RuntimeError as e:
                retry_count += 1
                print(f"⚠️ 连接失败，重试第{retry_count}次...")
                time.sleep(5)
        else:
            raise RuntimeError("❌ CARLA连接超时（3次重试失败），请检查模拟器是否正常启动")

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

        # ========== 红绿灯奖惩核心配置（长间隔防重复） ==========
        self.red_light_penalty = -10.0    # 闯红灯扣分
        self.green_light_reward = 5.0     # 绿灯加分
        self.traffic_light_cooldown = 10.0 # 延长冷却到10秒（核心修改）
        self.last_traffic_light_time = 0  # 上次奖惩时间
        self.traffic_light_trigger_distance = 10.0  # 触发距离
        self.traffic_light_reset_distance = 15.0    # 离开15米才重置标记
        self.has_triggered_red = False    # 红灯触发标记
        self.has_triggered_green = False  # 绿灯触发标记

        # 出生点碰撞检测配置
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

    # ========== 核心修复：红绿灯判定（长间隔+防重复） ==========
    def _check_traffic_light(self):
        """
        优化逻辑：
        1. 冷却时间延长到10秒，单次闯红灯仅扣1次分
        2. 车辆离开15米范围才重置标记，低速行驶不重复判定
        """
        current_time = time.time()
        # 10秒冷却内不判定
        if current_time - self.last_traffic_light_time < self.traffic_light_cooldown:
            return 0.0

        if not self.vehicle or not self.vehicle.is_alive:
            return 0.0

        vehicle_loc = self.vehicle.get_transform().location
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        reward = 0.0
        has_near_light = False  # 是否有近距离灯

        for light in traffic_lights:
            dist = vehicle_loc.distance(light.get_transform().location)
            # 标记：有10米内的灯
            if dist <= self.traffic_light_trigger_distance:
                has_near_light = True
                light_state = light.state
                # 红灯：仅首次触发扣分
                if light_state == carla.TrafficLightState.Red and not self.has_triggered_red:
                    reward = self.red_light_penalty
                    print(f"⚠️ 闯红灯！扣分{self.red_light_penalty}")
                    self.has_triggered_red = True
                    self.has_triggered_green = False
                    self.last_traffic_light_time = current_time
                    break
                # 绿灯：仅首次触发加分
                elif light_state == carla.TrafficLightState.Green and not self.has_triggered_green:
                    reward = self.green_light_reward
                    print(f"✅ 绿灯合规通过！加分{self.green_light_reward}")
                    self.has_triggered_green = True
                    self.has_triggered_red = False
                    self.last_traffic_light_time = current_time
                    break
            # 离开15米范围，才重置触发标记
            elif dist > self.traffic_light_reset_distance:
                self.has_triggered_red = False
                self.has_triggered_green = False

        # 无近距离灯时，也重置标记（避免卡标记）
        if not has_near_light:
            self.has_triggered_red = False
            self.has_triggered_green = False

        return reward

    # ========== 安全生成车辆（保留） ==========
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
                    print(f"✅ 车辆生成成功（重试{attempt}次）")
                    return vehicle
            except RuntimeError as e:
                print(f"⚠️ 出生点碰撞，重试第{attempt+1}次...")
                continue

        raise RuntimeError("❌ 所有出生点都有碰撞，无法生成车辆！")

    # ========== 优化NPC生成（保留） ==========
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

    # ========== 初始化碰撞传感器（保留） ==========
    def _init_collision_sensor(self):
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, collision_transform, attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._collision_callback(event))

    # ========== 初始化相机（保留） ==========
    def _init_camera(self):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '128')
        camera_bp.set_attribute('image_size_y', '128')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.0')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda img: self._camera_callback(img))

    # ========== 重置环境（保留+重置红绿灯标记） ==========
    def reset(self):
        # 清理旧资源
        if self.vehicle is not None and self.vehicle.is_alive:
            self.vehicle.destroy()
        if self.camera is not None and self.camera.is_alive:
            self.camera.destroy()
        if self.collision_sensor is not None and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
        self.image_data = None
        self.has_collision = False

        # 重置碰撞标记
        self.hit_vehicle = False
        self.hit_pedestrian = False

        # 重置红绿灯触发标记
        self.last_traffic_light_time = 0
        self.has_triggered_red = False
        self.has_triggered_green = False

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

    # ========== 碰撞回调（保留） ==========
    def _collision_callback(self, event):
        self.has_collision = True
        other_actor_type = event.other_actor.type_id
        if 'vehicle' in other_actor_type:
            self.hit_vehicle = True
        elif 'walker' in other_actor_type:
            self.hit_pedestrian = True

    # ========== 相机回调（保留） ==========
    def _camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.image_data = array.reshape((image.height, image.width, 4))[:, :, :3]

    # ========== 视角跟随（保留） ==========
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

    # ========== 获取观测（保留） ==========
    def get_observation(self):
        return self.image_data.copy() if self.image_data is not None else np.zeros((128, 128, 3), dtype=np.uint8)

    # ========== 执行单步动作（保留+红绿灯奖惩） ==========
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

        # 计算总奖励（基础奖励+红绿灯奖惩）
        base_reward = 0.1 if throttle > 0 else (-0.1 if throttle < 0 else 0.0)
        traffic_light_reward = self._check_traffic_light()
        total_reward = base_reward + traffic_light_reward

        next_state = self.get_observation()
        done = self.hit_vehicle

        return next_state, total_reward, done, {
            "base_reward": base_reward,
            "traffic_light_reward": traffic_light_reward,
            "total_reward": total_reward
        }

    # ========== 关闭环境（保留） ==========
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
            print(f"⚠️ 恢复异步模式时警告：{e}")

        # 销毁核心对象
        try:
            if self.vehicle is not None and self.vehicle.is_alive:
                self.vehicle.destroy()
        except Exception as e:
            print(f"⚠️ 销毁车辆时警告：{e}")
        
        try:
            if self.camera is not None and self.camera.is_alive:
                self.camera.destroy()
        except Exception as e:
            print(f"⚠️ 销毁相机时警告：{e}")
        
        try:
            if self.collision_sensor is not None and self.collision_sensor.is_alive:
                self.collision_sensor.destroy()
        except Exception as e:
            print(f"⚠️ 销毁碰撞传感器时警告：{e}")

        time.sleep(0.5)
        print("✅ CARLA环境已关闭（同步模式已恢复为异步）")