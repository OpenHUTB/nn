import carla
import numpy as np
import gym
import time
import socket  # 端口检测
import random  # 仅新增：随机生成NPC位置/选择蓝图

class CarlaEnvironment(gym.Env):
    def __init__(self):
        super(CarlaEnvironment, self).__init__()
        # 初始化CARLA客户端
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)  # 超时20秒

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
                # 加载CARLA默认地图（避免map not found）
                self.world = self.client.get_world()  
                break
            except RuntimeError as e:
                retry_count += 1
                print(f"⚠️ 连接失败，重试第{retry_count}次...")
                time.sleep(5)
        else:
            raise RuntimeError("❌ CARLA连接超时（3次重试失败），请检查模拟器是否正常启动")

        self.blueprint_library = self.world.get_blueprint_library()

        # ========== 同步模式配置（视角不抖核心） ==========
        self.sync_settings = self.world.get_settings()
        self.sync_settings.synchronous_mode = True  # 同步模式是视角不抖的关键
        self.sync_settings.fixed_delta_seconds = 1.0 / 30  # 固定30fps，避免帧率波动
        self.sync_settings.no_rendering_mode = False
        self.world.apply_settings(self.sync_settings)

        # ========== NPC配置（50辆车+5行人） ==========
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        self.npc_vehicle_list = []  # 存储NPC车辆
        self.npc_pedestrian_list = []  # 存储NPC行人
        self.hit_vehicle = False  # 撞车标记（终止）
        self.hit_pedestrian = False  # 撞人标记（不终止）

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

        # 视角参数（核心修改：进一步后移视角，确保看到车屁股）
        self.view_height = 6.0       # 基础高度保持6米（高视角）
        self.view_pitch = -15.0      # 俯视角度保持15度（向下看）
        self.view_distance = 8.0     # 正后方距离从6→8米（大幅后移，必看车屁股）
        self.z_offset = 0.5          # z轴补偿保留，避免上坡卡地下

    # ========== NPC生成逻辑（完全保留） ==========
    def _spawn_small_npc(self):
        # 清理旧NPC
        for v in self.npc_vehicle_list:
            if v.is_alive:
                v.destroy()
        self.npc_vehicle_list.clear()
        for p in self.npc_pedestrian_list:
            if p.is_alive:
                p.destroy()
        self.npc_pedestrian_list.clear()

        # 生成50辆NPC车辆（优化：过滤主角车辆附近的生成点，避免初始碰撞）
        vehicle_bps = self.blueprint_library.filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()
        
        # 优化1：如果预设生成点不足50个，补充随机生成点
        if len(spawn_points) < 50:
            for _ in range(50 - len(spawn_points)):
                random_x = np.random.uniform(-200, 200)
                random_y = np.random.uniform(-200, 200)
                spawn_points.append(carla.Transform(carla.Location(x=random_x, y=random_y, z=0.5)))
        
        # 优化2：过滤距离主角车辆<15米的生成点，避免初始碰撞
        if self.vehicle is not None:
            hero_loc = self.vehicle.get_transform().location
            valid_spawn = []
            for sp in spawn_points:
                dist = np.linalg.norm([sp.location.x - hero_loc.x, sp.location.y - hero_loc.y])
                if dist > 15.0:
                    valid_spawn.append(sp)
            spawn_points = valid_spawn[:50]  # 仅取前50个安全生成点
        else:
            spawn_points = spawn_points[:50]

        # 生成50辆NPC车辆
        for sp in spawn_points:
            try:
                npc_vehicle = self.world.spawn_actor(random.choice(vehicle_bps), sp)
                self.npc_vehicle_list.append(npc_vehicle)
                npc_vehicle.set_autopilot(True, self.traffic_manager.get_port())
            except:
                continue

        # 生成5个NPC行人（数量不变）
        pedestrian_bps = self.blueprint_library.filter('walker.pedestrian.*')
        for _ in range(5):
            # 随机生成行人位置（地图内合法范围）
            loc = carla.Location(x=random.uniform(-100, 100), y=random.uniform(-100, 100), z=0)
            try:
                pedestrian = self.world.try_spawn_actor(random.choice(pedestrian_bps), carla.Transform(loc))
                if pedestrian:
                    self.npc_pedestrian_list.append(pedestrian)
            except:
                continue

    # ========== 补全缺失的 _init_collision_sensor 方法（核心修复） ==========
    def _init_collision_sensor(self):
        """初始化碰撞传感器（原配置）"""
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, collision_transform, attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._collision_callback(event))

    def _init_camera(self):
        """初始化RGB相机（完全保留）"""
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '128')
        camera_bp.set_attribute('image_size_y', '128')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.0')  # 同步模式下无延迟
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda img: self._camera_callback(img))

    def reset(self):
        """重置环境（完全保留）"""
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

        # 仅随机选择出生点（无任何打印）
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        
        # 随机选预设出生点（优先），无则随机生成坐标
        if len(spawn_points) > 0:
            spawn_point = np.random.choice(spawn_points)
        else:
            # 随机生成合法范围坐标
            random_x = np.random.uniform(-200, 200)
            random_y = np.random.uniform(-200, 200)
            spawn_point = carla.Transform(carla.Location(x=random_x, y=random_y, z=0.5))

        # 生成车辆
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(False)

        # 初始化传感器（调用补全的方法）
        self._init_camera()
        self._init_collision_sensor()

        # 调用生成NPC
        self._spawn_small_npc()

        # 等待传感器就绪
        timeout = 0
        while self.image_data is None and timeout < 30:
            self.world.tick()  # 同步tick，保证传感器稳定
            time.sleep(0.001)
            timeout += 1

        # 绑定视角（优化后的逻辑）
        self.follow_vehicle()
        self.world.tick()
        return self.image_data.copy() if self.image_data is not None else np.zeros((128,128,3), dtype=np.uint8)

    def _collision_callback(self, event):
        """碰撞回调（完全保留）"""
        self.has_collision = True
        other_actor_type = event.other_actor.type_id
        if 'vehicle' in other_actor_type:
            self.hit_vehicle = True  # 撞车标记
        elif 'walker' in other_actor_type:
            self.hit_pedestrian = True  # 撞人标记

    def _camera_callback(self, image):
        """相机数据回调（完全保留）"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.image_data = array.reshape((image.height, image.width, 4))[:, :, :3]

    def follow_vehicle(self):
        """视角跟随车辆（完全保留逻辑，仅参数变化）"""
        spectator = self.world.get_spectator()
        if not spectator or not self.vehicle or not self.vehicle.is_alive:
            return

        # 精准计算正后方视角（核心优化：适配车辆z轴高度+坡度）
        vehicle_tf = self.vehicle.get_transform()
        vehicle_loc = vehicle_tf.location  # 获取车辆当前3D位置（包含z轴，适配上坡）
        yaw_rad = np.radians(vehicle_tf.rotation.yaw)

        # 1. 计算正后方水平偏移（x/y轴）→ 因view_distance增大，后移更明显
        cam_x = vehicle_loc.x - (np.cos(yaw_rad) * self.view_distance)
        cam_y = vehicle_loc.y - (np.sin(yaw_rad) * self.view_distance)
        
        # 2. 计算z轴高度（核心优化：基于车辆当前z轴+基础高度+补偿）
        cam_z = vehicle_loc.z + self.view_height + self.z_offset

        # 3. 强制设置视角（同步模式不变）
        spectator.set_transform(carla.Transform(
            carla.Location(x=cam_x, y=cam_y, z=cam_z),
            carla.Rotation(pitch=self.view_pitch, yaw=vehicle_tf.rotation.yaw, roll=0.0)
        ))

    def get_observation(self):
        """获取观测数据（完全保留）"""
        return self.image_data.copy() if self.image_data is not None else np.zeros((128, 128, 3), dtype=np.uint8)

    def step(self, action):
        """执行单步动作（完全保留）"""
        if self.vehicle is None or not self.vehicle.is_alive:
            raise RuntimeError("车辆未初始化/已销毁，请先调用reset()")

        # 平滑车辆控制（原参数）
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
        
        self.world.tick()  # 同步帧推进（30fps，无帧率波动）
        self.follow_vehicle()  # 调用优化后的视角跟随

        next_state = self.get_observation()
        reward = 0.1 if throttle > 0 else (-0.1 if throttle < 0 else 0.0)
        
        # 终止条件（撞车才结束）
        done = self.hit_vehicle

        return next_state, reward, done, {}

    def close(self):
        """安全关闭环境（完全保留）"""
        # 清理NPC
        for v in self.npc_vehicle_list:
            if v.is_alive:
                v.destroy()
        for p in self.npc_pedestrian_list:
            if p.is_alive:
                p.destroy()
        self.traffic_manager.set_synchronous_mode(False)

        # 原有清理逻辑
        try:
            self.sync_settings.synchronous_mode = False
            self.world.apply_settings(self.sync_settings)
        except Exception as e:
            print(f"⚠️ 恢复异步模式时警告：{e}")

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

