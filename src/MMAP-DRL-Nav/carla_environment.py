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

        # ========== 修复核心错误：从World获取蓝图库（而非Map） ==========
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

        # ========== 天气系统：恢复所有原始天气 + 昼夜切换 ==========
        self.weather_mode = "dynamic"
        # 恢复所有原始天气类型 + 自定义昼夜天气
        self.weather_name_map = {
            # 白天天气（原有）
            carla.WeatherParameters.ClearNoon: "晴天（中午）",
            carla.WeatherParameters.CloudyNoon: "多云（中午）",
            carla.WeatherParameters.WetNoon: "小雨（中午）",
            carla.WeatherParameters.WetCloudyNoon: "多云转小雨（中午）",
            carla.WeatherParameters.MidRainyNoon: "中雨（中午）",
            carla.WeatherParameters.HardRainNoon: "大雨（中午）",
            carla.WeatherParameters.SoftRainNoon: "细雨（中午）",
            # 自定义夜晚天气（补充）
            "ClearNight": "晴天（夜晚）",
            "CloudyNight": "多云（夜晚）",
            "WetNight": "小雨（夜晚）"
        }
        
        # 预设天气列表：恢复所有原有 + 自定义夜晚
        self.builtin_weathers = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.CloudyNoon,
            carla.WeatherParameters.WetNoon,
            carla.WeatherParameters.WetCloudyNoon,
            carla.WeatherParameters.MidRainyNoon,
            carla.WeatherParameters.HardRainNoon,
            carla.WeatherParameters.SoftRainNoon
        ]
        self.custom_night_weathers = ["ClearNight", "CloudyNight", "WetNight"]
        self.all_weather_ids = self.builtin_weathers + self.custom_night_weathers
        
        # 随机选择初始天气
        self.current_weather_id = random.choice(self.all_weather_ids)
        self._set_weather_by_id(self.current_weather_id)
        print(f"初始天气：{self.weather_name_map[self.current_weather_id]}")

        # ========== 大灯奖惩配置：夜晚开灯奖励 + 未开灯轻微惩罚 ==========
        self.headlight_required_weathers = ["ClearNight", "CloudyNight", "WetNight"]  # 所有夜晚天气需要开灯
        self.headlight_on_reward = 0.5               # 夜晚开灯奖励（适中）
        self.headlight_off_penalty = -0.2            # 夜晚未开灯轻微惩罚（和轻微车道偏离一致）
        self.headlight_check_cooldown = 2.0          # 检测间隔2秒
        self.last_headlight_check_time = 0
        self.vehicle_headlights_on = False

        # ========== 其他配置完全保留 ==========
        self.target_location = None          
        self.target_radius = 5.0             
        self.nav_reward_per_meter = 0.02     
        self.goal_completion_reward = 100.0  
        self.last_dist_to_target = 0.0       

        self.max_steps = 800                 
        self.current_step = 0                

        self.gear_config = {
            0: 1,
            10: 2,
            20: 3,
            30: 4,
            40: 5
        }
        self.gear_correct_reward = 0.1
        self.last_speed_range = 0
        self.last_gear = 1
        self.is_initial_gear_logged = False

        self.spawn_retry_times = 20
        self.spawn_safe_radius = 2.0

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
        )

        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.image_data = None
        self.has_collision = False

        self.view_height = 6.0
        self.view_pitch = -15.0
        self.view_distance = 8.0
        self.z_offset = 0.5

    # ========== 恢复原始天气设置 + 完善夜晚天气参数 ==========
    def _set_weather_by_id(self, weather_id):
        """兼容内置天气枚举和自定义夜晚天气"""
        if isinstance(weather_id, carla.WeatherParameters):
            # 原有白天天气：直接使用CARLA内置枚举
            self.world.set_weather(weather_id)
        else:
            # 自定义夜晚天气：基于对应白天天气调整光照
            weather = carla.WeatherParameters()
            if weather_id == "ClearNight":
                # 晴天夜晚：极低光照+月光
                weather.sun_altitude_angle = -90.0
                weather.moon_altitude_angle = 30.0
                weather.moon_intensity = 1.0
                weather.cloudiness = 0.0
                weather.precipitation = 0.0
                weather.brightness = 0.1
                weather.directionald_light_intensity = 0.01
                weather.ambient_occlusion_intensity = 0.2
            elif weather_id == "CloudyNight":
                # 多云夜晚
                weather.sun_altitude_angle = -90.0
                weather.moon_altitude_angle = 25.0
                weather.moon_intensity = 0.7
                weather.cloudiness = 80.0
                weather.precipitation = 0.0
                weather.brightness = 0.08
                weather.directionald_light_intensity = 0.008
                weather.ambient_occlusion_intensity = 0.25
            elif weather_id == "WetNight":
                # 小雨夜晚
                weather.sun_altitude_angle = -90.0
                weather.moon_altitude_angle = 20.0
                weather.moon_intensity = 0.5
                weather.cloudiness = 90.0
                weather.precipitation = 20.0
                weather.precipitation_deposits = 10.0
                weather.brightness = 0.07
                weather.directionald_light_intensity = 0.005
                weather.ambient_occlusion_intensity = 0.3
            self.world.set_weather(weather)

    # ========== 大灯奖惩：夜晚开灯奖励 + 未开灯轻微惩罚 ==========
    def _check_headlight_status(self):
        current_time = time.time()
        # 冷却时间内不检测
        if current_time - self.last_headlight_check_time < self.headlight_check_cooldown:
            return 0.0
        
        self.last_headlight_check_time = current_time

        if not self.vehicle or not self.vehicle.is_alive:
            return 0.0
        
        # 仅夜晚天气检测大灯
        need_headlight = self.current_weather_id in self.headlight_required_weathers
        if not need_headlight:
            return 0.0
        
        # 获取大灯状态
        try:
            vehicle_light_state = self.vehicle.get_light_state()
            self.vehicle_headlights_on = vehicle_light_state == carla.VehicleLightState.HeadlightOn
        except Exception as e:
            self.vehicle_headlights_on = False
            print(f"检测大灯状态失败：{e}")
        
        reward = 0.0
        if need_headlight:
            if self.vehicle_headlights_on:
                # 夜晚开灯：奖励
                reward = self.headlight_on_reward
                print(f"[{self.weather_name_map[self.current_weather_id]}] 大灯已开启，加分{self.headlight_on_reward}")
            else:
                # 夜晚未开灯：轻微惩罚
                reward = self.headlight_off_penalty
                print(f"[{self.weather_name_map[self.current_weather_id]}] 大灯未开启，扣分{abs(self.headlight_off_penalty)}")
        
        return reward

    # ========== 以下所有方法完全保留原始代码，未做任何修改 ==========
    def _calculate_gear_reward(self):
        if not self.vehicle or not self.vehicle.is_alive:
            return 0.0
        
        velocity = self.vehicle.get_velocity()
        speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = round(speed_m_s * 3.6, 1)
        
        current_speed_range = 0
        if speed_kmh < 0:
            current_speed_range = -1
            recommended_gear = 'R'
        else:
            for range_val in sorted(self.gear_config.keys(), reverse=True):
                if speed_kmh >= range_val:
                    current_speed_range = range_val
                    recommended_gear = self.gear_config[range_val]
                    break
        
        control = self.vehicle.get_control()
        current_gear = control.gear
        if control.reverse or speed_kmh < 0:
            current_gear = 'R'

        gear_reward = 0.0
        if not self.is_initial_gear_logged:
            is_correct = (current_gear == recommended_gear)
            if is_correct:
                gear_reward = self.gear_correct_reward
                if speed_kmh < 0:
                    print(f"初始倒车档位正确加分：{self.gear_correct_reward}（车速{speed_kmh}km/h，当前R挡）")
                else:
                    print(f"初始档位正确加分：{self.gear_correct_reward}（车速{speed_kmh}km/h，当前{current_gear}挡，对应{current_speed_range}-{current_speed_range+10}km/h区间）")
            self.is_initial_gear_logged = True
        elif current_speed_range != self.last_speed_range:
            is_correct = (current_gear == recommended_gear)
            if is_correct:
                gear_reward = self.gear_correct_reward
                if speed_kmh < 0:
                    print(f"倒车档位正确加分：{self.gear_correct_reward}（车速{speed_kmh}km/h，当前R挡）")
                else:
                    print(f"车速进入{current_speed_range}-{current_speed_range+10}km/h区间，档位{current_gear}正确加分：{self.gear_correct_reward}")
            self.last_speed_range = current_speed_range
        
        return gear_reward

    def switch_weather(self):
        available_weathers = [w for w in self.all_weather_ids if w != self.current_weather_id]
        self.current_weather_id = random.choice(available_weathers)
        self._set_weather_by_id(self.current_weather_id)
        print(f"天气已切换为：{self.weather_name_map[self.current_weather_id]}")

    def _generate_random_target(self):
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            hero_loc = self.vehicle.get_transform().location if self.vehicle else carla.Location(x=0, y=0, z=0.5)
            random_x = hero_loc.x + random.uniform(-200, 200)
            random_y = hero_loc.y + random.uniform(-200, 200)
            self.target_location = carla.Location(x=random_x, y=random_y, z=0.5)
        else:
            target_spawn = random.choice(spawn_points)
            self.target_location = target_spawn.location
        if self.vehicle and self.vehicle.is_alive:
            self.last_dist_to_target = self.vehicle.get_transform().location.distance(self.target_location)
        print(f"生成导航目标点：({self.target_location.x:.2f}, {self.target_location.y:.2f})，初始距离：{self.last_dist_to_target:.2f}米")

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

        velocity = self.vehicle.get_velocity()
        speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        is_stopped = speed_m_s < 0.1

        for light in traffic_lights:
            dist = vehicle_loc.distance(light.get_transform().location)
            if dist <= self.traffic_light_trigger_distance:
                has_near_light = True
                light_state = light.state

                if light_state == carla.TrafficLightState.Red:
                    if not is_stopped and not self.has_triggered_red:
                        reward = self.red_light_penalty
                        print(f"闯红灯！扣分{self.red_light_penalty}")
                        self.has_triggered_red = True
                        self.last_traffic_light_time = current_time
                    elif is_stopped:
                        reward = self.red_light_stop_reward
                    break

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

    def _check_over_speed(self):
        current_time = time.time()
        if current_time - self.last_over_speed_time < self.over_speed_cooldown:
            return 0.0

        if not self.vehicle or not self.vehicle.is_alive:
            return 0.0

        velocity = self.vehicle.get_velocity()
        speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_km_h = speed_m_s * 3.6

        over_speed = speed_km_h - self.speed_limit_urban
        reward = 0.0
        if over_speed > 0:
            if 0 < over_speed <= 10:
                reward = self.over_speed_light_penalty
            else:
                reward = self.over_speed_heavy_penalty
            self.last_over_speed_time = current_time

        return reward

    def _check_lane_offset(self):
        current_time = time.time()
        if current_time - self.last_lane_check_time < self.lane_check_interval:
            return 0.0
        
        self.last_lane_check_time = current_time

        if not self.vehicle or not self.vehicle.is_alive:
            return 0.0

        vehicle_transform = self.vehicle.get_transform()
        waypoint = self.world.get_map().get_waypoint(vehicle_transform.location, project_to_road=True)
        
        if not waypoint:
            if self.enable_lane_log and current_time - self.last_lane_log_time >= self.lane_log_interval:
                print(f"完全偏离道路！扣分{self.lane_heavy_penalty}")
                self.last_lane_log_time = current_time
            return self.lane_heavy_penalty

        lane_center = waypoint.transform.location
        vehicle_loc = vehicle_transform.location
        yaw_rad = np.radians(waypoint.transform.rotation.yaw)
        
        offset_x = vehicle_loc.x - lane_center.x
        offset_y = vehicle_loc.y - lane_center.y
        offset = np.abs(offset_x * np.sin(yaw_rad) - offset_y * np.cos(yaw_rad))

        reward = 0.0
        if offset < self.lane_offset_light:
            reward = self.lane_keep_reward
        elif offset < self.lane_offset_heavy:
            reward = self.lane_light_penalty
            if self.enable_lane_log and current_time - self.last_lane_log_time >= self.lane_log_interval:
                print(f"轻微偏离车道（偏移{offset:.2f}m）！扣分{self.lane_light_penalty}")
                self.last_lane_log_time = current_time
        else:
            reward = self.lane_heavy_penalty
            if self.enable_lane_log and current_time - self.last_lane_log_time >= self.lane_log_interval:
                print(f"严重偏离车道（偏移{offset:.2f}m）！扣分{self.lane_heavy_penalty}")
                self.last_lane_log_time = current_time

        return reward

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

    def _spawn_small_npc(self):
        for v in self.npc_vehicle_list:
            if v.is_alive:
                v.destroy()
        self.npc_vehicle_list.clear()
        for p in self.npc_pedestrian_list:
            if p.is_alive:
                p.destroy()
        self.npc_pedestrian_list.clear()

        vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*')
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

        pedestrian_bps = self.world.get_blueprint_library().filter('walker.pedestrian.*')
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

    def _init_collision_sensor(self):
        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, collision_transform, attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._collision_callback(event))

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

    def _init_camera(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '128')
        camera_bp.set_attribute('image_size_y', '128')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.0')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda img: self._camera_callback(img))

    def reset(self):
        # 清理资源
        if self.vehicle is not None and self.vehicle.is_alive:
            self.vehicle.destroy()
        if self.camera is not None and self.camera.is_alive:
            self.camera.destroy()
        if self.collision_sensor is not None and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
        self.image_data = None
        
        # 重置状态
        self.has_collision = False
        self.hit_vehicle = False
        self.hit_pedestrian = False
        self.hit_static = False
        self.collision_penalty_applied = False
        self.last_traffic_light_time = 0
        self.has_triggered_red = False
        self.has_triggered_green = False
        self.last_over_speed_time = 0
        self.last_lane_check_time = 0
        self.last_lane_log_time = 0

        self.current_step = 0
        self.last_speed_range = 0
        self.last_gear = 1
        self.is_initial_gear_logged = False
        self.last_headlight_check_time = 0
        self.vehicle_headlights_on = False

        # 切换天气
        self.switch_weather()

        # 生成车辆
        vehicle_bp = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        self.vehicle = self._spawn_vehicle_safely(vehicle_bp)

        # 夜晚自动开灯
        if self.current_weather_id in self.headlight_required_weathers:
            try:
                self.vehicle.set_light_state(carla.VehicleLightState.HeadlightOn)
                self.vehicle_headlights_on = True
                print(f"当前为{self.weather_name_map[self.current_weather_id]}，已自动开启大灯")
            except Exception as e:
                self.vehicle_headlights_on = False
                print(f"自动开启大灯失败：{e}")

        # 生成目标点 + 初始化传感器 + 生成NPC
        self._generate_random_target()
        self._init_camera()
        self._init_collision_sensor()
        self._spawn_small_npc()

        # 等待相机数据
        timeout = 0
        while self.image_data is None and timeout < 30:
            self.world.tick()
            time.sleep(0.001)
            timeout += 1

        # 跟随车辆
        self.follow_vehicle()
        self.world.tick()
        
        return self.image_data.copy() if self.image_data is not None else np.zeros((128,128,3), dtype=np.uint8)

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

    def get_observation(self):
        return self.image_data.copy() if self.image_data is not None else np.zeros((128, 128, 3), dtype=np.uint8)

    def _camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.image_data = array.reshape((image.height, image.width, 4))[:, :, :3]

    def step(self, action):
        if self.vehicle is None or not self.vehicle.is_alive:
            raise RuntimeError("车辆未初始化/已销毁，请先调用reset()")

        self.current_step += 1

        # 解析动作
        throttle = 0.0
        steer = 0.0
        if action == 0:
            throttle = 0.5
        elif action == 1:
            throttle = 0.4
            steer = -0.1
        elif action == 2:
            throttle = 0.4
            steer = 0.1
        elif action == 3:
            throttle = -0.2

        # 计算车速和档位
        velocity = self.vehicle.get_velocity()
        speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        current_speed = speed_m_s * 3.6
        
        gear = 1
        if throttle < 0:
            gear = -1
        else:
            if 0 <= current_speed < 10:
                gear = 1
            elif 10 <= current_speed < 20:
                gear = 2
            elif 20 <= current_speed < 30:
                gear = 3
            elif 30 <= current_speed < 40:
                gear = 4
            else:
                gear = 5

        # 控制车辆
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            hand_brake=False,
            reverse=(throttle < 0),
            gear=gear,
            manual_gear_shift=True
        ))
        
        self.world.tick()
        self.follow_vehicle()

        # 计算各项奖励
        base_reward = 0.1 if throttle > 0 else (-0.1 if throttle < 0 else 0.0)
        traffic_light_reward = self._check_traffic_light()
        over_speed_reward = self._check_over_speed()
        lane_reward = self._check_lane_offset()
        collision_reward = 0.0
        gear_reward = self._calculate_gear_reward()
        headlight_reward = self._check_headlight_status()
        
        done = False

        # 碰撞惩罚
        if self.has_collision and not self.collision_penalty_applied:
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
                done = False
            self.collision_penalty_applied = True

        # 导航奖励
        nav_reward = 0.0
        dist_to_target = -1.0
        if self.target_location is not None and self.vehicle.is_alive:
            vehicle_loc = self.vehicle.get_transform().location
            dist_to_target = vehicle_loc.distance(self.target_location)
            
            if dist_to_target < self.target_radius:
                nav_reward = self.goal_completion_reward
                print(f"到达目标点！奖励{self.goal_completion_reward}，终止训练")
                done = True
            else:
                dist_diff = self.last_dist_to_target - dist_to_target
                nav_reward = dist_diff * self.nav_reward_per_meter
                nav_reward = max(nav_reward, -0.01)
                self.last_dist_to_target = dist_to_target

        # 步数超限
        if self.current_step >= self.max_steps and not done:
            print(f"达到最大步数{self.max_steps}，终止训练（当前距离目标点：{dist_to_target:.2f}米）")
            done = True

        # 总奖励
        total_reward = base_reward + traffic_light_reward + over_speed_reward + lane_reward + collision_reward + nav_reward + gear_reward + headlight_reward
        next_state = self.get_observation()

        return next_state, total_reward, done, {
            "base_reward": base_reward,
            "traffic_light_reward": traffic_light_reward,
            "over_speed_reward": over_speed_reward,
            "lane_reward": lane_reward,
            "collision_reward": collision_reward,
            "nav_reward": nav_reward,
            "gear_reward": gear_reward,
            "headlight_reward": headlight_reward,
            "total_reward": total_reward,
            "current_step": self.current_step,
            "dist_to_target": dist_to_target,
            "current_weather": self.weather_name_map[self.current_weather_id]
        }

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

        # 恢复同步模式
        try:
            self.sync_settings.synchronous_mode = False
            self.world.apply_settings(self.sync_settings)
        except Exception as e:
            print(f"恢复异步模式时警告：{e}")

        # 销毁车辆和传感器
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