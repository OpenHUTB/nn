class CarlaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, host, port, town, fps, obs_sensor, obs_res, view_res, reward_fn, action_smoothing,
                 allow_render=True, allow_spectator=True):
        # 初始化环境参数
        self.obs_width, self.obs_height = obs_res  # 观测图像的宽高
        self.spectator_width, self.spectator_height = view_res  # 观察者视角（渲染窗口）的宽高
        self.allow_render = allow_render  # 是否允许渲染图形界面
        self.allow_spectator = allow_spectator  # 是否允许使用观察者视角相机
        self.spectator_camera = None  # 观察者相机对象
        self.episode_idx = -2  # 任务序号计数器
        self.world = None  # CARLA 世界对象
        self.fps = fps  # 帧率
        self.actions = CarlaActions()  # 动作处理类实例
        self.observations = CarlaObservations(self.obs_height, self.obs_width)  # 观测处理类实例
        self.obs_sensor = obs_sensor  # 观测传感器类型
        self.control = carla.VehicleControl()  # 车辆控制对象
        self.action_space = self.actions.get_action_space()  # 动作空间
        self.observation_space = self.observations.get_observation_space()  # 观测空间
        self.max_distance = 3000  # 单次任务最大行驶距离（米）
        self.action_smoothing = action_smoothing  # 动作平滑系数，防止控制量突变
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn  # 奖励函数

        try:
            # 连接 CARLA 服务器
            self.client = carla.Client(host, port)
            self.client.set_timeout(100.0)
            self.client.load_world(map_name=town)  # 加载指定地图
            self.world = self.client.get_world()
            self.world.set_weather(carla.WeatherParameters.ClearNoon)  # 设置天气为晴朗正午
            # 应用世界设置：开启同步模式，固定时间步长
            self.world.apply_settings(
                carla.WorldSettings(
                    synchronous_mode=True,
                    fixed_delta_seconds=1.0 / fps,
                ))
            self.client.reload_world(False)  # 重新加载地图但保留设置

            self.map = self.world.get_map()  # 获取当前地图对象

            # --- 车辆生成与设置 ---
            # 获取特斯拉 Model 3 的蓝图
            self.tesla = self.world.get_blueprint_library().filter('model3')[0]
            self.start_transform = self._get_start_transform()  # 获取初始生成位置
            self.curr_loc = self.start_transform.location
            # 在世界中生成车辆
            self.vehicle = self.world.spawn_actor(self.tesla, self.start_transform)

            # --- 传感器生成 ---
            # 碰撞传感器
            colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
            # 车道入侵传感器
            lanesensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
            # 生成传感器并绑定到车辆
            self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
            self.lanesensor = self.world.spawn_actor(lanesensor, carla.Transform(), attach_to=self.vehicle)
            # 设置传感器监听回调函数
            self.colsensor.listen(self._collision_data)
            self.lanesensor.listen(self._lane_invasion_data)

            # --- 图形界面初始化 (Pygame) ---
            if self.allow_render:
                pygame.init()
                pygame.font.init()
                self.display = pygame.display.set_mode((self.spectator_width, self.spectator_height),
                                                       pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()
                self.hud = HUD(self.spectator_width, self.spectator_height)
                self.hud.set_vehicle(self.vehicle)
                self.world.on_tick(self.hud.on_world_tick)  # 注册世界时间步回调

            # --- 观测相机设置 ---
            if 'rgb' in self.obs_sensor:
                self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
            elif 'semantic' in self.obs_sensor:
                self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            else:
                raise NotImplementedError('unknown sensor type')

            # 设置相机属性：分辨率和视场角
            self.rgb_cam.set_attribute('image_size_x', f'{self.obs_width}')
            self.rgb_cam.set_attribute('image_size_y', f'{self.obs_height}')
            self.rgb_cam.set_attribute('fov', '90')
            bound_x = self.vehicle.bounding_box.extent.x
            transform_front = carla.Transform(carla.Location(x=bound_x, z=1.0))  # 相机位于车头前方
            # 生成观测相机
            self.sensor_front = self.world.spawn_actor(self.rgb_cam, transform_front, attach_to=self.vehicle)
            self.sensor_front.listen(self._set_observation_image)  # 设置回调函数

            # --- 观察者视角相机 (Spectator) ---
            if self.allow_spectator:
                self.spectator_camera = self.world.get_blueprint_library().find('sensor.camera.rgb')
                self.spectator_camera.set_attribute('image_size_x', f'{self.spectator_width}')
                self.spectator_camera.set_attribute('image_size_y', f'{self.spectator_height}')
                self.spectator_camera.set_attribute('fov', '100')
                # 设置位于车辆后方上方的视角
                transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-10.0))
                self.spectator_sensor = self.world.spawn_actor(self.spectator_camera, transform, attach_to=self.vehicle)
                self.spectator_sensor.listen(self._set_viewer_image)

        except RuntimeError as msg:
            pass

        self.reset()  # 初始化完成后重置环境

    # 重置环境以开始新任务
    def reset(self):
        self.episode_idx += 1
        self.num_routes_completed = -1  # 路线完成计数
        # 生成随机路线
        self.generate_route()
        self.closed = False
        self.terminate = False  # 任务是否终止
        self.success_state = False  # 是否成功完成
        self.extra_info = []  # HUD 显示的额外信息
        self.observation = self.observation_buffer = None  # 观测缓冲区
        self.viewer_image = self.viewer_image_buffer = None  # 观察者图像缓冲区
        self.step_count = 0  # 步数计数器

        # 重置指标数据
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0  # 行驶距离
        self.center_lane_deviation = 0.0  # 车道中心偏离度
        self.speed_accum = 0.0  # 速度累积
        self.routes_completed = 0.0  # 完成路线比例
        self.world.tick()  # 触发一次世界更新

        # 返回初始观测值
        time.sleep(0.2)
        obs = self.step(None)[0]
        time.sleep(0.2)
        return obs

    # 生成随机路线
    def generate_route(self):
        # 软重置（传送车辆）
        self.control.steer = float(0.0)
        self.control.throttle = float(0.0)
        self.vehicle.set_simulate_physics(False)  # 关闭物理模拟以便瞬移

        # 随机选择生成点和终点
        spawn_points_list = np.random.choice(self.map.get_spawn_points(), 2, replace=False)
        route_length = 1
        # 确保生成的路线长度大于1
        while route_length <= 1:
            self.start_wp, self.end_wp = [self.map.get_waypoint(spawn.location) for spawn in spawn_points_list]
            self.route_waypoints = compute_route_waypoints(self.map, self.start_wp, self.end_wp, resolution=1.0)
            route_length = len(self.route_waypoints)
            if route_length <= 1:
                spawn_points_list = np.random.choice(self.map.get_spawn_points(), 2, replace=False)

        self.distance_from_center_history = deque(maxlen=30)  # 历史偏离记录
        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        self.vehicle.set_transform(self.start_wp.transform)  # 移动车辆到起点
        time.sleep(0.2)
        self.vehicle.set_simulate_physics(True)  # 重新开启物理模拟

    # 执行一步环境交互
    def step(self, action):
        if action is not None:
            # 如果到达路线终点，生成新路线
            if self.current_waypoint_index >= len(self.route_waypoints) - 1:
                self.success_state = True

            throttle, steer = [float(a) for a in action]
            # 执行动作（带平滑处理）
            self.control.throttle = smooth_action(self.control.throttle, throttle, self.action_smoothing)
            self.control.steer = smooth_action(self.control.steer, steer, self.action_smoothing)
            self.vehicle.apply_control(self.control)  # 应用车辆控制

        self.world.tick()  # 触发世界更新（同步模式关键）

        # 获取最新的观测和图像
        self.observation = self._get_observation()
        if self.allow_spectator:
            self.viewer_image = self._get_viewer_image()

        # 获取车辆当前变换矩阵
        transform = self.vehicle.get_transform()

        # --- 路线追踪逻辑 ---
        # 寻找车辆当前最接近的路线点索引
        self.prev_waypoint_index = self.current_waypoint_index
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            # 计算车辆位置相对于路线点的方向向量点积
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0:  # 如果车辆超过了该点
                waypoint_index += 1
            else:
                break
        self.current_waypoint_index = waypoint_index

        # 检查路线完成情况并更新指标
        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
                (self.current_waypoint_index + 1) % len(self.route_waypoints)]
            self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
                self.current_waypoint_index % len(self.route_waypoints)]
            self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(
                self.route_waypoints)

            # 计算车道中心偏离度
            self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                         vector(self.next_waypoint.transform.location),
                                                         vector(transform.location))
            self.center_lane_deviation += self.distance_from_center

        # 计算行驶距离和累积速度
        if action is not None:
            self.distance_traveled += self.previous_location.distance(transform.location)
            self.previous_location = transform.location
        self.speed_accum += self.get_vehicle_lon_speed()  # 累积纵向速度

        # 终止条件：超过最大距离
        if self.distance_traveled >= self.max_distance and not self.eval:
            self.success_state = True

        # 更新历史偏离记录
        self.distance_from_center_history.append(self.distance_from_center)

        # 调用奖励函数计算奖励
        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward
        self.step_count += 1

        # 处理渲染和用户输入
        if self.allow_render:
            pygame.event.pump()
            if pygame.key.get_pressed()[pygame.K_ESCAPE]:
                self.close()
                self.terminate = True
        self.render()

        # 构建返回信息字典
        info = {
            "closed": self.closed,
            'total_reward': self.total_reward,
            'routes_completed': self.routes_completed,
            'total_distance': self.distance_traveled,
            'avg_center_dev': (self.center_lane_deviation / self.step_count),
            'avg_speed': (self.speed_accum / self.step_count),
            'mean_reward': (self.total_reward / self.step_count)
        }
        return self.get_semantic_image(self.observation), self.last_reward, self.terminate or self.success_state, info

    # 关闭环境，清理资源
    def close(self):
        pygame.quit()
        if self.world is not None:
            self.world.destroy()  # 销毁所有生成的 Actor
        self.closed = True

    # 渲染函数
    def render(self, mode="human"):
        self.clock.tick()
        self.hud.tick(self.world, self.clock)

        # 准备 HUD 显示的额外信息
        self.extra_info.extend([
            "Episode {}".format(self.episode_idx),
            "Reward: % 19.2f" % self.last_reward,
            "",
            "Routes completed: % 7.2f" % self.routes_completed,
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance: % 7.2f m" % self.distance_from_center,
            "Avg center dev: % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed: % 7.2f km/h" % (self.speed_accum / self.step_count),
            "Total reward: % 7.2f" % self.total_reward,
        ])

        if self.allow_spectator:
            # 绘制路线路径
            self.viewer_image = self._draw_path(self.spectator_camera, self.viewer_image)
            # 将图像显示到 Pygame 窗口
            self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

            # 在屏幕右上角叠加显示观测图像
            obs_h, obs_w = self.observation.height, self.observation.width
            pos_observation = (self.display.get_size()[0] - obs_w - 10, 10)
            self.display.blit(pygame.surfarray.make_surface(self.get_semantic_image(self.observation).swapaxes(0, 1)),
                              pos_observation)

        # 渲染 HUD 并更新屏幕
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []  # 清空额外信息
        pygame.display.flip()

    # 获取车辆纵向速度
    def get_vehicle_lon_speed(self):
        carla_velocity_vec3 = self.vehicle.get_velocity()
        vec4 = np.array([carla_velocity_vec3.x, carla_velocity_vec3.y, carla_velocity_vec3.z, 1]).reshape(4, 1)
        carla_trans = np.array(self.vehicle.get_transform().get_matrix())
        carla_trans.reshape(4, 4)
        carla_trans[0:3, 3] = 0.0
        vel_in_vehicle = np.linalg.inv(carla_trans) @ vec4
        return vel_in_vehicle[0]

    # 将 CARLA 图像数据转换为 OpenCV 格式 (RGB)
    def get_rgb_image(self, input):
        image = np.frombuffer(input.raw_data, dtype=np.uint8)
        image = image.reshape((input.height, input.width, 4))
        image = image[:, :, :3]  # 去除 Alpha 通道
        image = image[:, :, ::-1].copy()  # RGB 转 BGR (OpenCV 格式)
        return image

    # 将语义分割图像数据转换为可视化颜色
    def get_semantic_image(self, input):
        image = np.frombuffer(input.raw_data, dtype=np.uint8)
        image = image.reshape((input.height, input.width, 4))
        image = image[:, :, 2]  # 提取标签通道
        # 定义各类别的颜色映射
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
        result = np.zeros((image.shape[0], image.shape[1], 3))
        for key, value in classes.items():
            result[np.where(image == key)] = value
        return result

    # 销毁所有 Agent (传感器等)
    def _destroy_agents(self):
        for actor in self.actor_list:
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()
            if actor.is_alive:
                actor.destroy()
        self.actor_list = []

    # 碰撞回调函数
    def _collision_data(self, event):
        if get_actor_display_name(event.other_actor) != "Road":
            self.terminate = True  # 非路面碰撞导致终止
        if self.allow_render:
            self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

    # 车道入侵回调函数
    def _lane_invasion_data(self, event):
        self.terminate = True  # 越线即终止
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        if self.allow_render:
            self.hud.notification("Crossed line %s" % " and ".join(text))

    # 获取观测数据
    def _get_observation(self):
        while self.observation_buffer is None:
            pass
        obs = self.observation_buffer
        self.observation_buffer = None
        return obs

    # 获取观察者图像
    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer
        self.viewer_image_buffer = None
        return image

    # 获取随机起始位置
    def _get_start_transform(self):
        return random.choice(self.map.get_spawn_points())

    # 设置观测图像缓冲区
    def _set_observation_image(self, image):
        self.observation_buffer = image

    # 设置观察者图像缓冲区
    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    # 在图像上绘制路线路径
    def _draw_path(self, camera, image):
        """ 使用单应性矩阵在图像上绘制从路线起点到终点的连接路径。 """
        vehicle_vector = vector(self.vehicle.get_transform().location)
        world_2_camera = np.array(image.transform.get_inverse_matrix())
        image_w = int(image.height)
        image_h = int(image.width)
        fov = float(image.fov)
        image = self.get_rgb_image(image)

        for i in range(self.current_waypoint_index, len(self.route_waypoints)):
            waypoint_location = self.route_waypoints[i][0].transform.location + carla.Location(z=1.25)
            waypoint_vector = vector(waypoint_location)

            # 过滤距离过近或过远的点
            if not (2 < abs(np.linalg.norm(vehicle_vector - waypoint_vector)) < 50):
                continue

            # 计算投影矩阵将 3D 点投影到 2D 图像平面
            K = build_projection_matrix(image_h, image_w, fov)
            x, y = get_image_point(waypoint_location, K, world_2_camera)

            # 终点用红色，其他点用蓝色
            if i == len(self.route_waypoints) - 1:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            image = cv2.circle(image, (x, y), radius=3, color=color, thickness=-1)
        return image