import carla
import time
import math
import numpy as np
import cv2
import queue
import random

# ======================== 核心配置（车道硬约束+细障碍物检测+红绿灯）========================
TARGET_SPEED_KMH = 10.0  # 更低速，确保车道纠偏反应时间
TARGET_SPEED_MPS = TARGET_SPEED_KMH / 3.6
SYNC_FPS = 20

# 障碍物检测（针对电线杆等细障碍物）
LIDAR_RANGE = 15.0  # 覆盖路边障碍物
OBSTACLE_EMERGENCY_DIST = 1.0  # 1米紧急避障
OBSTACLE_WARNING_DIST = 2.5  # 提前预警
DETECT_THRESHOLD = 2  # 仅需2个点（细障碍物点少）
# 车道硬约束（核心：防止偏离撞路边障碍物）
LANE_BOUNDARY_STRICT = 0.8  # 车道边界强制纠偏力度
LANE_CENTER_BIAS = 0.1  # 轻微偏向车道中心
MAX_LANE_DEVIATION = 0.5  # 最大允许偏离车道0.5米
# 红绿灯配置
TRAFFIC_LIGHT_RANGE = 20.0  # 检测20米内的交通信号灯
YELLOW_LIGHT_SPEED_KMH = 5.0  # 黄灯时限速5km/h
YELLOW_LIGHT_SPEED_MPS = YELLOW_LIGHT_SPEED_KMH / 3.6
VISUALIZATION = True


# ======================== PID速度控制器 =========================
class SimplePID:
    def __init__(self):
        self.kp = 0.3
        self.ki = 0.008
        self.kd = 0.02
        self.error_sum = 0.0
        self.last_error = 0.0

    def update(self, current_speed):
        error = TARGET_SPEED_MPS - current_speed
        self.error_sum += error * (1 / SYNC_FPS)
        self.error_sum = np.clip(self.error_sum, -0.8, 0.8)
        derivative = (error - self.last_error) * SYNC_FPS
        self.last_error = error

        throttle = self.kp * error + self.ki * self.error_sum + self.kd * derivative
        brake = 0.0 if error > -0.1 else 0.15
        return np.clip(throttle, 0.0, 1.0), brake


# ======================== 车道边界+细障碍物+红绿灯检测 =========================
class LaneBoundaryDetector:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()

        # 障碍物状态（细障碍物专用）
        self.has_obstacle = False
        self.obs_distance = float('inf')
        self.obs_direction = 0.0
        # 车道边界状态（核心：防止撞路边障碍物）
        self.lane_deviation = 0.0  # 偏离车道中心线距离（米）
        self.lane_steer_correction = 0.0  # 车道纠偏转向
        self.is_near_lane_edge = False  # 是否靠近车道边缘
        # 红绿灯状态（新增核心）
        self.traffic_light = None  # 当前车辆需遵守的交通灯
        self.traffic_light_state = carla.TrafficLightState.Unknown  # 红绿灯状态
        self.traffic_light_distance = float('inf')  # 到红绿灯的距离

        self.frame_queue = queue.Queue(maxsize=1) if VISUALIZATION else None

        # LiDAR（针对细障碍物优化：高密度+宽视野）
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(LIDAR_RANGE))
        lidar_bp.set_attribute('points_per_second', '25000')  # 超高密度，捕捉细障碍物
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('horizontal_fov', '90')  # 覆盖车道两侧
        lidar_bp.set_attribute('noise_stddev', '0.0')
        self.lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=1.5, z=1.2)), attach_to=vehicle)
        self.lidar.listen(self._lidar_callback)

        # 摄像头
        if VISUALIZATION:
            cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', '640')
            cam_bp.set_attribute('image_size_y', '480')
            self.cam = world.spawn_actor(cam_bp,
                                         carla.Transform(carla.Location(x=2.0, z=1.8), carla.Rotation(pitch=-8)),
                                         attach_to=vehicle)
            self.cam.listen(self._cam_callback)

    def _lidar_callback(self, data):
        """检测细障碍物（电线杆/路障/护栏等）"""
        points = np.frombuffer(data.raw_data, np.float32).reshape(-1, 4)[:, :3]
        vehicle_loc = self.vehicle.get_transform().location
        yaw = math.radians(self.vehicle.get_transform().rotation.yaw)

        # 车辆本地坐标系
        x_w = points[:, 0] - vehicle_loc.x
        y_w = points[:, 1] - vehicle_loc.y
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x_local = x_w * cos_yaw + y_w * sin_yaw
        y_local = -x_w * sin_yaw + y_w * cos_yaw

        # 过滤：覆盖车道两侧（左右4米），捕捉路边障碍物
        mask = (
                (x_local > 0.3) & (x_local < LIDAR_RANGE) &
                (abs(y_local) < 4.0) &  # 车道两侧各4米
                (points[:, 2] > 0.0) & (points[:, 2] < 4.0)  # 障碍物高度0-4米
        )
        valid_points = points[mask]

        if len(valid_points) >= DETECT_THRESHOLD:
            dists = np.sqrt((valid_points[:, 0] - vehicle_loc.x) ** 2 + (valid_points[:, 1] - vehicle_loc.y) ** 2)
            self.obs_distance = np.min(dists)
            self.has_obstacle = self.obs_distance < OBSTACLE_WARNING_DIST
            if self.has_obstacle:
                min_idx = np.argmin(dists)
                min_y_local = y_local[mask][min_idx]
                self.obs_direction = 1.0 if min_y_local > 0 else -1.0
        else:
            self.has_obstacle = False
            self.obs_distance = float('inf')

    def check_lane_boundary(self):
        """车道边界硬约束：计算偏离度，强制拉回中心"""
        vehicle_loc = self.vehicle.get_transform().location
        # 获取当前车道的中心线和边界
        current_waypoint = self.map.get_waypoint(vehicle_loc, project_to_road=True)
        lane_width = current_waypoint.lane_width  # 车道宽度（米）

        # 计算车辆到车道中心线的横向距离（偏离度）
        lane_center = current_waypoint.transform.location
        # 转换为车辆本地坐标系的横向距离（y轴）
        y_diff = (lane_center.y - vehicle_loc.y) * math.cos(math.radians(current_waypoint.transform.rotation.yaw)) - \
                 (lane_center.x - vehicle_loc.x) * math.sin(math.radians(current_waypoint.transform.rotation.yaw))
        self.lane_deviation = y_diff

        # 判断是否靠近车道边缘
        self.is_near_lane_edge = abs(self.lane_deviation) > (lane_width / 2 - MAX_LANE_DEVIATION)

        # 强制纠偏转向：偏离越多，纠偏力度越大
        if self.is_near_lane_edge:
            # 靠近边缘时，强力拉回中心
            self.lane_steer_correction = np.clip(self.lane_deviation / (lane_width / 4), -LANE_BOUNDARY_STRICT,
                                                 LANE_BOUNDARY_STRICT)
        else:
            # 轻微偏离时，柔和纠偏
            self.lane_steer_correction = np.clip(self.lane_deviation / (lane_width / 2), -0.3, 0.3) + LANE_CENTER_BIAS

    def check_traffic_light(self):
        """新增：检测当前车辆需要遵守的交通信号灯及状态"""
        # 获取车辆当前需遵守的交通灯（Carla原生API，自动匹配车道对应的信号灯）
        self.traffic_light = self.vehicle.get_traffic_light()

        if self.traffic_light is not None:
            # 计算车辆到交通灯的直线距离
            tl_loc = self.traffic_light.get_transform().location
            vehicle_loc = self.vehicle.get_transform().location
            self.traffic_light_distance = math.hypot(
                tl_loc.x - vehicle_loc.x,
                tl_loc.y - vehicle_loc.y
            )

            # 仅处理20米内的有效交通灯
            if self.traffic_light_distance < TRAFFIC_LIGHT_RANGE:
                self.traffic_light_state = self.traffic_light.state
            else:
                self.traffic_light_state = carla.TrafficLightState.Unknown
        else:
            # 无交通灯或不在路口
            self.traffic_light_state = carla.TrafficLightState.Unknown
            self.traffic_light_distance = float('inf')

    def _cam_callback(self, data):
        frame = np.frombuffer(data.raw_data, np.uint8).reshape(data.height, data.width, 4)[:, :, :3].copy()
        if not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        self.frame_queue.put(frame, block=False)

    def draw_status(self):
        """新增：可视化添加红绿灯状态显示"""
        if not VISUALIZATION:
            return
        try:
            frame = self.frame_queue.get(timeout=0.01)
            speed = math.hypot(self.vehicle.get_velocity().x, self.vehicle.get_velocity().y) * 3.6

            # 原有状态显示
            cv2.putText(frame, f"Speed: {speed:.1f}km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Lane Deviation: {self.lane_deviation:.2f}m", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if self.is_near_lane_edge else (0, 255, 0), 2)
            cv2.putText(frame, f"Obs Dist: {self.obs_distance:.2f}m", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)

            # 新增：红绿灯状态显示（不同颜色区分）
            tl_color = (255, 255, 255)  # 默认白色（未知）
            if self.traffic_light_state == carla.TrafficLightState.Red:
                tl_color = (0, 0, 255)
            elif self.traffic_light_state == carla.TrafficLightState.Yellow:
                tl_color = (0, 255, 255)
            elif self.traffic_light_state == carla.TrafficLightState.Green:
                tl_color = (0, 255, 0)

            cv2.putText(frame, f"Traffic Light: {self.traffic_light_state.name}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, tl_color, 2)
            cv2.putText(frame, f"TL Distance: {self.traffic_light_distance:.2f}m", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

            cv2.imshow("Lane & Obstacle & Traffic Light Detection", frame)
            cv2.waitKey(1)
        except:
            pass

    def destroy(self):
        self.lidar.stop()
        self.lidar.destroy()
        if VISUALIZATION:
            self.cam.stop()
            self.cam.destroy()
            cv2.destroyAllWindows()


# ======================== 修复：动态生成路边障碍物（适配所有Carla版本）========================
def spawn_roadside_obstacle(world, vehicle):
    """
    动态生成路边障碍物（适配所有Carla版本）：
    1. 优先找电线杆/路灯，找不到则用路障（static.prop.streetbarrier，所有版本都有）
    2. 生成在车道边缘，测试避障
    """
    bp_lib = world.get_blueprint_library()
    # 定义优先级列表：优先细障碍物，兜底用路障
    obstacle_blueprints = [
        'static.prop.pole',
        'static.prop.streetlight',
        'static.prop.streetbarrier',  # 兜底：所有版本都有
        'static.prop.trafficcone',
        'static.prop.barrier'
    ]

    # 查找可用的蓝图
    obstacle_bp = None
    for bp_name in obstacle_blueprints:
        try:
            obstacle_bp = bp_lib.find(bp_name)
            print(f"✅ 找到可用障碍物蓝图：{bp_name}")
            break
        except IndexError:
            continue

    if obstacle_bp is None:
        # 终极兜底：随机选一个静态道具
        static_bps = [bp for bp in bp_lib if bp.id.startswith('static.prop.')]
        if static_bps:
            obstacle_bp = random.choice(static_bps)
            print(f"✅ 使用随机静态道具：{obstacle_bp.id}")
        else:
            raise RuntimeError("❌ 没有找到任何静态障碍物蓝图！")

    # 在车道右侧边缘0.5米处生成（前方10米）
    current_waypoint = world.get_map().get_waypoint(vehicle.get_transform().location)
    lane_width = current_waypoint.lane_width
    pole_waypoint = current_waypoint.next(10.0)[0]
    # 车道边缘位置（右侧0.5米）
    obstacle_loc = pole_waypoint.transform.location + pole_waypoint.transform.get_right_vector() * (
                lane_width / 2 + 0.5)
    obstacle_loc.z += 0.2  # 离地高度，避免碰撞

    # 生成障碍物（增加重试）
    for attempt in range(2):
        try:
            obstacle = world.spawn_actor(obstacle_bp, carla.Transform(obstacle_loc))
            print(f"✅ 生成路边障碍物：车道右侧0.5米，前方10米处（{obstacle_loc.x:.1f}, {obstacle_loc.y:.1f}）")
            return [obstacle]
        except RuntimeError as e:
            if "collision" in str(e).lower():
                # 微调位置避免碰撞
                obstacle_loc.x += 0.5
                obstacle_loc.y += 0.5
                continue
            else:
                raise e
    raise RuntimeError("❌ 障碍物生成失败（位置碰撞）")


# ======================== 安全生成车辆（解决碰撞问题）========================
def spawn_vehicle_safely(world, bp):
    """
    安全生成车辆，避免碰撞：
    1. 筛选无碰撞的生成点
    2. 重试机制
    3. 自定义安全位置兜底
    """
    spawn_points = world.get_map().get_spawn_points()
    # 重试3次生成
    for attempt in range(3):
        if spawn_points:
            # 随机选择生成点，优先选车道中心的
            random.shuffle(spawn_points)
            for spawn_point in spawn_points:
                try:
                    # 检查生成点是否在行驶车道上
                    wp = world.get_map().get_waypoint(spawn_point.location)
                    if wp.lane_type != carla.LaneType.Driving:
                        continue
                    # 尝试生成车辆
                    vehicle = world.spawn_actor(bp, spawn_point)
                    print(
                        f"✅ 第{attempt + 1}次尝试：成功生成车辆（位置：{spawn_point.location.x:.1f}, {spawn_point.location.y:.1f}）")
                    return vehicle
                except RuntimeError as e:
                    if "collision" in str(e).lower():
                        continue
                    else:
                        raise e
        else:
            # 无默认生成点，使用自定义安全位置
            safe_loc = carla.Location(x=100.0, y=100.0, z=0.5)  # 自定义远离建筑的位置
            safe_transform = carla.Transform(safe_loc, carla.Rotation(yaw=0))
            try:
                vehicle = world.spawn_actor(bp, safe_transform)
                print(f"✅ 使用自定义安全位置生成车辆（位置：{safe_loc.x:.1f}, {safe_loc.y:.1f}）")
                return vehicle
            except RuntimeError as e:
                print(f"❌ 自定义位置生成失败：{e}")
                attempt += 1
    raise RuntimeError("❌ 所有生成点都有碰撞，无法生成车辆！")


# ======================== 核心逻辑（红绿灯 > 车道硬约束 > 障碍物避障）========================
def main():
    # 1. 连接Carla
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1 / SYNC_FPS
    world.apply_settings(settings)

    # 2. 清理所有残留
    for actor in world.get_actors():
        if actor.type_id in ['vehicle.*', 'sensor.*', 'static.prop.*']:
            actor.destroy()
    time.sleep(1)

    # 3. 生成车辆（安全生成，避免碰撞）
    bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    bp.set_attribute('color', '255,0,0')
    try:
        vehicle = spawn_vehicle_safely(world, bp)
    except RuntimeError as e:
        print(e)
        return
    vehicle.set_simulate_physics(True)
    vehicle.set_autopilot(False)

    # 4. 生成路边障碍物（核心修复：适配所有版本）
    try:
        obstacles = spawn_roadside_obstacle(world, vehicle)
    except RuntimeError as e:
        print(e)
        # 销毁车辆后退出
        vehicle.destroy()
        return

    # 5. 第三人称视角（清晰看车道+障碍物+红绿灯）
    spectator = world.get_spectator()

    def third_person_view():
        trans = vehicle.get_transform()
        spectator_loc = trans.location - trans.get_forward_vector() * 4.5 + carla.Location(
            z=2.5) + trans.get_right_vector() * 0.5
        spectator_rot = carla.Rotation(pitch=-20, yaw=trans.rotation.yaw, roll=0)
        spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))

    # 6. 初始化检测器和控制器
    detector = LaneBoundaryDetector(world, vehicle)
    pid = SimplePID()
    current_steer = 0.0

    # 7. 核心行驶循环（新增红绿灯逻辑）
    print("\n🚗 开始测试：红绿灯规则 > 车道硬约束 > 路边障碍物避障")
    print("核心规则：红灯停，黄灯减速，绿灯行；严格贴车道中心，1米内避开障碍物")
    print("按Ctrl+C停止\n")
    try:
        while True:
            world.tick()
            third_person_view()

            # 1. 检测车道边界 + 红绿灯（核心新增）
            detector.check_lane_boundary()
            detector.check_traffic_light()

            # 2. 速度控制（基础PID）
            current_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
            base_throttle, base_brake = pid.update(current_speed)

            # 3. 核心控制逻辑（优先级：红绿灯 > 车道边缘 > 紧急避障 > 预警避障）
            target_steer = 0.0
            throttle = base_throttle
            brake = base_brake

            # 【最高优先级】红绿灯处理
            if detector.traffic_light_state == carla.TrafficLightState.Red:
                # 红灯：强制刹车停稳
                print(f"🔴 红灯！距离{detector.traffic_light_distance:.2f}米 | 停车等待", end='\r')
                brake = 1.0
                throttle = 0.0
                target_steer = 0.0  # 停车时保持方向

            elif detector.traffic_light_state == carla.TrafficLightState.Yellow:
                # 黄灯：减速到5km/h，同时遵守车道约束
                print(f"🟡 黄灯！距离{detector.traffic_light_distance:.2f}米 | 减速慢行", end='\r')
                yellow_error = YELLOW_LIGHT_SPEED_MPS - current_speed
                if yellow_error > 0:
                    throttle = np.clip(yellow_error * 0.5, 0.0, 0.3)
                    brake = 0.0
                else:
                    throttle = 0.0
                    brake = np.clip(-yellow_error * 0.5, 0.0, 0.3)
                target_steer = detector.lane_steer_correction

            # 【次优先级】车道边缘约束
            elif detector.is_near_lane_edge:
                print(f"🔴 靠近车道边缘！偏离{detector.lane_deviation:.2f}米 | 强制拉回中心", end='\r')
                target_steer = detector.lane_steer_correction
                throttle *= 0.1  # 降速纠偏

            # 【次优先级】紧急避障
            elif detector.obs_distance < OBSTACLE_EMERGENCY_DIST:
                print(f"⚠️ 紧急避障：距离障碍物{detector.obs_distance:.2f}米 | 贴车道绕开", end='\r')
                brake = 1.0
                throttle = 0.0
                target_steer = (-detector.obs_direction * 0.6) + detector.lane_steer_correction

            # 【次优先级】预警避障
            elif detector.has_obstacle:
                print(f"🔶 预警避障：距离障碍物{detector.obs_distance:.2f}米 | 顺车道绕开", end='\r')
                throttle *= 0.2
                target_steer = (-detector.obs_direction * 0.3) + detector.lane_steer_correction

            # 【正常状态】绿灯/无红绿灯，严格贴车道中心
            else:
                light_status = "🟢 绿灯" if detector.traffic_light_state == carla.TrafficLightState.Green else "🚦 无信号灯"
                print(
                    f"✅ 正常行驶：{light_status} | 车道偏离{detector.lane_deviation:.2f}米 | 速度{current_speed * 3.6:.1f}km/h",
                    end='\r')
                target_steer = detector.lane_steer_correction

            # 转向平滑+硬限制（防止急转弯）
            current_steer += (target_steer - current_steer) * 0.25
            current_steer = np.clip(current_steer, -0.7, 0.7)

            # 下发车辆控制指令
            vehicle.apply_control(carla.VehicleControl(
                throttle=throttle, steer=current_steer, brake=brake, hand_brake=False
            ))

            # 可视化显示（含红绿灯状态）
            detector.draw_status()

    except KeyboardInterrupt:
        print("\n\n🛑 测试停止，清理资源...")
    finally:
        # 清理所有资源
        detector.destroy()
        vehicle.destroy()
        for obs in obstacles:
            obs.destroy()
        # 恢复Carla设置
        settings.synchronous_mode = False
        world.apply_settings(settings)
        cv2.destroyAllWindows()
        print("✅ 清理完成！")


if __name__ == "__main__":
    main()