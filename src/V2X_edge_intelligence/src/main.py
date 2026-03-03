import time
import math
import numpy as np
import cv2
import queue
import random

# ======================== 核心配置（车道硬约束+细障碍物检测）========================
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


# ======================== 车道边界检测+细障碍物识别 =========================
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

    def _cam_callback(self, data):
        frame = np.frombuffer(data.raw_data, np.uint8).reshape(data.height, data.width, 4)[:, :, :3].copy()
        if not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        self.frame_queue.put(frame, block=False)

    def draw_status(self):
        if not VISUALIZATION:
            return
        try:
            frame = self.frame_queue.get(timeout=0.01)
            speed = math.hypot(self.vehicle.get_velocity().x, self.vehicle.get_velocity().y) * 3.6
            # 叠加车道偏离+障碍物检测状态
            cv2.putText(frame, f"Speed: {speed:.1f}km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Lane Deviation: {self.lane_deviation:.2f}m", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if self.is_near_lane_edge else (0, 255, 0), 2)
            cv2.putText(frame, f"Obs Dist: {self.obs_distance:.2f}m", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)
            cv2.imshow("Lane & Obstacle Detection", frame)
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


# ======================== 核心逻辑（车道硬约束+障碍物避障）========================
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

    # 5. 第三人称视角（清晰看车道+障碍物）
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

    # 7. 核心行驶循环
    print("\n🚗 开始测试：车道硬约束 + 路边障碍物避障")
    print("核心规则：严格贴车道中心，1米内避开障碍物，零碰撞")
    print("按Ctrl+C停止\n")
    try:
        while True:
            world.tick()
            third_person_view()

            # 1. 检测车道边界（优先级最高）
            detector.check_lane_boundary()

            # 2. 速度控制
            current_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
            throttle, brake = pid.update(current_speed)

            # 3. 转向逻辑：车道硬约束 > 1米紧急避障 > 预警避障
            target_steer = 0.0
            if detector.is_near_lane_edge:
                # 靠近车道边缘：强制拉回中心
                print(f"🔴 靠近车道边缘！偏离{detector.lane_deviation:.2f}米 | 强制拉回中心", end='\r')
                target_steer = detector.lane_steer_correction
                throttle *= 0.1  # 降速纠偏
            elif detector.obs_distance < OBSTACLE_EMERGENCY_DIST:
                # 1米内障碍物：紧急避障+车道约束
                print(f"⚠️ 紧急避障：距离障碍物{detector.obs_distance:.2f}米 | 贴车道绕开", end='\r')
                brake = 1.0
                throttle = 0.0
                # 避障+车道纠偏：既绕开又不越线
                target_steer = (-detector.obs_direction * 0.6) + detector.lane_steer_correction
            elif detector.has_obstacle:
                # 预警避障：贴车道绕行
                print(f"🔶 预警避障：距离障碍物{detector.obs_distance:.2f}米 | 顺车道绕开", end='\r')
                throttle *= 0.2
                target_steer = (-detector.obs_direction * 0.3) + detector.lane_steer_correction
            else:
                # 正常行驶：严格贴车道中心
                print(f"✅ 正常行驶：车道偏离{detector.lane_deviation:.2f}米 | 速度{current_speed * 3.6:.1f}km/h",
                      end='\r')
                target_steer = detector.lane_steer_correction

            # 转向平滑+硬限制
            current_steer += (target_steer - current_steer) * 0.25
            current_steer = np.clip(current_steer, -0.7, 0.7)

            # 下发控制
            vehicle.apply_control(carla.VehicleControl(
                throttle=throttle, steer=current_steer, brake=brake, hand_brake=False
            ))

            # 可视化
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