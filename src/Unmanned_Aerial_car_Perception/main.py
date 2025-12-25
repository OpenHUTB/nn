import carla
import time
import math
import numpy as np
import cv2
import threading
import queue
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field


# ======================== 全局配置（核心参数可直接调整）========================
@dataclass
class Config:
    # 核心匀速参数（目标50km/h）
    TARGET_SPEED_KMH: float = 50.0
    TARGET_SPEED_MPS: float = 50.0 / 3.6  # 转换为米/秒（≈13.89）
    SPEED_DEADZONE: float = 0.2  # 速度死区（±0.2km/h，避免频繁调整）

    # PID速度控制器参数（优化50km/h匀速）
    PID_KP_LOW: float = 0.2  # 低速段比例项（<40km/h）
    PID_KP_MID: float = 0.15  # 中速段比例项（40-50km/h）
    PID_KP_HIGH: float = 0.1  # 高速段比例项（>50km/h）
    PID_KI: float = 0.005  # 积分项（消除稳态误差）
    PID_KD: float = 0.03  # 微分项（抑制超调）
    PID_INTEGRAL_LIMIT: float = 0.8  # 积分饱和限制
    PID_INTEGRAL_RESET_THRESH: float = 1.0  # 误差超1km/h重置积分

    # 障碍物避障参数
    LIDAR_RANGE: float = 8.0  # LiDAR检测范围（米）
    OBSTACLE_EMERGENCY_DIST: float = 2.0  # 紧急制动距离（<2米刹车）
    OBSTACLE_WARNING_DIST: float = 4.0  # 避障预警距离（<4米转向）
    OBSTACLE_ANGLE_THRESHOLD: float = 45  # 检测角度（前方45°）
    AVOID_STEER_MAX: float = 0.3  # 最大避障转向角（0-1，1为最大）
    STEER_SMOOTH_FACTOR: float = 0.1  # 转向平滑因子（越大越灵敏）
    STEER_RETURN_FACTOR: float = 0.05  # 避障后回正因子

    # 传感器参数（降负载，避免卡顿）
    LIDAR_POINTS_PER_SECOND: int = 20000  # LiDAR点云数量（降负载）
    CAMERA_RESOLUTION: Tuple[int, int] = (480, 360)  # 摄像头分辨率
    PERCEPTION_FREQ: int = 10  # 感知频率（Hz）
    SYNC_FPS: int = 20  # 同步帧率（降负载）
    VISUALIZATION_ENABLE: bool = True  # 可视化开关（True=显示窗口）

    # 基础运行参数
    DRIVE_DURATION: int = 120  # 行驶时长（秒）
    CARLA_PORTS: List[int] = field(default_factory=lambda: [2000, 2001, 2002])
    PREFERRED_VEHICLES: List[str] = field(
        default_factory=lambda: ["vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.bmw.grandtourer"])


CONFIG = Config()


# ======================== 速度滤波：指数平滑+滑动平均 ========================
class EnhancedSpeedFilter:
    def __init__(self, initial_speed: float = 0.0):
        self.smoothed_speed = initial_speed
        self.speed_history = []
        self.window_size = 6  # 滑动窗口大小

    def update(self, measured_speed: float) -> float:
        # 指数平滑（降低瞬时波动）
        self.smoothed_speed = 0.3 * measured_speed + 0.7 * self.smoothed_speed
        # 滑动平均（进一步稳定）
        self.speed_history.append(self.smoothed_speed)
        if len(self.speed_history) > self.window_size:
            self.speed_history.pop(0)
        return np.mean(self.speed_history) if self.speed_history else measured_speed


# ======================== PID速度控制器（精准50km/h）========================
class DynamicSpeedController:
    def __init__(self):
        self.target_speed = CONFIG.TARGET_SPEED_MPS
        self.last_error = 0.0
        self.error_integral = 0.0
        self.speed_filter = EnhancedSpeedFilter()

    def _get_dynamic_kp(self, current_speed_mps: float) -> float:
        """根据当前速度动态调整KP，避免超调"""
        current_kmh = current_speed_mps * 3.6
        if current_kmh < 40:
            return CONFIG.PID_KP_LOW
        elif 40 <= current_kmh <= 50:
            return CONFIG.PID_KP_MID
        else:
            return CONFIG.PID_KP_HIGH

    def update(self, current_speed_mps: float, dt: float = 1 / CONFIG.SYNC_FPS) -> Tuple[float, float]:
        # 速度滤波（稳定输入）
        filtered_speed = self.speed_filter.update(current_speed_mps)
        # 计算误差（米/秒）
        error = self.target_speed - filtered_speed
        error_kmh = error * 3.6

        # 积分项（消除稳态误差，避免速度飘移）
        if abs(error_kmh) < CONFIG.PID_INTEGRAL_RESET_THRESH:
            self.error_integral += error * dt
        else:
            self.error_integral = 0.0  # 误差过大重置积分
        self.error_integral = np.clip(self.error_integral, -CONFIG.PID_INTEGRAL_LIMIT, CONFIG.PID_INTEGRAL_LIMIT)

        # 微分项（抑制超调）
        error_derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        # 动态PID计算
        kp = self._get_dynamic_kp(filtered_speed)
        throttle = kp * error + CONFIG.PID_KI * self.error_integral + CONFIG.PID_KD * error_derivative
        throttle = np.clip(throttle, 0.0, 1.0)  # 油门限制0-1

        # 刹车逻辑（仅速度超目标+误差>死区时刹车）
        brake = 0.0
        if error < -CONFIG.SPEED_DEADZONE / 3.6:  # 转换为米/秒
            brake = np.clip(-kp * error * 0.4, 0.0, 1.0)
            throttle = 0.0  # 刹车时关闭油门

        return throttle, brake


# ======================== 避障感知类（自动绕开障碍物）========================
class ObstacleAvoidancePerception:
    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        self.world = world
        self.vehicle = vehicle
        self.bp_lib = world.get_blueprint_library()

        # 感知数据
        self.perception_data = {
            "lidar_points": np.array([]),
            "camera_frame": None,
            "has_obstacle": False,
            "has_emergency": False,
            "obstacle_dist": float("inf"),
            "obstacle_dir": 0.0,  # -1=左，1=右，0=正前
            "multi_obstacle": False
        }

        # 可视化线程（解决窗口未响应）
        self.frame_queue = queue.Queue(maxsize=1)
        self.draw_thread = None
        self.draw_running = False
        if CONFIG.VISUALIZATION_ENABLE:
            self.draw_running = True
            self.draw_thread = threading.Thread(target=self._draw_loop, daemon=True)
            self.draw_thread.start()

        # 初始化传感器
        self.lidar_sensor = None
        self.camera_sensor = None
        self._init_lidar()
        self._init_camera()

    def _init_lidar(self):
        """初始化LiDAR，检测前方障碍物位置（左/右/正前）"""
        try:
            lidar_bp = self.bp_lib.find('sensor.lidar.ray_cast')
            # 逐个设置LiDAR参数（修复set_attributes错误）
            lidar_bp.set_attribute('range', str(CONFIG.LIDAR_RANGE))
            lidar_bp.set_attribute('points_per_second', str(CONFIG.LIDAR_POINTS_PER_SECOND))
            lidar_bp.set_attribute('rotation_frequency', str(CONFIG.SYNC_FPS))
            lidar_bp.set_attribute('channels', '32')  # 降为32线（减少负载）
            lidar_bp.set_attribute('upper_fov', '5')
            lidar_bp.set_attribute('lower_fov', '-20')
            lidar_bp.set_attribute('noise_stddev', '0.001')
            lidar_bp.set_attribute('dropoff_general_rate', '0.005')

            # LiDAR安装位置（车辆前保险杠）
            lidar_transform = carla.Transform(carla.Location(x=1.0, z=1.2))
            self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)

            def lidar_callback(point_cloud):
                # 解析点云
                points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
                x, y, z, _ = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

                # 过滤：只保留前方45°、地面以上的点
                vehicle_yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
                point_yaw = np.arctan2(y, x)
                angle_diff = np.degrees(np.abs(point_yaw - vehicle_yaw))
                mask = (
                        (z > -0.5) & (z < 2.0) &  # 高度过滤
                        (np.hypot(x, y) > 0.3) &  # 排除车辆自身
                        (angle_diff < CONFIG.OBSTACLE_ANGLE_THRESHOLD)  # 前方角度
                )
                valid_points = points[mask][:, :3]

                self.perception_data["lidar_points"] = valid_points
                if len(valid_points) == 0:
                    # 无障碍物
                    self.perception_data.update({
                        "has_obstacle": False,
                        "has_emergency": False,
                        "obstacle_dist": float("inf"),
                        "obstacle_dir": 0.0,
                        "multi_obstacle": False
                    })
                    return

                # 计算障碍物距离和方向
                distances = np.hypot(valid_points[:, 0], valid_points[:, 1])
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                min_y = valid_points[min_dist_idx, 1]  # y<0=左，y>0=右

                # 更新感知数据
                self.perception_data["obstacle_dist"] = min_dist
                self.perception_data["has_obstacle"] = min_dist < CONFIG.OBSTACLE_WARNING_DIST
                self.perception_data["has_emergency"] = min_dist < CONFIG.OBSTACLE_EMERGENCY_DIST
                self.perception_data["multi_obstacle"] = len(valid_points) > 50
                # 障碍物方向：-1（左）/1（右），绝对值=距离越近方向越明显
                self.perception_data["obstacle_dir"] = np.sign(min_y) * (1 - min_dist / CONFIG.OBSTACLE_WARNING_DIST)

            self.lidar_sensor.listen(lidar_callback)
            print("✅ LiDAR初始化完成（障碍物检测）")
        except Exception as e:
            print(f"⚠️ LiDAR初始化失败：{e}")

    def _init_camera(self):
        """初始化摄像头，独立线程绘图（解决窗口未响应）"""
        try:
            camera_bp = self.bp_lib.find('sensor.camera.rgb')
            # 逐个设置摄像头参数
            camera_bp.set_attribute('image_size_x', str(CONFIG.CAMERA_RESOLUTION[0]))
            camera_bp.set_attribute('image_size_y', str(CONFIG.CAMERA_RESOLUTION[1]))
            camera_bp.set_attribute('fov', '110')
            camera_bp.set_attribute('sensor_tick', str(1 / CONFIG.PERCEPTION_FREQ))
            camera_bp.set_attribute('gamma', '2.2')

            # 摄像头安装位置（车辆前挡风玻璃）
            camera_transform = carla.Transform(carla.Location(x=1.2, z=1.5))
            self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

            def camera_callback(image):
                # 创建可写图像副本（修复OpenCV只读错误）
                frame = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                    (image.height, image.width, 4)
                )[:, :, :3].copy()
                self.perception_data["camera_frame"] = frame
                # 放入队列（绘图线程处理）
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame, block=False)

            self.camera_sensor.listen(camera_callback)
            print("✅ 摄像头初始化完成（独立绘图线程）")
        except Exception as e:
            print(f"⚠️ 摄像头初始化失败：{e}")

    def _draw_loop(self):
        """独立绘图线程：避免阻塞Carla同步逻辑"""
        cv2.namedWindow("Smart Perception", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Perception", CONFIG.CAMERA_RESOLUTION[0], CONFIG.CAMERA_RESOLUTION[1])
        while self.draw_running:
            try:
                frame = self.frame_queue.get(timeout=0.01)
                # 叠加关键信息
                speed_kmh = math.hypot(self.vehicle.get_velocity().x, self.vehicle.get_velocity().y) * 3.6
                cv2.putText(frame, f"Target Speed: {CONFIG.TARGET_SPEED_KMH:.1f}km/h | Current: {speed_kmh:.1f}km/h",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, f"Obstacle Dist: {self.perception_data['obstacle_dist']:.2f}m",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Obstacle Dir: {self.perception_data['obstacle_dir']:.2f} (L/R)",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Emergency: {'YES' if self.perception_data['has_emergency'] else 'NO'}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                # 刷新窗口
                cv2.imshow("Smart Perception", frame)
                cv2.waitKey(1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ 绘图线程异常：{e}")
                break

    def get_obstacle_status(self) -> Tuple[bool, bool, float, float, bool]:
        """返回：是否有障碍、是否紧急、障碍距离、障碍方向、是否多障碍"""
        return (
            self.perception_data["has_obstacle"],
            self.perception_data["has_emergency"],
            self.perception_data["obstacle_dist"],
            self.perception_data["obstacle_dir"],
            self.perception_data["multi_obstacle"]
        )

    def destroy(self):
        """销毁传感器和绘图线程"""
        self.draw_running = False
        if self.draw_thread:
            self.draw_thread.join(timeout=1.0)
        if self.lidar_sensor:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()
        if self.camera_sensor:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
        if CONFIG.VISUALIZATION_ENABLE:
            cv2.destroyWindow("Smart Perception")
        print("🗑️ 感知模块已销毁")


# ======================== 工具函数 ========================
def get_carla_client() -> Optional[Tuple[carla.Client, carla.World]]:
    """连接Carla服务器"""
    for port in CONFIG.CARLA_PORTS:
        try:
            client = carla.Client("127.0.0.1", port)
            client.set_timeout(60.0)
            world = client.get_world()
            # 设置同步模式
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / CONFIG.SYNC_FPS
            world.apply_settings(settings)
            print(f"✅ 成功连接Carla（端口：{port}）")
            return client, world
        except Exception as e:
            print(f"⚠️ 端口{port}连接失败：{str(e)[:50]}")
    return None, None


def clean_actors(world: carla.World) -> None:
    """清理残留Actor（修复ActorList相加错误）"""
    print("\n🧹 清理残留Actor...")
    # 清理车辆
    for actor in world.get_actors().filter("vehicle.*"):
        try:
            actor.destroy()
        except Exception as e:
            print(f"⚠️ 销毁车辆失败：{e}")
    # 清理传感器
    for actor in world.get_actors().filter("sensor.*"):
        try:
            actor.destroy()
        except Exception as e:
            print(f"⚠️ 销毁传感器失败：{e}")
    time.sleep(1)


def spawn_vehicle_safely(world: carla.World) -> Optional[carla.Vehicle]:
    """安全生成车辆"""
    bp_lib = world.get_blueprint_library()
    # 选择优先车辆
    vehicle_bp = None
    for vehicle_name in CONFIG.PREFERRED_VEHICLES:
        try:
            vehicle_bp = bp_lib.find(vehicle_name)
            break
        except:
            continue
    if not vehicle_bp:
        vehicle_bp = bp_lib.filter('vehicle')[0]
    vehicle_bp.set_attribute('color', '255,0,0')  # 红色车辆

    # 选择生成点
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise Exception("❌ 无可用生成点")
    spawn_point = spawn_points[1] if len(spawn_points) >= 2 else spawn_points[0]

    # 尝试生成车辆（3次重试）
    for retry in range(3):
        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            if vehicle and vehicle.is_alive:
                vehicle.set_simulate_physics(True)
                vehicle.set_autopilot(False)
                print(f"✅ 车辆生成成功（ID：{vehicle.id}）")
                return vehicle
            elif vehicle:
                vehicle.destroy()
        except Exception as e:
            print(f"⚠️ 第{retry + 1}次生成失败：{str(e)[:50]}")
            time.sleep(0.5)
    raise Exception("❌ 车辆生成失败")


def init_spectator_follow(world: carla.World, vehicle: carla.Vehicle) -> callable:
    """ spectator视角跟随车辆 """
    spectator = world.get_spectator()
    view_update_counter = 0

    def follow_vehicle():
        nonlocal view_update_counter
        if view_update_counter % 3 == 0:
            trans = vehicle.get_transform()
            # 视角位置：车辆后上方10米
            spectator.set_transform(carla.Transform(
                trans.location + carla.Location(x=-10, z=5),
                carla.Rotation(pitch=-20, yaw=trans.rotation.yaw)
            ))
        view_update_counter += 1

    follow_vehicle()
    return follow_vehicle


# ======================== 主逻辑：匀速+避障 ========================
def main():
    vehicle: Optional[carla.Vehicle] = None
    perception: Optional[ObstacleAvoidancePerception] = None
    speed_controller: Optional[DynamicSpeedController] = None
    world: Optional[carla.World] = None
    follow_vehicle = None

    try:
        # 1. 连接Carla并初始化
        client, world = get_carla_client()
        if not client or not world:
            raise Exception("❌ 未连接到Carla服务器")
        clean_actors(world)
        vehicle = spawn_vehicle_safely(world)
        follow_vehicle = init_spectator_follow(world, vehicle)


        speed_controller = DynamicSpeedController()
        perception = ObstacleAvoidancePerception(world, vehicle)


        start_time = time.time()
        current_steer = 0.0  # 当前转向角
        print(f"\n🚙 开始行驶（目标速度：{CONFIG.TARGET_SPEED_KMH}km/h，时长：{CONFIG.DRIVE_DURATION}秒）")

        # 4. 主行驶循环
        while time.time() - start_time < CONFIG.DRIVE_DURATION:
            world.tick()  # 同步Carla世界
            follow_vehicle()  # 更新视角
            dt = 1.0 / CONFIG.SYNC_FPS

            # 4.1 获取车辆速度（米/秒）
            current_vel = vehicle.get_velocity()
            current_speed_mps = math.hypot(current_vel.x, current_vel.y)

            # 4.2 获取障碍物状态
            has_obstacle, has_emergency, obs_dist, obs_dir, multi_obs = perception.get_obstacle_status()


            # 4.3 速度控制（PID）
            throttle, brake = speed_controller.update(current_speed_mps, dt)

            # 4.4 避障转向控制（临时注释这一段）
            # if has_emergency:
            #     # 紧急制动：刹车+回正
            #     brake = 1.0
            #     throttle = 0.0
            #     target_steer = 0.0
            # elif has_obstacle:
            #     # 避障转向：根据障碍物方向调整（左/右）
            #     target_steer = obs_dir * CONFIG.AVOID_STEER_MAX
            # else:
            #     # 无障碍物：转向回正
            #     target_steer = current_steer * (1 - CONFIG.STEER_RETURN_FACTOR)

            # 临时强制设置：无刹车+固定转向+油门=0.5（测试车辆是否能动）
            brake = 0.0
            throttle = 0.5
            target_steer = 0.0
            # 4.5 下发控制指令
            vehicle.apply_control(carla.VehicleControl(
                throttle=float(throttle),
                steer=float(current_steer),
                brake=float(brake),
                hand_brake=False,
                reverse=False
            ))

            # 4.6 实时打印状态（每5帧打印一次，降负载）
            if int((time.time() - start_time) * CONFIG.SYNC_FPS) % 5 == 0:
                current_speed_kmh = current_speed_mps * 3.6
                speed_error = CONFIG.TARGET_SPEED_KMH - current_speed_kmh
                print(
                    f"速度：{current_speed_kmh:.1f}km/h（误差：{speed_error:.1f}）| 转向：{current_steer:.2f} | 障碍距离：{obs_dist:.2f}m",
                    end='\r')

        # 5. 平滑停车
        print("\n🛑 到达行驶时长，开始停车...")
        for i in range(20):
            world.tick()
            brake = (i / 20) * 1.0
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=brake))
            time.sleep(0.05)

        # 6. 打印统计信息
        final_speed_kmh = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y) * 3.6
        start_loc = vehicle.get_transform().location  # 初始位置
        end_loc = vehicle.get_transform().location  # 结束位置
        travel_distance = start_loc.distance(end_loc)
        avg_speed = (travel_distance / CONFIG.DRIVE_DURATION) * 3.6 if CONFIG.DRIVE_DURATION > 0 else 0.0
        print(f"\n📊 行驶完成统计：")
        print(f"   目标速度：{CONFIG.TARGET_SPEED_KMH:.1f}km/h | 最终速度：{final_speed_kmh:.1f}km/h")
        print(f"   平均速度：{avg_speed:.1f}km/h | 行驶距离：{travel_distance:.2f}米")

    except KeyboardInterrupt:
        print("\n⚠️ 程序被用户手动中断")
    except Exception as e:
        print(f"\n❌ 程序异常：{e}")
        print("\n========== 排查指南 ==========")
        print("1. 确保Carla模拟器已启动（管理员权限），地图加载完成")
        print("2. 确保carla库版本与模拟器一致（如0.9.15对应carla==0.9.15）")
        print("3. 关闭其他占用2000端口的程序（如其他Carla实例）")
    finally:
        # 资源清理
        if perception:
            perception.destroy()
        if vehicle:
            try:
                vehicle.destroy()
                print("🗑️ 车辆已销毁")
            except Exception as e:
                print(f"⚠️ 销毁车辆失败：{e}")
        if world:
            try:
                # 恢复Carla异步模式
                settings = world.get_settings()
                settings.synchronous_mode = False
                world.apply_settings(settings)
            except Exception as e:
                print(f"⚠️ 恢复世界设置失败：{e}")
        cv2.destroyAllWindows()
        print("✅ 所有资源清理完成！")


if __name__ == "__main__":
    main()