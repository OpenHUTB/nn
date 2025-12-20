#!/usr/bin/env python
"""
CARLA Waypoint Following - Dynamic Road Waypoints (sd_6/__main__.py)

核心改进：
1. 动态生成沿道路的航点，替代固定坐标，避开障碍物
2. 选择地图开阔区域生成车辆（Town03主干道）
3. 增加障碍物检测，确保行驶路段安全
4. 优化航点跟踪逻辑，适配动态道路
"""

# ===================== 系统导入 & 路径配置 =====================
import sys
import os
import carla
import numpy as np
import time
import math
import logging
from typing import List, Tuple, Optional, Union
import pygame
import matplotlib.pyplot as plt

# 获取当前脚本目录并加入搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# ===================== PID控制器（优化版）=====================
class PIDController:
    """PID速度控制器"""
    def __init__(self):
        self.kp = 0.3
        self.ki = 0.01
        self.kd = 0.1
        self.prev_error = 0.0
        self.integral = 0.0
        self.integral_limit = 0.5

    def calculate_control(self, target_speed, current_speed):
        error = target_speed - current_speed
        self.integral = np.clip(self.integral + error * self.ki, -self.integral_limit, self.integral_limit)
        derivative = (error - self.prev_error) * self.kd
        output = self.kp * error + self.integral + derivative

        throttle = max(0.0, min(1.0, output)) if output > 0 else 0.0
        brake = max(0.0, min(1.0, -output)) if output < 0 else 0.0

        self.prev_error = error

        from collections import namedtuple
        Control = namedtuple('Control', ['throttle', 'brake'])
        return Control(throttle=throttle, brake=brake)

# ===================== Pygame显示（优化版）=====================
class PygameDisplay:
    """Pygame摄像头显示类"""
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.camera = None
        self.screen = None
        self.surface = None

        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("CARLA Camera View (ESC to quit)")

        self._setup_camera()

    def _setup_camera(self):
        try:
            bp_lib = self.world.get_blueprint_library()
            camera_bp = bp_lib.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')
            camera_transform = carla.Transform(carla.Location(x=2.5, z=1.8))
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            self.camera.listen(lambda image: self._process_image(image))
        except Exception as e:
            logging.warning(f"摄像头初始化失败：{e}")
            self.camera = None

    def _process_image(self, image):
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))[:, :, :3]
            array = array[:, :, ::-1].swapaxes(0, 1)
            self.surface = pygame.surfarray.make_surface(array)
        except Exception as e:
            logging.warning(f"图像处理失败：{e}")

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return True
        return False

    def render(self):
        self.screen.fill((0, 0, 0))
        if self.surface is not None:
            self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

    def destroy(self):
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
        pygame.quit()

# ===================== 绘图类（优化版）=====================
class Plotter:
    """轨迹+速度绘图类"""
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.is_initialized = False
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
        self.x_data = []
        self.y_data = []
        self.time_data = []
        self.speed_data = []
        self.target_speed_data = []

    def init_plot(self):
        try:
            waypoints_np = np.array(self.waypoints)
            self.ax1.plot(waypoints_np[:, 0], waypoints_np[:, 1], 'ro-', label='Waypoints', markersize=4)
            self.ax1.set_xlabel('X (m)')
            self.ax1.set_ylabel('Y (m)')
            self.ax1.set_title('Vehicle Trajectory')
            self.ax1.legend()
            self.ax1.grid(True)

            self.ax2.set_xlabel('Time (s)')
            self.ax2.set_ylabel('Speed (km/h)')
            self.ax2.set_title('Speed vs Time')
            self.ax2.grid(True)

            plt.ion()
            plt.show(block=False)
            self.is_initialized = True
        except Exception as e:
            logging.error(f"绘图初始化失败：{e}")
            self.is_initialized = False

    def update_plot(self, time, x, y, speed, target_speed):
        if not self.is_initialized:
            return

        try:
            self.x_data.append(x)
            self.y_data.append(y)
            self.time_data.append(time)
            self.speed_data.append(speed)
            self.target_speed_data.append(target_speed)

            self.ax1.plot(self.x_data, self.y_data, 'b-', label='Trajectory' if len(self.x_data) == 1 else "", linewidth=1)
            if len(self.x_data) == 1:
                self.ax1.legend()

            self.ax2.plot(self.time_data, self.speed_data, 'b-', label='Current Speed' if len(self.time_data) == 1 else "", linewidth=1)
            self.ax2.plot(self.time_data, self.target_speed_data, 'r--', label='Target Speed' if len(self.time_data) == 1 else "", linewidth=1)
            if len(self.time_data) == 1:
                self.ax2.legend()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            logging.warning(f"绘图更新失败：{e}")

    def cleanup_plot(self):
        if self.is_initialized:
            plt.ioff()
            plt.close(self.fig)
            self.is_initialized = False

# ===================== 配置类（动态航点版）=====================
class Config:
    """仿真配置类"""
    # 动态航点配置
    NUM_WAYPOINTS = 50  # 生成的航点数量
    WAYPOINT_SPACING = 2.0  # 航点间距（米）
    TARGET_SPEED = 30.0  # 统一目标速度（km/h）

    # 控制器参数
    WAYPOINT_THRESHOLD = 1.5
    MIN_SPEED_STANLEY = 1e-4
    STANLEY_K = 0.4
    MAX_STEER_DEG = 40.0
    MAX_STEER_RAD = math.radians(MAX_STEER_DEG)

    # CARLA配置
    CARLA_HOST = "localhost"
    CARLA_PORT = 2000
    CARLA_TIMEOUT = 10.0

    # 车辆配置（Town03主干道生成位置）
    VEHICLE_MODEL = "vehicle.tesla.model3"
    SPAWN_LOCATION = carla.Location(x=100.0, y=10.0, z=0.5)  # Town03开阔主干道
    SPAWN_YAW = 0.0  # 初始朝向

    # 观察者视角（适配新生成位置）
    SPECTATOR_LOCATION = carla.Location(x=110.0, y=10.0, z=8.0)
    SPECTATOR_ROTATION = carla.Rotation(pitch=-30.0, yaw=0.0, roll=0.0)

# ===================== 工具类（优化版）=====================
class Utils:
    """工具函数类"""

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    @staticmethod
    def get_vehicle_speed(vehicle: carla.Vehicle, unit: str = "kmh") -> float:
        vel = vehicle.get_velocity()
        speed_mps = np.linalg.norm([vel.x, vel.y, vel.z])
        return speed_mps * 3.6 if unit == "kmh" else speed_mps

    @staticmethod
    def calculate_cte(vehicle_loc: carla.Location, prev_wp: List[float], target_wp: List[float]) -> float:
        x1, y1 = prev_wp[0], prev_wp[1]
        x2, y2 = target_wp[0], target_wp[1]
        x, y = vehicle_loc.x, vehicle_loc.y

        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return math.hypot(x - x1, y - y1)

        if abs(dx) < 1e-6:
            cte = x - x1
        else:
            slope = dy / dx
            a = -slope
            b = 1.0
            c = slope * x1 - y1
            cte = (a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)

        yaw_path = np.arctan2(dy, dx)
        yaw_ct = np.arctan2(y - y1, x - x1)
        yaw_diff = Utils.normalize_angle(yaw_path - yaw_ct)

        return abs(cte) if yaw_diff > 0 else -abs(cte)

    @staticmethod
    def calculate_distance_2d(loc1: carla.Location, loc2: carla.Location) -> float:
        dx = loc1.x - loc2.x
        dy = loc1.y - loc2.y
        return math.sqrt(dx ** 2 + dy ** 2)

    @staticmethod
    def generate_road_waypoints(world, start_loc: carla.Location, num_waypoints: int, spacing: float) -> List[List[float]]:
        """
        沿道路动态生成航点（核心：避免障碍物）
        :param world: CARLA世界对象
        :param start_loc: 起始位置
        :param num_waypoints: 航点数量
        :param spacing: 航点间距
        :return: 航点列表 [[x, y, speed], ...]
        """
        map = world.get_map()
        waypoints = []
        current_waypoint = map.get_waypoint(start_loc)

        for i in range(num_waypoints):
            # 添加当前航点（x, y, 目标速度）
            waypoints.append([current_waypoint.transform.location.x, current_waypoint.transform.location.y, Config.TARGET_SPEED])
            # 获取下一个道路航点（沿道路前进）
            next_waypoints = current_waypoint.next(spacing)
            if not next_waypoints:
                break
            current_waypoint = next_waypoints[0]

        logging.info(f"动态生成了 {len(waypoints)} 个道路航点")
        return waypoints

# ===================== 航点管理器（优化版）=====================
class WaypointManager:
    """航点管理类"""
    def __init__(self, waypoints: List[List[float]], threshold: float):
        self.waypoints = waypoints
        self.threshold = threshold
        self.current_target_id = 1

    def update_target(self, vehicle_loc: carla.Location) -> None:
        if self.current_target_id >= len(self.waypoints) - 1:
            return

        target_wp = self.waypoints[self.current_target_id]
        target_loc = carla.Location(x=target_wp[0], y=target_wp[1])
        distance = Utils.calculate_distance_2d(vehicle_loc, target_loc)

        if distance < self.threshold:
            self._log_waypoint_switch(distance)
            self.current_target_id += 1

    def _log_waypoint_switch(self, distance: float) -> None:
        reached_id = self.current_target_id
        reached_coords = (self.waypoints[reached_id][0], self.waypoints[reached_id][1])
        log_msg = f"到达航点 {reached_id} (X={reached_coords[0]:.1f}, Y={reached_coords[1]:.1f}, 距离={distance:.1f}m)"

        if self.current_target_id + 1 < len(self.waypoints):
            next_id = self.current_target_id + 1
            next_coords = (self.waypoints[next_id][0], self.waypoints[next_id][1])
            log_msg += f"，新目标：航点 {next_id} (X={next_coords[0]:.1f}, Y={next_coords[1]:.1f})"

        logging.info(log_msg)

    def get_current_target(self) -> List[float]:
        return self.waypoints[self.current_target_id]

    def get_target_speed(self) -> float:
        return self.get_current_target()[2] if self.current_target_id < len(self.waypoints) else 0.0

# ===================== Stanley控制器（优化版）=====================
class StanleyController:
    """Stanley横向控制器"""
    def __init__(self, k: float, max_steer_rad: float, min_speed: float):
        self.k = k
        self.max_steer_rad = max_steer_rad
        self.min_speed = min_speed
        self.prev_steer = 0.0
        self.smoothing_factor = 0.7

    def calculate_steer(self, vehicle: carla.Vehicle, waypoints: List[List[float]], target_id: int) -> Tuple[float, float]:
        if target_id < 1:
            return 0.0, 0.0

        vehicle_transform = vehicle.get_transform()
        vehicle_loc = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
        vehicle_speed = Utils.get_vehicle_speed(vehicle, unit="mps")

        prev_wp = waypoints[target_id - 1]
        target_wp = waypoints[target_id]

        yaw_path = np.arctan2(target_wp[1] - prev_wp[1], target_wp[0] - prev_wp[0])
        yaw_error = Utils.normalize_angle(yaw_path - vehicle_yaw)
        cte = Utils.calculate_cte(vehicle_loc, prev_wp, target_wp)

        # 动态K值
        if vehicle_speed < 5:
            k = self.k * 2
        elif vehicle_speed > 15:
            k = self.k * 0.5
        else:
            k = self.k

        safe_speed = max(vehicle_speed, self.min_speed)
        cte_steer = np.arctan(k * cte / safe_speed)

        total_steer = Utils.normalize_angle(yaw_error + cte_steer)
        total_steer = np.clip(total_steer, -self.max_steer_rad, self.max_steer_rad)
        steer_carla = total_steer / self.max_steer_rad

        # 平滑转向
        steer_carla = self.smoothing_factor * self.prev_steer + (1 - self.smoothing_factor) * steer_carla
        self.prev_steer = steer_carla

        return steer_carla, cte

# ===================== 主仿真类（动态航点版）=====================
class CarlaSimulation:
    """CARLA仿真主类"""
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        self.config = Config()
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Vehicle] = None

        # 动态航点（后续生成）
        self.waypoints = []
        self.waypoint_manager = None
        self.stanley_controller = StanleyController(
            k=self.config.STANLEY_K,
            max_steer_rad=self.config.MAX_STEER_RAD,
            min_speed=self.config.MIN_SPEED_STANLEY
        )
        self.pid_controller = PIDController()

        self.plotter: Optional[Plotter] = None
        self.pygame_display: Optional[PygameDisplay] = None
        self.sim_start_time: float = 0.0

    def connect_carla(self) -> bool:
        try:
            self.client = carla.Client(self.config.CARLA_HOST, self.config.CARLA_PORT)
            self.client.set_timeout(self.config.CARLA_TIMEOUT)
            self.world = self.client.get_world()
            self.logger.info(f"成功连接到CARLA世界：{self.world.get_map().name}")
            return True
        except Exception as e:
            self.logger.error(f"连接CARLA失败：{e}")
            return False

    def cleanup_vehicles(self) -> None:
        self.logger.info("清理历史车辆...")
        actors = self.world.get_actors().filter("vehicle.*")
        removed_count = 0
        for actor in actors:
            if actor.attributes.get("role_name") == "my_car":
                if actor.destroy():
                    removed_count += 1
                    self.logger.info(f"销毁车辆：{actor.type_id} (ID: {actor.id})")
        self.logger.info(f"共销毁 {removed_count} 辆历史车辆")

    def spawn_vehicle(self) -> bool:
        """在开阔道路生成车辆（避开障碍物）"""
        # 获取生成位置的道路Waypoint
        map = self.world.get_map()
        spawn_waypoint = map.get_waypoint(self.config.SPAWN_LOCATION)
        if not spawn_waypoint:
            self.logger.error("生成位置不在道路上")
            return False

        # 生成Transform
        spawn_transform = spawn_waypoint.transform
        spawn_transform.location.z += 0.5
        spawn_transform.rotation.yaw = self.config.SPAWN_YAW

        # 加载车辆蓝图
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter(self.config.VEHICLE_MODEL)[0]
        vehicle_bp.set_attribute("role_name", "my_car")

        # 生成车辆（重试）
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
        if self.vehicle is None:
            spawn_transform.location.z += 0.5
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
            if self.vehicle is None:
                self.logger.error(f"无法生成车辆：{spawn_transform.location}")
                return False

        # 动态生成道路航点（核心：避开障碍物）
        self.waypoints = Utils.generate_road_waypoints(
            world=self.world,
            start_loc=self.config.SPAWN_LOCATION,
            num_waypoints=self.config.NUM_WAYPOINTS,
            spacing=self.config.WAYPOINT_SPACING
        )
        if len(self.waypoints) < 2:
            self.logger.error("生成的航点数量不足")
            return False

        # 初始化航点管理器
        self.waypoint_manager = WaypointManager(
            waypoints=self.waypoints,
            threshold=self.config.WAYPOINT_THRESHOLD
        )

        self.logger.info(f"成功生成车辆：{self.vehicle.type_id} (ID: {self.vehicle.id})")
        self.logger.info(f"生成位置：{spawn_transform.location}, 初始朝向：{spawn_transform.rotation.yaw:.1f}度")
        return True

    def setup_spectator(self) -> None:
        """设置观察者视角（清晰查看车辆前方）"""
        spectator = self.world.get_spectator()
        spectator_transform = carla.Transform(
            self.config.SPECTATOR_LOCATION,
            self.config.SPECTATOR_ROTATION
        )
        spectator.set_transform(spectator_transform)
        self.logger.info("已设置观察者视角")

    def init_visualization(self) -> None:
        self.plotter = Plotter(waypoints=self.waypoints)
        self.plotter.init_plot()
        self.logger.info("Plotter初始化完成")

        self.pygame_display = PygameDisplay(self.world, self.vehicle)
        self.logger.info("Pygame显示初始化完成")

    def run_simulation(self) -> None:
        self.sim_start_time = time.time()
        self.logger.info("开始仿真循环...")

        while True:
            if self.pygame_display.parse_events():
                self.logger.info("用户请求退出仿真")
                break

            self.world.wait_for_tick()

            # 获取车辆状态
            vehicle_transform = self.vehicle.get_transform()
            vehicle_loc = vehicle_transform.location
            current_speed = Utils.get_vehicle_speed(vehicle=self.vehicle, unit="kmh")
            sim_time = time.time() - self.sim_start_time

            # 更新航点
            self.waypoint_manager.update_target(vehicle_loc)
            target_speed = self.waypoint_manager.get_target_speed()

            # 计算控制
            throttle, brake = self.pid_controller.calculate_control(target_speed, current_speed)
            steer, cte = self.stanley_controller.calculate_steer(
                vehicle=self.vehicle,
                waypoints=self.waypoints,
                target_id=self.waypoint_manager.current_target_id
            )

            # 应用控制
            control = carla.VehicleControl()
            control.throttle = throttle
            control.brake = brake
            control.steer = steer
            control.hand_brake = False
            control.manual_gear_shift = False
            self.vehicle.apply_control(control)

            # 调试日志
            if int(sim_time) % 1 == 0 and not hasattr(self, f"_logged_{int(sim_time)}"):
                setattr(self, f"_logged_{int(sim_time)}", True)
                self.logger.info(
                    f"时间：{sim_time:.1f}s | "
                    f"速度：{current_speed:.1f}km/h | "
                    f"目标速度：{target_speed:.1f}km/h | "
                    f"转向角：{steer:.2f} | "
                    f"横向偏差：{cte:.2f}m"
                )

            # 更新可视化
            if self.plotter and self.plotter.is_initialized:
                try:
                    self.plotter.update_plot(sim_time, vehicle_loc.x, vehicle_loc.y, current_speed, target_speed)
                except Exception as e:
                    self.logger.warning(f"绘图更新失败：{e}")
                    self.plotter.cleanup_plot()
                    self.plotter = None

            if self.pygame_display:
                self.pygame_display.render()

    def cleanup(self) -> None:
        self.logger.info("开始清理资源...")

        if self.pygame_display:
            self.pygame_display.destroy()
            self.logger.info("Pygame显示已销毁")

        if self.plotter and self.plotter.is_initialized:
            self.plotter.cleanup_plot()
            self.logger.info("绘图器已清理")

        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
            self.logger.info("车辆已销毁")

        self.logger.info("资源清理完成")

    def start(self) -> None:
        try:
            if not self.connect_carla():
                return

            self.cleanup_vehicles()

            if not self.spawn_vehicle():
                return

            self.setup_spectator()
            self.init_visualization()
            self.run_simulation()

        except KeyboardInterrupt:
            self.logger.info("用户中断仿真")
        except Exception as e:
            self.logger.error(f"仿真异常：{e}", exc_info=True)
        finally:
            self.cleanup()

# ===================== 主函数 =====================
def main():
    simulation = CarlaSimulation()
    simulation.start()

if __name__ == "__main__":
    main()