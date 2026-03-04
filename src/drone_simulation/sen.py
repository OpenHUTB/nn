"""
MuJoCo 四旋翼无人机仿真 - 感知避障系统版
✅ 多传感器融合感知
✅ 动态路径规划
✅ 智能避障决策
✅ 实时环境感知
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import os
from collections import deque


class SensorSystem:
    """无人机感知系统"""

    def __init__(self, drone_pos, obstacle_positions, obstacle_sizes):
        self.drone_pos = drone_pos
        self.obstacle_positions = obstacle_positions
        self.obstacle_sizes = obstacle_sizes

        # 传感器参数
        self.lidar_range = 5.0  # 激光雷达探测距离
        self.lidar_resolution = 36  # 扫描分辨率（每10度一个点）
        self.ultrasonic_range = 3.0  # 超声波传感器距离
        self.camera_fov = 60  # 相机视场角（度）

        # 传感器数据
        self.lidar_points = []  # 激光雷达点云
        self.obstacle_directions = []  # 障碍物方向
        self.danger_zones = []  # 危险区域
        self.safe_directions = []  # 安全方向

        # 历史数据
        self.history = deque(maxlen=10)

    def update(self, drone_pos):
        """更新传感器数据"""
        self.drone_pos = drone_pos
        self.lidar_points = []
        self.obstacle_directions = []
        self.danger_zones = []

        # 激光雷达扫描
        for angle in range(0, 360, 10):  # 每10度扫描一次
            rad = math.radians(angle)
            direction = np.array([math.cos(rad), math.sin(rad), 0])

            # 射线投射
            hit_point, hit_dist, hit_obs = self.ray_cast(drone_pos, direction)
            if hit_point is not None:
                self.lidar_points.append({
                    'angle': angle,
                    'distance': hit_dist,
                    'point': hit_point,
                    'obstacle': hit_obs
                })

                # 判断是否危险
                if hit_dist < self.safety_distance():
                    self.danger_zones.append({
                        'angle': angle,
                        'distance': hit_dist,
                        'obstacle': hit_obs
                    })

        # 计算安全方向
        self.calculate_safe_directions()

        # 保存历史
        self.history.append({
            'pos': drone_pos.copy(),
            'danger_zones': self.danger_zones.copy()
        })

    def ray_cast(self, start, direction, max_dist=5.0):
        """射线投射检测障碍物"""
        min_dist = max_dist
        hit_point = None
        hit_obs = None

        for obs_name, obs_pos in self.obstacle_positions.items():
            # 计算射线到障碍物的距离
            to_obs = obs_pos - start
            obs_size = self.obstacle_sizes.get(obs_name, 0.5)

            # 计算投影距离
            proj_dist = np.dot(to_obs, direction)
            if proj_dist < 0 or proj_dist > max_dist:
                continue

            # 计算垂直距离
            perp_dist = np.linalg.norm(to_obs - proj_dist * direction)

            # 如果垂直距离小于障碍物半径，说明射线击中
            if perp_dist < obs_size:
                if proj_dist < min_dist:
                    min_dist = proj_dist
                    hit_point = start + direction * proj_dist
                    hit_obs = obs_name

        return hit_point, min_dist, hit_obs

    def safety_distance(self):
        """动态安全距离"""
        base_dist = 1.0
        # 根据速度调整安全距离
        if len(self.history) > 1:
            last_pos = self.history[-2]['pos']
            velocity = np.linalg.norm(self.drone_pos - last_pos) / 0.01
            return base_dist + velocity * 0.5
        return base_dist

    def calculate_safe_directions(self):
        """计算安全飞行方向"""
        self.safe_directions = []

        # 将360度分成36个方向
        for angle in range(0, 360, 10):
            rad = math.radians(angle)
            direction = np.array([math.cos(rad), math.sin(rad), 0])

            # 检查这个方向是否安全
            is_safe = True
            min_safe_dist = self.safety_distance()

            for danger in self.danger_zones:
                angle_diff = abs(angle - danger['angle'])
                if angle_diff < 30:  # 30度范围内都受影响
                    is_safe = False
                    break

            if is_safe:
                self.safe_directions.append(direction)

    def get_best_direction(self, target_direction):
        """获取最佳飞行方向"""
        if not self.safe_directions:
            return None

        # 找出最接近目标方向的安全方向
        best_dir = None
        best_score = -1

        for safe_dir in self.safe_directions:
            # 计算与目标方向的相似度
            similarity = np.dot(safe_dir, target_direction)
            # 归一化分数
            score = (similarity + 1) / 2

            if score > best_score:
                best_score = score
                best_dir = safe_dir

        return best_dir


class PathPlanner:
    """路径规划器"""

    def __init__(self):
        self.waypoints = []
        self.current_waypoint = 0
        self.path_history = []

        # 预设航点（覆盖整个环境）
        self.default_waypoints = [
            np.array([0.0, 0.0, 1.5]),
            np.array([3.0, 3.0, 1.5]),
            np.array([3.0, -3.0, 1.5]),
            np.array([-3.0, -3.0, 1.5]),
            np.array([-3.0, 3.0, 1.5]),
            np.array([0.0, 0.0, 1.5]),
            np.array([4.0, 0.0, 1.5]),
            np.array([0.0, 4.0, 1.5]),
            np.array([-4.0, 0.0, 1.5]),
            np.array([0.0, -4.0, 1.5]),
        ]

    def set_waypoints(self, waypoints):
        self.waypoints = waypoints
        self.current_waypoint = 0

    def get_next_waypoint(self, current_pos):
        """获取下一个航点"""
        if not self.waypoints:
            self.waypoints = self.default_waypoints

        target = self.waypoints[self.current_waypoint]

        # 检查是否到达当前航点
        dist = np.linalg.norm(current_pos[:2] - target[:2])
        if dist < 1.0:
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
            target = self.waypoints[self.current_waypoint]

        return target

    def get_exploration_target(self, current_pos, safe_directions):
        """获取探索目标（无预设航点时）"""
        if not safe_directions:
            return None

        # 随机选择一个安全方向
        direction = safe_directions[np.random.randint(len(safe_directions))]
        # 向前飞行3米
        target = current_pos + direction * 3.0
        target[2] = 1.5  # 保持高度

        return target


class QuadrotorSimulation:
    def __init__(self, xml_path="quadrotor_detailed_city.xml"):
        """初始化：从XML文件加载模型"""
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"找不到XML文件: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"✓ 模型加载成功: {xml_path}")
        self.data = mujoco.MjData(self.model)
        self.n_actuators = self.model.nu

        # 基础推力
        self.base_thrust = 600
        self.data.ctrl[:] = [self.base_thrust] * self.n_actuators

        # ========== 飞行参数 ==========
        self.flight_phase = "takeoff"
        self.phase_start_time = 0.0
        self.takeoff_height = 1.5
        self.cruise_height = 1.5
        self.landing_height = 0.2

        # ========== 运动参数 ==========
        self.max_speed = 2.0
        self.acceleration = 1.0
        self.current_velocity = np.zeros(3)
        self.target_pos = np.array([0, 0, self.takeoff_height])

        # ========== 感知系统 ==========
        # 障碍物位置和尺寸
        self.obstacle_positions = {
            "building_office": np.array([5.0, 5.0, 1.0]),
            "building_tower": np.array([8.0, 4.0, 1.5]),
            "building_apartment": np.array([3.0, 8.0, 1.2]),
            "building_shop": np.array([-5.0, 5.0, 1.0]),
            "building_cafe": np.array([-8.0, 4.0, 1.0]),
            "building_house1": np.array([-5.0, -5.0, 0.8]),
            "building_house2": np.array([-8.0, -5.0, 0.8]),
            "building_school": np.array([5.0, -5.0, 1.2]),
            "building_library": np.array([8.0, -5.0, 1.0]),
            "tree_1": np.array([2.0, 2.0, 0.8]),
            "tree_2": np.array([-2.0, 2.0, 0.8]),
            "tree_3": np.array([2.0, -2.0, 0.8]),
            "tree_4": np.array([-2.0, -2.0, 0.8]),
            "light_1": np.array([3.0, 3.0, 0.6]),
            "light_2": np.array([-3.0, 3.0, 0.6]),
            "light_3": np.array([3.0, -3.0, 0.6]),
            "light_4": np.array([-3.0, -3.0, 0.6]),
            "car_1": np.array([2.0, 0.0, 0.3]),
            "car_2": np.array([-2.0, 0.0, 0.3])
        }
        self.obstacle_sizes = {
            "building_office": 1.5,
            "building_tower": 1.0,
            "building_apartment": 1.2,
            "building_shop": 1.2,
            "building_cafe": 1.0,
            "building_house1": 0.8,
            "building_house2": 0.8,
            "building_school": 1.2,
            "building_library": 1.0,
            "tree_1": 0.5,
            "tree_2": 0.5,
            "tree_3": 0.5,
            "tree_4": 0.5,
            "light_1": 0.3,
            "light_2": 0.3,
            "light_3": 0.3,
            "light_4": 0.3,
            "car_1": 0.5,
            "car_2": 0.5
        }

        # 初始化感知系统
        self.sensor_system = SensorSystem(
            self.data.qpos[0:3].copy(),
            self.obstacle_positions,
            self.obstacle_sizes
        )

        # 初始化路径规划器
        self.path_planner = PathPlanner()

        # 状态记录
        self.avoidance_count = 0
        self.last_avoidance_time = 0

    def update_flight_phase(self, current_time):
        """更新飞行阶段"""
        if self.flight_phase == "takeoff":
            elapsed = current_time - self.phase_start_time
            progress = min(elapsed * 0.5, 1.0)
            current_height = 0.2 + (self.takeoff_height - 0.2) * progress

            if progress >= 1.0:
                self.flight_phase = "cruise"
                self.phase_start_time = current_time
                print("\n🚁 起飞完成，开始巡航")

            return current_height

        elif self.flight_phase == "cruise":
            return self.cruise_height

        else:  # landing
            elapsed = current_time - self.phase_start_time
            progress = min(elapsed * 0.3, 1.0)
            current_height = self.cruise_height - (self.cruise_height - 0.2) * progress

            return current_height

    def calculate_safe_movement(self, current_pos, target_pos):
        """计算安全移动方向"""
        # 更新感知系统
        self.sensor_system.update(current_pos)

        # 目标方向
        to_target = target_pos - current_pos
        dist_to_target = np.linalg.norm(to_target)

        if dist_to_target < 0.1:
            return np.zeros(3)

        target_dir = to_target / dist_to_target

        # 获取最佳安全方向
        best_dir = self.sensor_system.get_best_direction(target_dir)

        if best_dir is not None:
            # 有安全方向，向该方向移动
            move_dir = best_dir
        else:
            # 没有安全方向，紧急处理
            print("⚠️ 紧急情况：所有方向都有危险！")
            self.avoidance_count += 1

            # 向上飞行作为最后手段
            move_dir = np.array([0, 0, 1])

        # 计算速度
        target_velocity = move_dir * self.max_speed

        # 平滑加速
        self.current_velocity += (target_velocity - self.current_velocity) * 0.1

        # 计算新位置
        new_pos = current_pos + self.current_velocity * self.model.opt.timestep * 50

        # 确保高度合适
        new_pos[2] = max(0.3, min(2.0, new_pos[2]))

        return new_pos

    def get_next_target(self, current_pos):
        """获取下一个目标点"""
        if self.flight_phase == "takeoff":
            return np.array([0, 0, self.takeoff_height])
        elif self.flight_phase == "landing":
            return np.array([0, 0, 0.2])
        else:
            # 巡航阶段：使用路径规划器
            return self.path_planner.get_next_waypoint(current_pos)

    def simulation_loop(self, viewer, duration):
        """主仿真循环"""
        start_time = time.time()
        last_print_time = time.time()
        self.phase_start_time = 0.0

        # 平滑位置
        smooth_pos = np.array([0, 0, 0.2])

        while (viewer is None or (viewer and viewer.is_running())) and (time.time() - start_time) < duration:
            step_start = time.time()
            current_time = self.data.time

            # 物理仿真步进
            mujoco.mj_step(self.model, self.data)

            # 获取当前位置
            current_pos = self.data.qpos[0:3].copy()

            # 更新飞行高度
            target_height = self.update_flight_phase(current_time)

            # 获取下一个目标点
            next_target = self.get_next_target(current_pos)
            next_target[2] = target_height

            # 计算安全移动
            new_pos = self.calculate_safe_movement(current_pos, next_target)

            # 平滑移动
            smooth_pos = smooth_pos + (new_pos - smooth_pos) * 0.3

            # 设置无人机位置
            self.data.qpos[0] = smooth_pos[0]
            self.data.qpos[1] = smooth_pos[1]
            self.data.qpos[2] = smooth_pos[2]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 保持水平

            # 旋翼旋转视觉效果
            for i in range(4):
                self.data.qpos[7 + i] += 15.0 * self.model.opt.timestep

            if viewer:
                viewer.sync()

            # 打印状态
            if time.time() - last_print_time > 2.0:
                # 获取感知数据
                danger_count = len(self.sensor_system.danger_zones)
                safe_dir_count = len(self.sensor_system.safe_directions)

                # 计算最近障碍物
                min_dist = 999
                closest = "无"
                for obs_name, obs_pos in self.obstacle_positions.items():
                    dist = np.linalg.norm(smooth_pos - obs_pos) - self.obstacle_sizes[obs_name]
                    if dist < min_dist:
                        min_dist = dist
                        closest = obs_name

                phase_names = {
                    "takeoff": "🔼 起飞",
                    "cruise": "✈️ 巡航感知",
                    "landing": "🔽 降落"
                }

                print(f"\n{'=' * 60}")
                print(f"时间: {current_time:.1f}s | 阶段: {phase_names[self.flight_phase]}")
                print(f"位置: ({smooth_pos[0]:.2f}, {smooth_pos[1]:.2f}, {smooth_pos[2]:.2f})")
                print(f"速度: {np.linalg.norm(self.current_velocity):.2f} m/s")
                print(f"目标: ({next_target[0]:.1f}, {next_target[1]:.1f})")
                print(f"\n【感知系统】")
                print(f"  危险区域: {danger_count}处")
                print(f"  安全方向: {safe_dir_count}个")
                print(f"  最近障碍: {closest} 距离 {min_dist:.2f}m")
                print(f"  避障次数: {self.avoidance_count}")
                print(f"{'=' * 60}")

                last_print_time = time.time()

            # 控制仿真速率
            elapsed = time.time() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_simulation(self, duration=60.0, use_viewer=True):
        """运行仿真"""
        print(f"\n{'🚁' * 10} 无人机感知避障系统 {'🚁' * 10}")
        print(f"▶ 激光雷达范围: {self.sensor_system.lidar_range}m")
        print(f"▶ 最大速度: {self.max_speed}m/s")
        print(f"▶ 障碍物数量: {len(self.obstacle_positions)}")
        print(f"{'=' * 60}")

        try:
            if use_viewer:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    # 设置相机视角
                    viewer.cam.azimuth = -45
                    viewer.cam.elevation = 30
                    viewer.cam.distance = 15.0
                    viewer.cam.lookat[:] = [0.0, 0.0, 0.8]

                    print("\n🔼 无人机开始起飞...")
                    print("感知系统已激活，正在扫描环境...")
                    self.simulation_loop(viewer, duration)
            else:
                self.simulation_loop(None, duration)
        except Exception as e:
            print(f"⚠ 仿真错误: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n{'✅' * 10} 仿真结束 {'✅' * 10}")


def main():
    print("🚁 MuJoCo 四旋翼无人机 - 感知避障系统版")
    print("=" * 60)

    try:
        # XML文件路径
        xml_path = "quadrotor_detailed_city.xml"
        sim = QuadrotorSimulation(xml_path)

        # ========== 可调参数 ==========
        sim.max_speed = 2.0  # 最大速度
        sim.takeoff_height = 1.5  # 起飞高度
        sim.cruise_height = 1.5  # 巡航高度

        # 选择飞行模式
        # 巡航阶段会自动在航点间移动

        print("✅ 初始化完成")
        print("▶ 感知系统已就绪")
        sim.run_simulation(duration=60.0, use_viewer=True)

    except FileNotFoundError as e:
        print(f"\n❌ 文件错误: {e}")
        print("请确保 quadrotor_detailed_city.xml 文件在同一目录下")
    except KeyboardInterrupt:
        print("\n\n⏹ 仿真被用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()