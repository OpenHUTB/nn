"""
MuJoCo 四旋翼无人机仿真 - 探索轨迹版
✅ 无人机沿8字形轨迹飞行
✅ 自动避开障碍物
✅ 更自然的飞行路径
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import os


class QuadrotorSimulation:
    def __init__(self, xml_path="quadrotor_detailed_city.xml"):
        """初始化：从XML文件加载模型"""
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"找不到XML文件: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"✓ 模型加载成功: {xml_path}")
        self.data = mujoco.MjData(self.model)
        self.n_actuators = self.model.nu

        # 悬停推力参数
        hover_thrust = 650
        self.data.ctrl[:] = [hover_thrust] * self.n_actuators

        # ========== 基础参数（先定义） ==========
        self.hover_height = 1.0  # 基础高度
        self.trajectory_speed = 0.5  # 轨迹速度
        self.rotor_visual_speed = 10.0

        # ========== 轨迹参数 ==========
        self.trajectory_type = "figure8"  # "circle", "figure8", "random"
        self.trajectory_time = 0.0

        # 8字形参数
        self.figure8_width = 2.0
        self.figure8_height = 1.5
        self.figure8_center_x = 0.0
        self.figure8_center_y = 0.0

        # 随机游走参数
        self.random_points = []
        self.current_target_index = 0
        self.generate_random_path()  # 现在可以安全调用，因为hover_height已定义

        # ========== 避障参数 ==========
        self.safety_distance = 1.0
        self.avoidance_strength = 1.5
        self.influence_radius = 2.0

        # 平滑过渡参数
        self.smooth_factor = 0.15
        self.current_pos = np.array([0, 0, self.hover_height])
        self.target_pos = np.array([0, 0, self.hover_height])

        # 避障状态
        self.in_avoidance = False

        # 障碍物位置和尺寸
        self.obstacle_positions = {
            "big_tree": np.array([2.0, 2.0, 0.8]),
            "street_lamp": np.array([-2.0, 2.0, 0.6]),
            "small_shop": np.array([-2.0, -2.0, 0.8]),
            "office": np.array([3.0, 0.0, 1.2]),
            "tower": np.array([-3.0, 0.0, 1.5]),
            "car_red": np.array([1.5, -1.5, 0.3]),
            "car_blue": np.array([-1.5, 1.5, 0.3])
        }
        self.obstacle_sizes = {
            "big_tree": 0.6,
            "street_lamp": 0.3,
            "small_shop": 0.8,
            "office": 0.8,
            "tower": 0.6,
            "car_red": 0.4,
            "car_blue": 0.4
        }

    def generate_random_path(self, num_points=8):
        """生成随机路径点"""
        self.random_points = []
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        for i, angle in enumerate(angles):
            radius = 2.5 + 0.5 * math.sin(angle * 3)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle) * 1.2  # 椭圆形状
            z = self.hover_height + 0.2 * math.sin(angle * 2)
            self.random_points.append([x, y, z])
        self.current_target_index = 0

    def get_trajectory_target(self):
        """获取当前轨迹目标点"""
        t = self.trajectory_time * self.trajectory_speed

        if self.trajectory_type == "circle":
            # 圆形轨迹
            radius = 2.5
            x = radius * math.cos(t)
            y = radius * math.sin(t)
            z = self.hover_height

        elif self.trajectory_type == "figure8":
            # 8字形轨迹（李萨如曲线）
            x = self.figure8_width * math.sin(t)
            y = self.figure8_height * math.sin(t * 2)
            z = self.hover_height + 0.2 * math.sin(t * 3)

        elif self.trajectory_type == "random":
            # 随机点之间的线性插值
            if len(self.random_points) < 2:
                return np.array([0, 0, self.hover_height])

            # 获取当前目标点
            target = np.array(self.random_points[self.current_target_index])

            # 检查是否到达目标点
            dist_to_target = np.linalg.norm(self.current_pos[:2] - target[:2])
            if dist_to_target < 0.3:
                self.current_target_index = (self.current_target_index + 1) % len(self.random_points)
                target = np.array(self.random_points[self.current_target_index])

            return target

        return np.array([x, y, z])

    def calculate_avoidance_force(self, drone_pos):
        """计算避障力"""
        total_force = np.zeros(3)
        min_dist = float('inf')
        closest_obs = None

        for obs_name, obs_pos in self.obstacle_positions.items():
            # 计算到障碍物的向量和距离
            to_obs = drone_pos - obs_pos
            dist = np.linalg.norm(to_obs)
            obs_size = self.obstacle_sizes.get(obs_name, 0.3)

            # 实际距离（减去障碍物尺寸）
            actual_dist = max(dist - obs_size, 0.1)

            if actual_dist < min_dist:
                min_dist = actual_dist
                closest_obs = obs_name

            # 如果在影响范围内，计算排斥力
            if actual_dist < self.influence_radius:
                if dist > 0:
                    force_dir = to_obs / dist
                else:
                    force_dir = np.array([1, 0, 0])

                # 力的强度：距离越近越强
                strength = self.avoidance_strength * (1.0 - actual_dist / self.influence_radius) ** 2

                # 添加一些随机扰动，避免陷入局部最小
                if actual_dist < self.safety_distance * 0.5:
                    # 非常接近时，添加垂直方向的力
                    force_dir[2] += 0.3 * np.random.randn()

                total_force += force_dir * strength

        return total_force, min_dist, closest_obs

    def calculate_optimized_target(self):
        """计算优化后的目标点"""
        # 获取原始轨迹目标
        raw_target = self.get_trajectory_target()

        # 获取当前位置
        current_pos = self.data.qpos[0:3].copy()

        # 计算避障力
        force, min_dist, closest_obs = self.calculate_avoidance_force(current_pos)

        # 根据避障力调整目标点
        if min_dist < self.safety_distance:
            # 避障强度
            intensity = 1.0 - min(min_dist / self.safety_distance, 1.0)

            # 将避障力加到目标点上
            adjusted_target = raw_target + force * intensity * 0.5

            # 确保高度不会太低
            adjusted_target[2] = max(adjusted_target[2], 0.5)

            self.in_avoidance = True
        else:
            adjusted_target = raw_target
            self.in_avoidance = False

        # 平滑过渡
        self.target_pos = adjusted_target
        self.current_pos += (self.target_pos - self.current_pos) * self.smooth_factor

        return self.current_pos, min_dist, closest_obs

    def simulation_loop(self, viewer, duration):
        """核心：探索轨迹的仿真循环"""
        start_time = time.time()
        last_print_time = time.time()

        # 轨迹记录
        path_history = []

        while (viewer is None or (viewer and viewer.is_running())) and (time.time() - start_time) < duration:
            step_start = time.time()

            # 物理仿真步进
            mujoco.mj_step(self.model, self.data)

            # 更新时间
            self.trajectory_time += self.model.opt.timestep

            # 获取当前位置
            current_pos = self.data.qpos[0:3].copy()

            # 计算优化后的目标位置
            target_pos, min_dist, closest_obs = self.calculate_optimized_target()

            # 设置无人机位置
            self.data.qpos[0] = target_pos[0]
            self.data.qpos[1] = target_pos[1]
            self.data.qpos[2] = target_pos[2]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 保持水平

            # 旋翼旋转
            for i in range(4):
                self.data.qpos[7 + i] += self.rotor_visual_speed * self.model.opt.timestep * (i % 2 * 2 - 1)

            # 记录轨迹
            path_history.append([target_pos[0], target_pos[1], target_pos[2]])
            if len(path_history) > 200:
                path_history.pop(0)

            if viewer:
                viewer.sync()

            # 打印状态
            if time.time() - last_print_time > 1.0:
                status = "🚧 避障中" if self.in_avoidance else "✅ 正常飞行"

                print(f"\n{'='*50}")
                print(f"时间: {self.data.time:.1f}s")
                print(f"位置: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})")
                print(f"状态: {status}")
                print(f"最近障碍物: {min_dist:.2f}m | 安全距离: {self.safety_distance}m")
                if closest_obs and self.in_avoidance:
                    print(f"避障对象: {closest_obs}")
                print(f"轨迹类型: {self.trajectory_type}")
                print(f"{'='*50}")
                last_print_time = time.time()

            # 控制仿真速率
            elapsed = time.time() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_simulation(self, duration=60.0, use_viewer=True):
        """运行仿真"""
        print(f"\n{'🚁'*10} 无人机仿真启动 - 探索轨迹 {'🚁'*10}")
        print(f"▶ 轨迹类型: {self.trajectory_type}")
        print(f"▶ 轨迹速度: {self.trajectory_speed}")
        print(f"▶ 安全距离: {self.safety_distance}m")
        print(f"{'='*50}")

        try:
            if use_viewer:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    # 优化相机视角
                    viewer.cam.azimuth = -45
                    viewer.cam.elevation = 30
                    viewer.cam.distance = 15.0
                    viewer.cam.lookat[:] = [0.0, 0.0, self.hover_height]

                    print("▶ 仿真运行中... 按 ESC 退出")
                    self.simulation_loop(viewer, duration)
            else:
                self.simulation_loop(None, duration)
        except Exception as e:
            print(f"⚠ 仿真错误: {e}")

        print(f"\n{'✅'*10} 仿真结束 {'✅'*10}")


def main():
    print("🚁 MuJoCo 四旋翼无人机 - 探索轨迹版")
    print("=" * 50)

    try:
        # XML文件路径
        xml_path = "quadrotor_detailed_city.xml"
        sim = QuadrotorSimulation(xml_path)

        # ========== 轨迹参数选择 ==========
        # 可选: "circle", "figure8", "random"
        sim.trajectory_type = "figure8"  # 8字形轨迹，避免单调转圈

        sim.trajectory_speed = 0.6  # 轨迹速度
        sim.hover_height = 1.2  # 飞行高度

        # 8字形参数
        sim.figure8_width = 2.5
        sim.figure8_height = 2.0

        # 避障参数
        sim.safety_distance = 1.0
        sim.avoidance_strength = 1.5
        sim.influence_radius = 2.0
        sim.smooth_factor = 0.15

        print(f"✅ 初始化完成 - 轨迹类型: {sim.trajectory_type}")
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