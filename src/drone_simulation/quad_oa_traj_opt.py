"""
MuJoCo 四旋翼无人机仿真 - 优化版避障轨迹
✅ 无人机绕世界Z轴公转
✅ 自动避开障碍物（树木、路灯、建筑等）
✅ 优化的避障算法：动态半径 + 切线方向 + 高度微调
✅ 更平滑自然的飞行轨迹
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import os


class QuadrotorSimulation:
    def __init__(self, xml_path="quadrotor_model.xml"):
        """初始化：从XML文件加载模型"""
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"找不到XML文件: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"✓ 模型加载成功: {xml_path}")
        self.data = mujoco.MjData(self.model)
        self.n_actuators = self.model.nu

        # 悬停推力参数（调大一点让无人机更灵敏）
        hover_thrust = 650  # 从600调大到650
        self.data.ctrl[:] = [hover_thrust] * self.n_actuators

        # ========== 旋转参数 ==========
        self.base_radius = 1.0  # 基础公转半径
        self.rotate_speed = 1.2  # 从0.8调快到1.2
        self.hover_height = 0.8  # 固定高度
        self.rotate_angle = 0.0  # 公转角度累计
        self.rotor_visual_speed = 10.0  # 从8.0调快到10.0

        # ========== 优化后的避障参数 ==========
        self.safety_distance = 0.8  # 安全距离
        self.avoidance_offset = 0.6  # 避障偏移量
        self.avoidance_strength = 1.2  # 避障强度

        # 平滑过渡参数
        self.smooth_factor = 0.2  # 从0.15调大到0.2（响应更快）
        self.current_radius = self.base_radius
        self.target_radius = self.base_radius

        # 高度微调参数
        self.height_variation = 0.1  # 避障时的高度变化
        self.current_height = self.hover_height
        self.target_height = self.hover_height

        # 避障状态
        self.in_avoidance = False
        self.avoidance_timer = 0
        self.recovery_time = 2.0  # 恢复时间（秒）

        # 避障方向记忆
        self.avoid_direction = 1.0  # 1=向外，-1=向内

        # 障碍物位置和尺寸
        self.obstacle_positions = {
            "big_tree": np.array([2.0, 0.0, 0.8]),
            "street_lamp": np.array([-1.0, 1.0, 0.6]),
            "small_shop": np.array([0.0, -2.0, 0.8]),
            "park_bench": np.array([2.5, -0.5, 0.3]),
            "traffic_light": np.array([-1.8, 0.8, 0.5]),
            "parked_car": np.array([-0.5, -2.2, 0.3]),
            "bush": np.array([1.2, 1.5, 0.3]),
            "car2": np.array([1.8, 1.8, 0.3]),
            "mailbox": np.array([-1.5, -1.0, 0.3]),
            "fire_hydrant": np.array([2.2, -1.2, 0.3])
        }
        self.obstacle_sizes = {
            "big_tree": 0.6,
            "street_lamp": 0.3,
            "small_shop": 0.8,
            "park_bench": 0.4,
            "traffic_light": 0.3,
            "parked_car": 0.5,
            "bush": 0.3,
            "car2": 0.4,
            "mailbox": 0.2,
            "fire_hydrant": 0.2
        }

        # 障碍物影响范围
        self.influence_radius = 1.5  # 障碍物影响半径

    def get_obstacle_forces(self, drone_pos):
        """计算多个障碍物的合力（人工势场法）"""
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
                # 力的方向：远离障碍物
                if dist > 0:
                    force_dir = to_obs / dist
                else:
                    force_dir = np.array([1, 0, 0])

                # 力的强度：距离越近越强（二次方反比）
                strength = self.avoidance_strength * (1.0 / (actual_dist * actual_dist + 0.1))

                # 计算切线方向，让无人机绕着障碍物飞
                if abs(force_dir[0]) > 0.1 or abs(force_dir[1]) > 0.1:
                    # 生成垂直方向的力（切线）
                    tangent = np.array([-force_dir[1], force_dir[0], 0.2])
                    tangent = tangent / np.linalg.norm(tangent)

                    # 混合径向和切向力
                    force = 0.7 * force_dir + 0.3 * tangent
                else:
                    force = force_dir

                total_force += force * strength

        return total_force, min_dist, closest_obs

    def calculate_optimized_trajectory(self, drone_pos):
        """计算优化后的轨迹参数"""
        # 获取障碍物的合力
        force, min_dist, closest_obs = self.get_obstacle_forces(drone_pos)

        # 判断是否需要避障
        need_avoidance = min_dist < self.safety_distance

        if need_avoidance:
            # 计算避障强度
            intensity = 1.0 - min(min_dist / self.safety_distance, 1.0)

            # 根据合力方向调整目标半径
            force_xy = force[:2]
            if np.linalg.norm(force_xy) > 0:
                # 计算在当前角度方向上的投影
                current_dir = np.array([math.cos(self.rotate_angle), math.sin(self.rotate_angle)])
                dot_product = np.dot(force_xy / np.linalg.norm(force_xy), current_dir)

                # 根据投影调整半径（向外或向内）
                radius_adjust = self.avoidance_offset * intensity * (1 + dot_product)
                self.target_radius = self.base_radius + radius_adjust

                # 记忆避障方向
                self.avoid_direction = 1.0 if dot_product > 0 else -1.0
            else:
                self.target_radius = self.base_radius + self.avoidance_offset * intensity

            # 高度微调（增加避障维度）
            height_adjust = self.height_variation * intensity * math.sin(self.rotate_angle * 2)
            self.target_height = self.hover_height + height_adjust

            self.in_avoidance = True
            self.avoidance_timer = self.recovery_time
        else:
            # 逐渐恢复
            if self.in_avoidance:
                self.avoidance_timer -= self.model.opt.timestep
                if self.avoidance_timer <= 0:
                    self.in_avoidance = False
                    self.target_radius = self.base_radius
                    self.target_height = self.hover_height

            # 缓存在避障状态时平滑恢复
            if self.in_avoidance:
                recovery_factor = self.avoidance_timer / self.recovery_time
                self.target_radius = self.base_radius + (self.target_radius - self.base_radius) * recovery_factor
                self.target_height = self.hover_height + (self.target_height - self.hover_height) * recovery_factor
            else:
                self.target_radius = self.base_radius
                self.target_height = self.hover_height

        # 平滑过渡
        self.current_radius += (self.target_radius - self.current_radius) * self.smooth_factor
        self.current_height += (self.target_height - self.current_height) * self.smooth_factor * 0.5

        return self.current_radius, self.current_height, need_avoidance, closest_obs, min_dist

    def simulation_loop(self, viewer, duration):
        """核心：优化后的仿真循环"""
        start_time = time.time()
        last_print_time = time.time()

        # 轨迹记录（用于调试）
        trajectory = []

        while (viewer is None or (viewer and viewer.is_running())) and (time.time() - start_time) < duration:
            step_start = time.time()

            # 物理仿真步进
            mujoco.mj_step(self.model, self.data)

            # ========== 1. 更新公转角度 ==========
            self.rotate_angle += self.rotate_speed * self.model.opt.timestep
            if self.rotate_angle > 2 * math.pi:
                self.rotate_angle -= 2 * math.pi

            # ========== 2. 获取当前位置 ==========
            current_pos = self.data.qpos[0:3].copy()

            # ========== 3. 计算优化后的轨迹 ==========
            smooth_radius, smooth_height, need_avoidance, closest_obs, min_dist = self.calculate_optimized_trajectory(current_pos)

            # ========== 4. 计算目标位置 ==========
            target_x = smooth_radius * math.cos(self.rotate_angle)
            target_y = smooth_radius * math.sin(self.rotate_angle)
            target_z = smooth_height

            # ========== 5. 记录轨迹 ==========
            trajectory.append([target_x, target_y, target_z])
            if len(trajectory) > 100:
                trajectory.pop(0)

            # ========== 6. 设置无人机位置 ==========
            self.data.qpos[0] = target_x
            self.data.qpos[1] = target_y
            self.data.qpos[2] = target_z
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 保持水平姿态

            # ========== 7. 旋翼旋转 ==========
            rotor_speed = self.rotor_visual_speed
            for i in range(4):
                self.data.qpos[7 + i] += rotor_speed * self.model.opt.timestep * (i % 2 * 2 - 1)

            if viewer:
                viewer.sync()

            # ========== 8. 打印状态 ==========
            if time.time() - last_print_time > 1.0:
                status = "🚧 避障中" if self.in_avoidance else "✅ 正常飞行"
                force, _, _ = self.get_obstacle_forces(current_pos)

                print(f"\n{'='*50}")
                print(f"时间: {self.data.time:.1f}s | 角度: {self.rotate_angle:.2f}rad")
                print(f"位置: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
                print(f"半径: {smooth_radius:.2f}m | 高度: {smooth_height:.2f}m")
                print(f"状态: {status}")
                print(f"最近障碍物: {min_dist:.2f}m | 安全距离: {self.safety_distance}m")
                if closest_obs and self.in_avoidance:
                    print(f"避障对象: {closest_obs}")
                print(f"避障力: ({force[0]:.2f}, {force[1]:.2f}, {force[2]:.2f})")
                print(f"{'='*50}")
                last_print_time = time.time()

            # 控制仿真速率
            elapsed = time.time() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_simulation(self, duration=60.0, use_viewer=True):
        """运行仿真"""
        print(f"\n{'🚁'*10} 无人机仿真启动 {'🚁'*10}")
        print(f"▶ 基础半径: {self.base_radius}m | 旋转速度: {self.rotate_speed}rad/s")
        print(f"▶ 安全距离: {self.safety_distance}m | 避障偏移: {self.avoidance_offset}m")
        print(f"▶ 平滑因子: {self.smooth_factor} | 高度变化: {self.height_variation}m")
        print(f"▶ 障碍物数量: {len(self.obstacle_positions)}")
        print(f"{'='*50}")

        try:
            if use_viewer:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    # 优化相机视角
                    viewer.cam.azimuth = -45
                    viewer.cam.elevation = 25
                    viewer.cam.distance = 12.0
                    viewer.cam.lookat[:] = [0.0, 0.0, self.hover_height]

                    print("▶ 仿真运行中... 按 ESC 退出")
                    self.simulation_loop(viewer, duration)
            else:
                self.simulation_loop(None, duration)
        except Exception as e:
            print(f"⚠ 仿真错误: {e}")

        print(f"\n{'✅'*10} 仿真结束 {'✅'*10}")


def main():
    print("🚁 MuJoCo 四旋翼无人机 - 优化版避障轨迹")
    print("=" * 50)

    try:
        # XML文件路径
        xml_path = "quadrotor_model.xml"
        sim = QuadrotorSimulation(xml_path)

        # ========== 可调参数 ==========
        sim.base_radius = 1.0  # 公转半径
        sim.rotate_speed = 1.2  # 调快到1.2
        sim.hover_height = 0.8  # 飞行高度

        # 避障参数
        sim.safety_distance = 0.8  # 安全距离
        sim.avoidance_offset = 0.6  # 避障偏移量
        sim.avoidance_strength = 1.2  # 避障强度
        sim.smooth_factor = 0.2  # 平滑因子（调大一点让响应更快）
        sim.height_variation = 0.1  # 高度变化幅度
        sim.recovery_time = 1.8  # 恢复时间（稍微缩短）

        # 障碍物影响范围
        sim.influence_radius = 1.5  # 障碍物影响半径

        print("✅ 初始化完成")
        sim.run_simulation(duration=60.0, use_viewer=True)

    except FileNotFoundError as e:
        print(f"\n❌ 文件错误: {e}")
        print("请确保 quadrotor_model.xml 文件在同一目录下")
    except KeyboardInterrupt:
        print("\n\n⏹ 仿真被用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()