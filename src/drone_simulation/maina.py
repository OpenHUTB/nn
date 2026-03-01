"""
MuJoCo 四旋翼无人机仿真 - 简洁版避障轨迹优化
✅ 无人机绕世界Z轴公转
✅ 自动避开立方体/圆柱体/球体障碍物
✅ 平滑的避障轨迹
✅ 无需额外软件包
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

        # 悬停推力参数
        hover_thrust = 600
        self.data.ctrl[:] = [hover_thrust] * self.n_actuators

        # ========== 旋转参数 ==========
        self.base_radius = 1.0      # 基础公转半径
        self.rotate_speed = 1.0     # 公转角速度（rad/s）
        self.hover_height = 0.8     # 固定高度
        self.rotate_angle = 0.0     # 公转角度累计
        self.rotor_visual_speed = 8.0  # 旋翼旋转速度

        # ========== 简洁的避障参数 ==========
        self.safety_distance = 0.5      # 安全距离
        self.avoidance_offset = 0.8     # 避障偏移量

        # 平滑过渡参数
        self.smooth_factor = 0.2        # 平滑因子（0-1，越小越平滑）
        self.current_radius = self.base_radius
        self.target_radius = self.base_radius

        # 避障状态
        self.in_avoidance = False
        self.avoidance_time = 0
        self.recovery_time = 1.5        # 恢复时间（秒）

        # 障碍物位置和尺寸
        self.obstacle_positions = {
            "cube": np.array([2.0, 0.0, 0.75]),
            "cylinder": np.array([-1.0, 1.0, 0.5]),
            "sphere": np.array([0.0, -2.0, 1.0])
        }
        self.obstacle_sizes = {
            "cube": 0.5,        # 近似半径
            "cylinder": 0.4,
            "sphere": 0.4
        }

    def get_distance_to_obstacles(self, drone_pos):
        """计算到最近障碍物的距离"""
        min_distance = float('inf')
        self.closest_obstacle = None

        for obs_name, obs_pos in self.obstacle_positions.items():
            # 计算3D距离
            dist = np.linalg.norm(drone_pos - obs_pos)
            # 减去障碍物半径
            dist -= self.obstacle_sizes[obs_name]

            if dist < min_distance:
                min_distance = dist
                self.closest_obstacle = obs_name

        return min_distance

    def calculate_smooth_radius(self, drone_pos):
        """计算平滑的避障半径"""
        # 获取到最近障碍物的距离
        min_dist = self.get_distance_to_obstacles(drone_pos)

        # 判断是否需要避障
        need_avoidance = min_dist < self.safety_distance

        if need_avoidance:
            # 计算避障强度（距离越近，避障越强）
            intensity = 1.0 - min(min_dist / self.safety_distance, 1.0)
            # 目标半径 = 基础半径 + 偏移量 * 强度
            self.target_radius = self.base_radius + self.avoidance_offset * intensity
            self.in_avoidance = True
            self.avoidance_time = self.recovery_time
        else:
            # 如果正在避障，逐渐恢复
            if self.in_avoidance:
                self.avoidance_time -= self.model.opt.timestep
                if self.avoidance_time <= 0:
                    self.in_avoidance = False
                    self.target_radius = self.base_radius
            else:
                self.target_radius = self.base_radius

        # 平滑过渡到目标半径
        self.current_radius += (self.target_radius - self.current_radius) * self.smooth_factor

        return self.current_radius

    def simulation_loop(self, viewer, duration):
        """核心：平滑避障的仿真循环"""
        start_time = time.time()
        last_print_time = time.time()

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

            # ========== 3. 计算平滑的避障半径 ==========
            smooth_radius = self.calculate_smooth_radius(current_pos)

            # ========== 4. 计算目标位置 ==========
            target_x = smooth_radius * math.cos(self.rotate_angle)
            target_y = smooth_radius * math.sin(self.rotate_angle)
            target_z = self.hover_height  # 保持高度不变，更稳定

            # ========== 5. 设置无人机位置 ==========
            self.data.qpos[0] = target_x
            self.data.qpos[1] = target_y
            self.data.qpos[2] = target_z
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 姿态不变

            # ========== 6. 旋翼旋转 ==========
            rotor_speed = self.rotor_visual_speed
            for i in range(4):
                self.data.qpos[7 + i] += rotor_speed * self.model.opt.timestep * (i % 2 * 2 - 1)

            if viewer:
                viewer.sync()

            # ========== 7. 打印状态 ==========
            if time.time() - last_print_time > 1.0:
                min_dist = self.get_distance_to_obstacles(current_pos)
                status = "避障中" if self.in_avoidance else "正常飞行"

                print(f"\n时间: {self.data.time:.1f}s")
                print(f"位置: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
                print(f"半径: {smooth_radius:.2f}m | {status}")
                print(f"最近障碍物: {min_dist:.2f}m | 安全距离: {self.safety_distance}m")
                if self.in_avoidance:
                    print(f"恢复倒计时: {self.avoidance_time:.1f}s")
                last_print_time = time.time()

            # 控制仿真速率
            elapsed = time.time() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_simulation(self, duration=60.0, use_viewer=True):
        """运行仿真"""
        print(f"\n▶ 开始仿真（平滑避障），时长: {duration}秒")
        print(f"▶ 基础半径: {self.base_radius}m | 旋转速度: {self.rotate_speed}rad/s")
        print(f"▶ 安全距离: {self.safety_distance}m | 避障偏移: {self.avoidance_offset}m")
        print(f"▶ 平滑因子: {self.smooth_factor}")

        try:
            if use_viewer:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    # 设置相机视角
                    viewer.cam.azimuth = -45
                    viewer.cam.elevation = 20
                    viewer.cam.distance = 10.0
                    viewer.cam.lookat[:] = [0.0, 0.0, self.hover_height]
                    self.simulation_loop(viewer, duration)
            else:
                self.simulation_loop(None, duration)
        except Exception as e:
            print(f"⚠ 仿真错误: {e}")

        print("\n✅ 仿真结束")


def main():
    print("🚁 MuJoCo 四旋翼无人机 - 简洁版平滑避障")
    print("=" * 50)

    try:
        # XML文件路径
        xml_path = "quadrotor_model.xml"
        sim = QuadrotorSimulation(xml_path)

        # ========== 可调参数 ==========
        sim.base_radius = 1.0          # 公转半径
        sim.rotate_speed = 1.0         # 旋转速度
        sim.hover_height = 0.8         # 飞行高度
        sim.safety_distance = 0.5      # 安全距离
        sim.avoidance_offset = 0.8     # 避障偏移量
        sim.smooth_factor = 0.2        # 平滑因子（越小越平滑）
        sim.recovery_time = 1.5        # 恢复时间（秒）

        print("✅ 初始化完成")
        sim.run_simulation(duration=60.0, use_viewer=True)

    except FileNotFoundError as e:
        print(f"\n❌ 文件错误: {e}")
        print("请确保 quadrotor_model.xml 文件在同一目录下")
    except KeyboardInterrupt:
        print("\n\n⏹ 仿真被用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()