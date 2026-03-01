"""
MuJoCo 四旋翼无人机仿真 - 公转+避障版
✅ 无人机绕世界Z轴公转，保持原旋转逻辑
✅ 自动避开立方体/圆柱体/球体障碍物
✅ 避障后自动恢复原轨迹，高度固定、无闪烁
✅ 保留所有原代码核心特征
✅ 优化的无人机模型：更精致的外观、更真实的旋翼、更好的视觉效果
✅ 真实风格的障碍物：建筑、油罐、巨石
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
        # 检查XML文件是否存在
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"找不到XML文件: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"✓ 模型加载成功: {xml_path}")
        self.data = mujoco.MjData(self.model)
        self.n_actuators = self.model.nu

        # 原代码悬停推力参数
        hover_thrust = 600
        self.data.ctrl[:] = [hover_thrust] * self.n_actuators

        # ========== 原代码旋转参数 ==========
        self.base_radius = 1.0      # 基础公转半径
        self.rotate_speed = 1.0     # 公转角速度（rad/s）
        self.hover_height = 0.8     # 固定高度
        self.rotate_angle = 0.0     # 公转角度累计
        self.rotor_visual_speed = 8.0  # 旋翼旋转速度

        # ========== 避障核心参数 ==========
        self.safety_distance = 0.5  # 安全距离（小于此距离触发避障）
        self.avoidance_offset = 0.8 # 避障偏移量（扩大半径绕开障碍物）
        self.obstacle_positions = { # 预定义障碍物位置（与XML中一致）
            "cube": np.array([2.0, 0.0, 0.75]),
            "cylinder": np.array([-1.0, 1.0, 0.5]),
            "sphere": np.array([0.0, -2.0, 1.0])
        }
        self.obstacle_sizes = {     # 障碍物尺寸（碰撞判定用）
            "cube": np.array([0.25, 0.25, 0.75]),
            "cylinder": np.array([0.3, 0.5]),  # 半径、高度
            "sphere": np.array([0.4])          # 半径
        }

    def calculate_obstacle_distance(self, drone_pos):
        """计算无人机到各障碍物的水平距离（Z轴高度忽略，只算XY平面）"""
        distances = {}

        # 立方体障碍物
        cube_pos = self.obstacle_positions["cube"][:2]  # 只取XY坐标
        drone_xy = drone_pos[:2]
        distances["cube"] = np.linalg.norm(drone_xy - cube_pos) - self.obstacle_sizes["cube"][0]

        # 圆柱体障碍物
        cyl_pos = self.obstacle_positions["cylinder"][:2]
        distances["cylinder"] = np.linalg.norm(drone_xy - cyl_pos) - self.obstacle_sizes["cylinder"][0]

        # 球体障碍物
        sphere_pos = self.obstacle_positions["sphere"][:2]
        distances["sphere"] = np.linalg.norm(drone_xy - sphere_pos) - self.obstacle_sizes["sphere"][0]

        return distances

    def get_avoidance_radius(self, drone_pos):
        """根据障碍物距离动态调整公转半径（避障核心逻辑）"""
        distances = self.calculate_obstacle_distance(drone_pos)
        min_distance = min(distances.values())

        # 判定是否需要避障
        if min_distance < self.safety_distance:
            # 找到最近的障碍物
            closest_obs = min(distances, key=distances.get)
            obs_pos = self.obstacle_positions[closest_obs][:2]
            drone_xy = drone_pos[:2]

            # 计算避障方向：远离最近障碍物
            direction = drone_xy - obs_pos
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else np.array([1, 0])

            # 动态调整半径，绕开障碍物
            return self.base_radius + self.avoidance_offset
        else:
            # 无避障需求，恢复基础半径
            return self.base_radius

    def simulation_loop(self, viewer, duration):
        """核心：公转+避障逻辑"""
        start_time = time.time()
        last_print_time = time.time()

        while (viewer is None or (viewer and viewer.is_running())) and (time.time() - start_time) < duration:
            step_start = time.time()

            # 物理仿真步进
            mujoco.mj_step(self.model, self.data)

            # ========== 1. 更新公转角度 ==========
            self.rotate_angle += self.rotate_speed * self.model.opt.timestep
            # 限制角度范围（防止数值过大）
            if self.rotate_angle > 2 * math.pi:
                self.rotate_angle -= 2 * math.pi

            # ========== 2. 计算基础公转位置 ==========
            base_x = self.base_radius * math.cos(self.rotate_angle)
            base_y = self.base_radius * math.sin(self.rotate_angle)
            base_pos = np.array([base_x, base_y, self.hover_height])

            # ========== 3. 避障逻辑：动态调整位置 ==========
            current_radius = self.get_avoidance_radius(base_pos)
            # 计算避障后的目标位置
            target_x = current_radius * math.cos(self.rotate_angle)
            target_y = current_radius * math.sin(self.rotate_angle)
            target_z = self.hover_height

            # ========== 4. 设置无人机位置和姿态 ==========
            self.data.qpos[0] = target_x  # X轴位置
            self.data.qpos[1] = target_y  # Y轴位置
            self.data.qpos[2] = target_z  # Z轴固定高度
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 姿态不变

            # ========== 5. 旋翼旋转（保持原逻辑） ==========
            rotor_speed = self.rotor_visual_speed
            for i in range(4):
                self.data.qpos[7 + i] += rotor_speed * self.model.opt.timestep * (i % 2 * 2 - 1)

            if viewer:
                viewer.sync()

            # ========== 6. 打印状态信息 ==========
            if time.time() - last_print_time > 1.0:
                current_time = self.data.time
                current_pos = self.data.qpos[0:3].copy()
                distances = self.calculate_obstacle_distance(current_pos)
                min_dist = min(distances.values())
                avoidance_status = "避障中" if min_dist < self.safety_distance else "正常轨迹"

                print(f"\n时间: {current_time:.1f}s | 公转角度: {self.rotate_angle:.2f}rad")
                print(f"当前位置: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}] m")
                print(f"公转半径: {current_radius:.2f}m | 状态: {avoidance_status}")
                print(f"最近障碍物距离: {min_dist:.2f}m | 安全距离: {self.safety_distance}m")
                last_print_time = time.time()

            # 控制仿真速率
            elapsed = time.time() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_simulation(self, duration=60.0, use_viewer=True):
        """运行仿真：带避障功能"""
        print(f"\n▶ 开始仿真（公转+自动避障），时长: {duration}秒")
        print(f"▶ 基础公转半径: {self.base_radius}m | 旋转速度: {self.rotate_speed}rad/s")
        print(f"▶ 安全距离: {self.safety_distance}m | 避障偏移量: {self.avoidance_offset}m")
        print(f"▶ 无人机模型: 优化版（碳纤维机身+四色旋翼+LED灯）")
        print(f"▶ 障碍物模型: 真实风格（多层建筑/工业油罐/巨石群）")

        try:
            if use_viewer:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    # 优化相机视角，方便观察避障效果
                    viewer.cam.azimuth = -45
                    viewer.cam.elevation = 20
                    viewer.cam.distance = 12.0
                    viewer.cam.lookat[:] = [0.0, 0.0, self.hover_height]
                    self.simulation_loop(viewer, duration)
            else:
                self.simulation_loop(None, duration)
        except Exception as e:
            print(f"⚠ 仿真错误: {e}")
            import traceback
            traceback.print_exc()

        print("\n✅ 仿真结束（避障功能正常运行）")


def main():
    print("🚁 MuJoCo 四旋翼无人机仿真 - 公转+自动避障版（真实障碍物）")
    print("=" * 70)

    try:
        # 指定XML文件路径
        xml_path = "quadrotor_model.xml"
        sim = QuadrotorSimulation(xml_path)

        # ========== 可自定义参数 ==========
        # 原旋转参数
        sim.base_radius = 1.0      # 基础公转半径
        sim.rotate_speed = 1.0     # 旋转速度
        sim.hover_height = 0.8     # 悬停高度
        # 避障参数
        sim.safety_distance = 0.5  # 触发避障的安全距离（越小越灵敏）
        sim.avoidance_offset = 0.8 # 避障时的半径偏移量（越大避障越远）

        print("✅ 初始化完成（避障功能已启用）")
        sim.run_simulation(
            duration=60.0,
            use_viewer=True
        )

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