"""
MuJoCo 四旋翼无人机仿真 - 直线往返飞行避障版
✅ 无人机直线往返飞行
✅ 自动避开障碍物
✅ 包含升空和降落过程
✅ 轨迹清晰可见
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

        # 基础推力
        self.base_thrust = 600
        self.data.ctrl[:] = [self.base_thrust] * self.n_actuators

        # ========== 飞行阶段 ==========
        self.flight_phase = "takeoff"  # takeoff, cruise, landing
        self.phase_start_time = 0.0

        # 起飞参数
        self.takeoff_height = 1.5       # 起飞目标高度
        self.takeoff_speed = 0.5        # 起飞速度
        self.start_height = 0.2          # 起始高度

        # 巡航参数
        self.cruise_height = 1.5         # 巡航高度
        self.move_speed = 2.5            # 移动速度

        # 降落参数
        self.landing_height = 0.2         # 降落目标高度
        self.landing_speed = 0.3          # 降落速度

        # 直线往返参数
        self.line_distance = 15.0         # 直线飞行距离
        self.line_axis = "x"              # 沿X轴飞行（可选 "x", "y", "xy"）
        self.line_angle = 0.0
        self.line_direction = 1            # 1:正向, -1:反向

        # 计时器
        self.cruise_duration = 40.0       # 巡航时间（秒）
        self.cruise_start_time = 0.0

        # ========== 避障参数 ==========
        self.safety_distance = 1.5        # 安全距离
        self.avoidance_strength = 3.5     # 避障力强度

        # 障碍物位置和尺寸
        self.obstacle_positions = {
            "tree1": np.array([3.0, 2.0, 0.8]),
            "tree2": np.array([-4.0, -2.0, 0.8]),
            "building1": np.array([5.0, -3.0, 1.2]),
            "building2": np.array([-5.0, 4.0, 1.2]),
            "tower1": np.array([8.0, 1.0, 2.0]),
            "tower2": np.array([-8.0, -1.0, 2.0]),
            "car1": np.array([2.0, -5.0, 0.3]),
            "car2": np.array([-2.0, 5.0, 0.3])
        }
        self.obstacle_sizes = {
            "tree1": 0.6,
            "tree2": 0.6,
            "building1": 1.0,
            "building2": 1.0,
            "tower1": 0.8,
            "tower2": 0.8,
            "car1": 0.5,
            "car2": 0.5
        }

    def get_line_target(self):
        """获取直线往返目标点"""
        # 计算当前直线位置
        progress = (self.cruise_start_time > 0 and self.flight_phase == "cruise")

        if self.line_axis == "x":
            # 沿X轴往返
            x = self.line_direction * self.line_distance
            y = 0.0
        elif self.line_axis == "y":
            # 沿Y轴往返
            x = 0.0
            y = self.line_direction * self.line_distance
        else:
            # 沿对角线往返
            x = self.line_direction * self.line_distance * 0.7
            y = self.line_direction * self.line_distance * 0.7

        return np.array([x, y])

    def check_line_end(self, current_pos, target_pos):
        """检查是否到达直线端点"""
        dist_to_target = np.linalg.norm(current_pos[:2] - target_pos[:2])

        if dist_to_target < 2.0:  # 到达端点附近
            self.line_direction *= -1  # 反向
            return True
        return False

    def get_avoidance_force(self, current_pos):
        """计算避障力"""
        avoidance = np.zeros(2)
        min_dist = 999
        closest = ""

        for obs_name, obs_pos in self.obstacle_positions.items():
            to_obs = current_pos[:2] - obs_pos[:2]
            dist = np.linalg.norm(to_obs)
            obs_size = self.obstacle_sizes.get(obs_name, 0.3)

            actual_dist = max(dist - obs_size, 0.1)

            if actual_dist < min_dist:
                min_dist = actual_dist
                closest = obs_name

            if actual_dist < self.safety_distance:
                if dist > 0:
                    direction = to_obs / dist
                else:
                    direction = np.array([1, 0])

                strength = self.avoidance_strength * (1.0 - actual_dist / self.safety_distance) ** 2
                avoidance += direction * strength * 2.0

        return avoidance, min_dist, closest

    def update_flight_phase(self, current_time):
        """更新飞行阶段"""
        if self.flight_phase == "takeoff":
            elapsed = current_time - self.phase_start_time
            progress = min(elapsed * self.takeoff_speed, 1.0)
            current_height = self.start_height + (self.takeoff_height - self.start_height) * progress

            if progress >= 1.0:
                self.flight_phase = "cruise"
                self.phase_start_time = current_time
                self.cruise_start_time = current_time
                self.line_direction = 1  # 重置方向
                print("\n🚁 起飞完成，开始直线往返飞行")

            return current_height

        elif self.flight_phase == "cruise":
            elapsed = current_time - self.cruise_start_time

            if elapsed >= self.cruise_duration:
                self.flight_phase = "landing"
                self.phase_start_time = current_time
                print("\n🛬 开始降落")

            return self.cruise_height

        else:
            elapsed = current_time - self.phase_start_time
            progress = min(elapsed * self.landing_speed, 1.0)
            current_height = self.cruise_height - (self.cruise_height - self.landing_height) * progress

            return current_height

    def simulation_loop(self, viewer, duration):
        """主仿真循环"""
        start_time = time.time()
        last_print_time = time.time()

        self.phase_start_time = 0.0
        smooth_pos = np.array([0.0, 0.0, self.start_height])

        # 记录轨迹点
        path_points = []

        while (viewer is None or (viewer and viewer.is_running())) and (time.time() - start_time) < duration:
            step_start = time.time()
            current_time = self.data.time

            mujoco.mj_step(self.model, self.data)

            current_pos = self.data.qpos[0:3].copy()
            target_height = self.update_flight_phase(current_time)

            # 获取直线目标点
            line_target = self.get_line_target()

            # 计算避障力
            avoidance, min_dist, closest = self.get_avoidance_force(current_pos)

            # 检查是否到达端点
            if self.flight_phase == "cruise":
                self.check_line_end(current_pos, line_target)

            # 计算XY平面移动
            if self.flight_phase == "cruise":
                if np.linalg.norm(avoidance) > 0.1:
                    # 有障碍物：向避障方向移动
                    move_dir = avoidance / np.linalg.norm(avoidance)
                    new_xy = current_pos[:2] + move_dir * self.move_speed * self.model.opt.timestep * 30
                else:
                    # 无障碍物：沿直线飞行
                    to_target = line_target - current_pos[:2]
                    dist_to_target = np.linalg.norm(to_target)

                    if dist_to_target > 0.5:
                        move_dir = to_target / dist_to_target
                        new_xy = current_pos[:2] + move_dir * self.move_speed * self.model.opt.timestep * 30
                    else:
                        new_xy = line_target
            else:
                # 起飞或降落阶段，回到原点
                new_xy = current_pos[:2] * 0.9

            # 限制移动范围
            new_xy[0] = np.clip(new_xy[0], -18, 18)
            new_xy[1] = np.clip(new_xy[1], -18, 18)

            smooth_xy = smooth_pos[:2] + (new_xy - smooth_pos[:2]) * 0.3

            # 记录轨迹点
            path_points.append([smooth_xy[0], smooth_xy[1], target_height])
            if len(path_points) > 500:
                path_points.pop(0)

            # 设置无人机位置
            self.data.qpos[0] = smooth_xy[0]
            self.data.qpos[1] = smooth_xy[1]
            self.data.qpos[2] = target_height
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 保持水平，不旋转

            # 旋翼旋转
            for i in range(4):
                self.data.qpos[7 + i] += 15.0 * self.model.opt.timestep

            if viewer:
                viewer.sync()

            # 打印状态
            if time.time() - last_print_time > 2.0:
                phase_names = {
                    "takeoff": "🔼 起飞",
                    "cruise": "✈️ 直线往返",
                    "landing": "🔽 降落"
                }

                status = phase_names.get(self.flight_phase, "未知")

                print(f"\n{'='*70}")
                print(f"时间: {current_time:.1f}s")
                print(f"阶段: {status}")
                print(f"当前位置: ({smooth_xy[0]:.1f}, {smooth_xy[1]:.1f}, {target_height:.1f})")
                if self.flight_phase == "cruise":
                    direction_text = "正向 →" if self.line_direction > 0 else "反向 ←"
                    print(f"飞行方向: {direction_text}")
                    print(f"目标端点: ({line_target[0]:.1f}, {line_target[1]:.1f})")
                    print(f"飞行距离: {abs(self.line_direction * self.line_distance):.0f}m")
                    print(f"最近障碍物: {closest} 距离 {min_dist:.2f}m")
                else:
                    if self.flight_phase == "takeoff":
                        progress = (target_height - self.start_height) / (self.takeoff_height - self.start_height) * 100
                    else:
                        progress = (self.cruise_height - target_height) / (self.cruise_height - self.landing_height) * 100
                    print(f"进度: {progress:.0f}%")
                print(f"{'='*70}")

                last_print_time = time.time()

            # 控制仿真速率
            elapsed = time.time() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_simulation(self, duration=60.0, use_viewer=True):
        """运行仿真"""
        print(f"\n{'🚁'*10} 无人机直线往返飞行避障 {'🚁'*10}")
        print(f"▶ 起飞高度: {self.takeoff_height}m")
        print(f"▶ 巡航高度: {self.cruise_height}m")
        print(f"▶ 直线距离: {self.line_distance}m")
        print(f"▶ 飞行轴向: {self.line_axis}")
        print(f"▶ 巡航时间: {self.cruise_duration}秒")
        print(f"{'='*70}")

        try:
            if use_viewer:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    # 设置相机视角
                    viewer.cam.azimuth = -45
                    viewer.cam.elevation = 25
                    viewer.cam.distance = 20.0
                    viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

                    print("\n🔼 无人机开始起飞...")
                    self.simulation_loop(viewer, duration)
            else:
                self.simulation_loop(None, duration)
        except Exception as e:
            print(f"⚠ 仿真错误: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n{'✅'*10} 仿真结束 {'✅'*10}")


def main():
    print("🚁 MuJoCo 四旋翼无人机 - 直线往返飞行避障版")
    print("=" * 70)

    try:
        # XML文件路径
        xml_path = "quadrotor_detailed_city.xml"
        sim = QuadrotorSimulation(xml_path)

        # ========== 飞行参数 ==========
        sim.start_height = 0.2
        sim.takeoff_height = 1.5
        sim.takeoff_speed = 0.5
        sim.cruise_height = 1.5
        sim.landing_height = 0.2
        sim.landing_speed = 0.3

        sim.move_speed = 3.5              # 速度

        # 直线往返参数
        sim.line_distance = 18.0           # 飞行距离18米
        sim.line_axis = "x"                # 沿X轴往返（可选 "x", "y", "xy"）

        sim.cruise_duration = 45.0         # 巡航时间

        # ========== 避障参数 ==========
        sim.safety_distance = 1.5
        sim.avoidance_strength = 3.5

        print("✅ 初始化完成")
        print(f"🎯 飞行路线: 沿{sim.line_axis}轴往返 {sim.line_distance}米")
        sim.run_simulation(duration=sim.cruise_duration + 20.0, use_viewer=True)

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