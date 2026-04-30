import mujoco
import mujoco.viewer
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
import os
import warnings

# 忽略无关警告（提升体验）
warnings.filterwarnings("ignore", category=UserWarning)


class RoboticArmController:
    """6自由度机械臂控制器（修复兼容版）"""

    def __init__(self, model_path: str):
        """初始化机械臂控制器（保持原版初始化逻辑）"""
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载模型和数据（兼容不同MuJoCo版本）
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"加载模型失败：{e}")

        # 关节名称（保持和原版一致）
        self.arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

        # 获取关节ID（增加容错）
        self.arm_joint_ids = {}
        for name in self.arm_joint_names:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid == -1:
                    raise ValueError(f"关节 {name} 未找到")
                self.arm_joint_ids[name] = jid
            except Exception as e:
                print(f"警告：获取关节 {name} ID失败（可能不影响基础功能）：{e}")

        # 获取执行器ID（核心修复：兼容不同命名规则）
        self.actuator_ids = {}
        for name in self.arm_joint_names:
            actuator_name = f"motor_{name}"
            try:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if aid == -1:
                    # 尝试直接用关节名作为执行器名（兼容常见模型）
                    aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if aid == -1:
                    raise ValueError("执行器ID不存在")
                self.actuator_ids[name] = aid  # 简化映射：关节名 -> 执行器ID
            except Exception as e:
                print(f"警告：获取执行器 {actuator_name} ID失败：{e}")

        # 关节角度限制（保持原版）
        self.joint_limits = {
            "joint1": (-np.pi, np.pi),
            "joint2": (-2.0, 2.0),
            "joint3": (-2.5, 0.5),
            "joint4": (-2.0, 2.0),
            "joint5": (-1.8, 1.8),
            "joint6": (-1.8, 1.8)
        }

        # 绕圈参数（保持原版）
        self.circle_radius = 0.15
        self.circle_center = [0.2, 0.0, 0.2]
        self.circle_speed = 1.0
        self.last_angles = None

        # 缓存仿真步长（修复计时问题）
        self.sim_timestep = self.model.opt.timestep

    def clamp_joint_angle(self, joint_name: str, angle: float) -> float:
        """保持原版逻辑"""
        if joint_name not in self.joint_limits:
            return angle
        min_angle, max_angle = self.joint_limits[joint_name]
        return np.clip(angle, min_angle, max_angle)

    def smooth_set_joint_angle(self, joint_name: str, target_angle: float, speed: float = 0.8):
        """修复执行器ID映射问题"""
        if joint_name not in self.arm_joint_ids:
            print(f"跳过不存在的关节：{joint_name}")
            return True
        if joint_name not in self.actuator_ids:
            print(f"跳过无执行器的关节：{joint_name}")
            return True

        target_angle = self.clamp_joint_angle(joint_name, target_angle)
        actuator_id = self.actuator_ids[joint_name]  # 修复映射逻辑

        # 获取当前角度（兼容不同访问方式）
        try:
            current_angle = self.data.joint(joint_name).qpos[0]
            current_vel = self.data.joint(joint_name).qvel[0]
        except:
            # 备选方案：直接通过ID访问（更稳定）
            current_angle = self.data.qpos[self.arm_joint_ids[joint_name]]
            current_vel = self.data.qvel[self.arm_joint_ids[joint_name]]

        # 角度差计算（保持原版）
        angle_diff = target_angle - current_angle
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        if abs(angle_diff) < 0.001:
            self.data.ctrl[actuator_id] = 0
            return True

        # PD控制器（保持原版参数，避免过度优化导致问题）
        kp = 60.0
        kd = 8.0
        control = kp * angle_diff - kd * current_vel

        max_control = speed * 15
        control = np.clip(control, -max_control, max_control)
        self.data.ctrl[actuator_id] = control

        return False

    def generate_circle_trajectory(self, t: float):
        """保持原版轨迹生成逻辑"""
        theta = self.circle_speed * t
        target_x = self.circle_center[0] + self.circle_radius * np.cos(theta)
        target_y = self.circle_center[1] + self.circle_radius * np.sin(theta)
        target_z = self.circle_center[2]

        joint1_target = np.arctan2(target_y, target_x - self.circle_center[0])
        joint2_target = 0.6 + 0.2 * np.cos(theta)
        joint3_target = -0.8 + 0.2 * np.sin(theta)
        joint4_target = 0.0
        joint5_target = 0.0
        joint6_target = 0.0

        return {
            "joint1": joint1_target, "joint2": joint2_target, "joint3": joint3_target,
            "joint4": joint4_target, "joint5": joint5_target, "joint6": joint6_target
        }

    def simulate_circle_motion(self, duration: float = None):
        """修复仿真循环和viewer问题"""
        start_time = time.time()

        # 修复viewer启动逻辑（兼容不同MuJoCo版本）
        try:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except:
            # 备选启动方式（旧版本兼容）
            viewer = mujoco.viewer.Viewer(self.model, self.data)

        # 设置相机（保持原版）
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 30
        viewer.cam.elevation = -15
        viewer.cam.lookat = self.circle_center

        print(f"🔄 开始机械臂绕圈运动，绕圈中心：{self.circle_center}，半径：{self.circle_radius}m")
        print("🛑 关闭窗口停止仿真...")

        try:
            while True:
                # 检查窗口是否关闭（修复循环退出逻辑）
                if not viewer.is_running():
                    break

                # 时长限制（保持原版）
                if duration and (time.time() - start_time) > duration:
                    print("⏱️  仿真时长结束，停止运动")
                    break

                step_start = time.time()
                t = time.time() - start_time

                # 生成轨迹+控制关节（保持原版）
                target_angles = self.generate_circle_trajectory(t)
                for joint_name in self.arm_joint_names:
                    self.smooth_set_joint_angle(joint_name, target_angles[joint_name])

                # 步进仿真
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # 修复计时逻辑（避免负睡眠）
                time_elapsed = time.time() - step_start
                time_until_next_step = max(0, self.sim_timestep - time_elapsed)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        finally:
            # 确保viewer正确关闭
            viewer.close()

        # 归位（简化版，避免归位过程出错）
        print("🔄 正在归位到初始姿态...")
        self.reset_to_initial_pose_simple()
        print("✅ 已归位到初始姿态")

    def reset_to_initial_pose_simple(self):
        """简化版归位逻辑（避免复杂依赖）"""
        initial_poses = {"joint1": 0.0, "joint2": 0.3, "joint3": -0.5,
                         "joint4": 0.0, "joint5": 0.0, "joint6": 0.0}

        # 简单归位：直接设置目标角度，步进仿真
        for _ in range(500):
            all_reached = True
            for joint_name, target in initial_poses.items():
                if not self.smooth_set_joint_angle(joint_name, target, speed=0.3):
                    all_reached = False
            mujoco.mj_step(self.model, self.data)
            time.sleep(self.sim_timestep)
            if all_reached:
                break

        # 停止所有执行器
        for aid in self.actuator_ids.values():
            self.data.ctrl[aid] = 0


def main():
    """主函数（保持原版使用方式）"""
    model_path = "arm6dof_final.xml"

    try:
        controller = RoboticArmController(model_path)
        print("✅ 机械臂控制器初始化成功！")
        print(f"🔧 绕圈参数：半径={controller.circle_radius}m，中心={controller.circle_center}")
        print("▶️  开始绕圈仿真（运行30秒）...")

        controller.simulate_circle_motion(duration=30.0)

    except FileNotFoundError as e:
        print(f"❌ 错误：{e}")
        print("💡 请确保 arm6dof_final.xml 文件在当前目录下")
    except Exception as e:
        print(f"❌ 发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()