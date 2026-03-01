import mujoco
import mujoco.viewer
import numpy as np
import time
from typing import List, Dict
import os


class RoboticArmController:
    def __init__(self, model_path: str):
        """初始化机械臂控制器"""
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载模型和数据
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 获取关节名称列表
        self.arm_joint_names = [
            "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6"
        ]

        # 获取关节ID
        self.arm_joint_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.arm_joint_names
        }

        # 获取执行器ID
        self.actuator_ids = {
            f"motor_{name}": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}")
            for name in self.arm_joint_names
        }

        # 关节角度限制（弧度）
        self.joint_limits = {
            "joint1": (-np.pi, np.pi),
            "joint2": (-2.0, 2.0),
            "joint3": (-2.5, 0.5),
            "joint4": (-2.0, 2.0),
            "joint5": (-1.8, 1.8),
            "joint6": (-1.8, 1.8)
        }

        # 绕圈参数配置
        self.circle_radius = 0.15  # 绕圈半径（米）
        self.circle_center = [0.2, 0.0, 0.2]  # 绕圈中心点（x,y,z）
        self.circle_speed = 1.0  # 绕圈角速度（弧度/秒）
        self.last_angles = None  # 记录上一帧关节角度，用于平滑

    def clamp_joint_angle(self, joint_name: str, angle: float) -> float:
        """将关节角度限制在合法范围内"""
        min_angle, max_angle = self.joint_limits[joint_name]
        return np.clip(angle, min_angle, max_angle)

    def smooth_set_joint_angle(self, joint_name: str, target_angle: float, speed: float = 0.8):
        """
        平滑设置单个关节角度（优化版，增加平滑过渡）
        :param joint_name: 关节名称
        :param target_angle: 目标角度（弧度）
        :param speed: 运动速度（弧度/秒）
        :return: 是否到达目标角度
        """
        if joint_name not in self.arm_joint_ids:
            raise ValueError(f"关节 {joint_name} 不存在")

        target_angle = self.clamp_joint_angle(joint_name, target_angle)
        actuator_id = self.actuator_ids[f"motor_{joint_name}"]

        # 获取当前角度和速度
        current_angle = self.data.joint(joint_name).qpos[0]
        current_vel = self.data.joint(joint_name).qvel[0]

        # 计算角度差（考虑角度周期性）
        angle_diff = target_angle - current_angle
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # 归一化到[-π, π]

        # 如果角度差很小，直接停止
        if abs(angle_diff) < 0.001:
            self.data.ctrl[actuator_id] = 0
            return True

        # 优化的PD控制器（增加阻尼，减少抖动）
        kp = 60.0
        kd = 8.0
        control = kp * angle_diff - kd * current_vel

        # 限制控制信号（根据速度参数）
        max_control = speed * 15
        control = np.clip(control, -max_control, max_control)
        self.data.ctrl[actuator_id] = control

        return False

    def generate_circle_trajectory(self, t: float):
        """
        生成绕圈轨迹的目标关节角度
        :param t: 时间参数（秒）
        :return: 各关节目标角度字典
        """
        # 计算绕圈的角度（周期性变化，实现左右绕圈）
        theta = self.circle_speed * t
        # 生成环形轨迹的末端位置（x-y平面绕圈，z轴保持恒定）
        target_x = self.circle_center[0] + self.circle_radius * np.cos(theta)
        target_y = self.circle_center[1] + self.circle_radius * np.sin(theta)
        target_z = self.circle_center[2]

        # 简化版逆运动学（适配6DOF机械臂，核心是关节1-3配合实现末端绕圈）
        # 关节1：跟随y轴位置变化，实现左右旋转
        joint1_target = np.arctan2(target_y, target_x - self.circle_center[0])

        # 关节2：配合高度，保持末端z轴稳定
        joint2_target = 0.6 + 0.2 * np.cos(theta)

        # 关节3：补偿关节2的运动，保持末端水平
        joint3_target = -0.8 + 0.2 * np.sin(theta)

        # 关节4-6保持固定姿态（可根据需要调整）
        joint4_target = 0.0
        joint5_target = 0.0
        joint6_target = 0.0

        return {
            "joint1": joint1_target,
            "joint2": joint2_target,
            "joint3": joint3_target,
            "joint4": joint4_target,
            "joint5": joint5_target,
            "joint6": joint6_target
        }

    def simulate_circle_motion(self, duration: float = None):
        """运行绕圈运动仿真（核心优化功能）"""
        start_time = time.time()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 优化相机视角，清晰观察绕圈运动
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 30
            viewer.cam.elevation = -15
            viewer.cam.lookat = self.circle_center  # 相机对准绕圈中心

            print(f"🔄 开始机械臂绕圈运动，绕圈中心：{self.circle_center}，半径：{self.circle_radius}m")
            print("🛑 关闭窗口停止仿真...")

            while viewer.is_running():
                if duration and (time.time() - start_time) > duration:
                    print("⏱️  仿真时长结束，停止运动")
                    break

                step_start = time.time()
                t = time.time() - start_time

                # 1. 生成绕圈轨迹的目标关节角度
                target_angles = self.generate_circle_trajectory(t)

                # 2. 逐个关节平滑控制，实现协同绕圈
                for joint_name in self.arm_joint_names[:6]:  # 控制前6个关节
                    self.smooth_set_joint_angle(joint_name, target_angles[joint_name])

                # 3. 步进仿真
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # 4. 控制仿真速率（保持实时性）
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            # 仿真结束，归位到初始姿态
            print("🔄 正在归位到初始姿态...")
            self.reset_to_initial_pose(viewer)

    def reset_to_initial_pose(self, viewer):
        """归位到初始安全姿态"""
        initial_poses = {
            "joint1": 0.0,
            "joint2": 0.3,
            "joint3": -0.5,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0
        }

        # 平滑归位
        for _ in range(500):
            all_reached = True
            for joint_name, target in initial_poses.items():
                if not self.smooth_set_joint_angle(joint_name, target, speed=0.3):
                    all_reached = False
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)
            if all_reached:
                break
        print("✅ 已归位到初始姿态")


def main():
    """主函数 -  机械臂绕圈运动"""
    model_path = "arm6dof_final.xml"

    try:
        controller = RoboticArmController(model_path)
        print("✅ 机械臂控制器初始化成功！")
        print(f"🔧 绕圈参数：半径={controller.circle_radius}m，中心={controller.circle_center}")
        print("▶️  开始绕圈仿真（运行30秒）...")

        # 运行绕圈仿真 （30秒）
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