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

    def clamp_joint_angle(self, joint_name: str, angle: float) -> float:
        """将关节角度限制在合法范围内"""
        min_angle, max_angle = self.joint_limits[joint_name]
        return np.clip(angle, min_angle, max_angle)

    def set_joint_angle(self, joint_name: str, target_angle: float, speed: float = 0.5):
        """
        平滑设置单个关节角度
        :param joint_name: 关节名称
        :param target_angle: 目标角度（弧度）
        :param speed: 运动速度（弧度/秒）
        """
        if joint_name not in self.arm_joint_ids:
            raise ValueError(f"关节 {joint_name} 不存在")

        target_angle = self.clamp_joint_angle(joint_name, target_angle)
        actuator_id = self.actuator_ids[f"motor_{joint_name}"]

        # 获取当前角度
        current_angle = self.data.joint(joint_name).qpos[0]

        # 计算角度差
        angle_diff = target_angle - current_angle

        # 如果角度差很小，直接设置
        if abs(angle_diff) < 0.001:
            self.data.ctrl[actuator_id] = 0
            return True

        # PD控制器
        kp = 50.0
        kd = 5.0
        error = angle_diff
        d_error = -self.data.joint(joint_name).qvel[0]
        control = kp * error + kd * d_error

        # 限制控制信号
        control = np.clip(control, -speed * 10, speed * 10)
        self.data.ctrl[actuator_id] = control

        return False

    def rotate_joint_continuously(self, joint_name: str, direction: float = 1.0, speed: float = 0.5):
        """
        让关节持续旋转（核心功能）
        """
        if joint_name not in self.arm_joint_ids:
            raise ValueError(f"关节 {joint_name} 不存在")

        actuator_id = self.actuator_ids[f"motor_{joint_name}"]
        self.data.ctrl[actuator_id] = direction * speed * 10

    def simulate(self, duration: float = None):
        """运行仿真"""
        start_time = time.time()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 调整相机视角，方便观察
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20
            viewer.cam.lookat = [0.1, 0.0, 0.2]

            while viewer.is_running():
                if duration and (time.time() - start_time) > duration:
                    break

                step_start = time.time()
                t = time.time() - start_time

                # ========== 核心运动逻辑：关节1持续旋转，其他关节配合摆动 ==========
                # 关节1：持续顺时针旋转（核心需求）
                self.rotate_joint_continuously("joint1", direction=1.0, speed=0.3)

                # 关节2：缓慢正弦摆动
                self.set_joint_angle("joint2", 0.5 * np.sin(t * 0.5))

                # 关节3：缓慢余弦摆动
                self.set_joint_angle("joint3", -0.5 + 0.3 * np.cos(t * 0.5))

                # 步进仿真
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # 控制仿真速率
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main():
    """主函数 - 已修正为你的XML文件名"""
    # 关键修正：使用你的实际模型文件名 arm6dof_final.xml
    model_path = "arm6dof_final.xml"

    try:
        controller = RoboticArmController(model_path)
        print("✅ 机械臂控制器初始化成功！")
        print("🔧 关节列表：", controller.arm_joint_names)
        print("▶️  开始仿真，关节1将持续旋转...")
        controller.simulate(duration=30.0)  # 运行30秒

    except FileNotFoundError as e:
        print(f"❌ 错误：{e}")
        print("💡 请确保 arm6dof_final.xml 和 control_locus.py 在同一个文件夹里")
    except Exception as e:
        print(f"❌ 发生错误：{e}")


if __name__ == "__main__":
    main()