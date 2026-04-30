import mujoco
import mujoco.viewer
import numpy as np
import os


class RoboticArmSafeController:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 固定关节-执行器映射，确保稳定
        self.arm_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.actuator_map = {j: f"motor_{j}" for j in self.arm_joints}

        # 🔧 安全参数（可根据需要微调）
        self.SAFE_HEIGHT = 0.15  # 安全高度阈值
        self.WARN_HEIGHT = 0.20  # 预警高度（提前提醒）
        self.target_joint2 = 1.8  # 下压目标角度
        self.safe_pose = {  # 安全归位姿态
            "joint2": 0.3,
            "joint3": -0.4
        }

    def pd_control(self, joint_name: str, target: float, kp: float = 80.0, kd: float = 8.0):
        """平滑PD控制，减少抖动"""
        act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.actuator_map[joint_name])
        if act_id == -1:
            return
        curr = self.data.joint(joint_name).qpos[0]
        err = target - curr
        ctrl = kp * err - kd * self.data.joint(joint_name).qvel[0]
        self.data.ctrl[act_id] = np.clip(ctrl, -6, 6)

    def smooth_interpolate(self, start: float, end: float, t: float) -> float:
        """三次样条平滑插值，让运动更柔和"""
        t = np.clip(t, 0, 1)
        return start + t * t * (3 - 2 * t) * (end - start)

    def check_safety(self) -> tuple[int, str]:
        """
        分级安全检测：
        0 = 安全
        1 = 预警（高度 < WARN_HEIGHT）
        2 = 紧急（高度 < SAFE_HEIGHT）
        """
        z = self.data.site("grip_center").xpos[2]
        if z < self.SAFE_HEIGHT:
            return 2, f"⚠️ 高度过低: {z:.3f}m，触发紧急停止！"
        elif z < self.WARN_HEIGHT:
            return 1, f"⚠️ 高度预警: {z:.3f}m，即将触底！"
        else:
            return 0, f"✅ 高度安全: {z:.3f}m"

    def gradual_safe_stop(self):
        """分阶段减速停止，避免震荡"""
        print("🛑 执行分级安全停止...")
        start_joint2 = self.data.joint("joint2").qpos[0]
        start_joint3 = self.data.joint("joint3").qpos[0]
        target_joint2 = self.safe_pose["joint2"]
        target_joint3 = self.safe_pose["joint3"]

        # 平滑插值归位
        for step in range(400):
            t = step / 400.0
            curr_j2 = self.smooth_interpolate(start_joint2, target_joint2, t)
            curr_j3 = self.smooth_interpolate(start_joint3, target_joint3, t)
            self.pd_control("joint2", curr_j2, kp=60)
            self.pd_control("joint3", curr_j3, kp=60)
            mujoco.mj_step(self.model, self.data)
        print("✅ 已平稳回到安全姿态")

    def demo_safe_motion(self):
        """完整演示：平滑下压 → 分级预警 → 安全停止"""
        # 初始稳定
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        print("🚀 开始平滑下压机械臂")
        start_joint2 = self.data.joint("joint2").qpos[0]
        total_steps = 800  # 下压总步数，控制速度

        for step in range(total_steps):
            # 平滑插值下压
            t = step / total_steps
            target_j2 = self.smooth_interpolate(start_joint2, self.target_joint2, t)
            self.pd_control("joint2", target_j2)
            mujoco.mj_step(self.model, self.data)

            # 分级检测与提示
            level, msg = self.check_safety()
            print(msg)

            # 触发紧急停止
            if level == 2:
                self.gradual_safe_stop()
                break

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 优化视角，更清晰观察末端
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -25
            viewer.cam.lookat = [0.0, 0.0, 0.2]

            # 运行优化后的演示
            self.demo_safe_motion()

            # 保持窗口
            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


if __name__ == "__main__":
    model_path = "arm6dof_final.xml"
    try:
        ctrl = RoboticArmSafeController(model_path)
        ctrl.run()
    except Exception as e:
        print(f"错误: {e}")