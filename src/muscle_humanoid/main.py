import mujoco
import mujoco.viewer as viewer
import numpy as np
import os

# ===================== 全局配置 =====================
GAIT_FREQ = 0.12   # 动作节奏
STEP_AMP = 0.22    # 步幅
ARM_AMP = 0.35     # 摆臂幅度（肉眼清晰可见）
ELBOW_AMP = 0.3    # 肘部弯曲幅度
KNEE_AMP = 0.35    # 膝盖弯曲幅度
BOUNCE_AMP = 0.02  # 身体轻微起伏

class HumanoidEnv:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # 初始化姿态：手臂自然下垂，双腿直立
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = 1.2  # 初始高度

        # 相机视角：正对前方，能看到全身
        self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 4.5
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 90
        self.viewer.cam.lookat[:] = [0, 0, 0.8]

    def step(self, phase):
        """自然走路动作：前后迈步 + 反向摆臂 + 肘部联动"""
        # 1. 腿部：交替前后迈步
        left_hip = STEP_AMP * np.sin(phase)
        right_hip = STEP_AMP * np.sin(phase + np.pi)
        
        # 膝盖弯曲：抬脚时弯，落地时直
        left_knee = KNEE_AMP * np.clip(np.sin(phase + np.pi/2), 0, 1)
        right_knee = KNEE_AMP * np.clip(np.sin(phase - np.pi/2), 0, 1)

        self.data.qpos[self.model.joint("left_hip_pitch").qposadr] = left_hip
        self.data.qpos[self.model.joint("right_hip_pitch").qposadr] = right_hip
        self.data.qpos[self.model.joint("left_knee").qposadr] = left_knee
        self.data.qpos[self.model.joint("right_knee").qposadr] = right_knee

        # 2. 手臂：与腿部反向摆动（迈左腿，摆右臂）
        # 手臂从下垂位置前后摆动，不是举着
        left_arm = ARM_AMP * np.sin(phase + np.pi)
        right_arm = ARM_AMP * np.sin(phase)
        # 肘部同步弯曲：抬手时手肘弯起，放下时伸直
        left_elbow = ELBOW_AMP * np.abs(np.sin(phase + np.pi))
        right_elbow = ELBOW_AMP * np.abs(np.sin(phase))

        self.data.qpos[self.model.joint("left_shoulder_pitch").qposadr] = left_arm
        self.data.qpos[self.model.joint("right_shoulder_pitch").qposadr] = right_arm
        self.data.qpos[self.model.joint("left_elbow").qposadr] = left_elbow
        self.data.qpos[self.model.joint("right_elbow").qposadr] = right_elbow

        # 3. 身体轻微起伏 + 俯仰，模拟重心变化
        self.data.qpos[2] = 1.2 + BOUNCE_AMP * np.abs(np.cos(phase))
        self.data.qpos[self.model.joint("root").qposadr + 3] = -0.04 * np.sin(phase)

        # 强制锁定水平位置，防止模型跑丢
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0

        # 渲染画面
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    def close(self):
        self.viewer.close()

def main():
    xml_path = os.path.join(os.path.dirname(__file__), "humanoid.xml")
    env = HumanoidEnv(xml_path)

    total_step = 0
    print("===== 正面自然行走 | 手臂自然摆动 =====")
    print("动作：交替迈步 + 反向摆臂 + 肘部联动\n")

    try:
        while env.viewer.is_running():
            total_step += 1
            time = total_step * env.model.opt.timestep
            phase = 2 * np.pi * GAIT_FREQ * time

            env.step(phase)

            if total_step % 50 == 0:
                print(f"步数:{total_step:04d}")

    except KeyboardInterrupt:
        print("\n模拟终止")
    finally:
        env.close()
        print("环境已关闭")

if __name__ == "__main__":
    main()