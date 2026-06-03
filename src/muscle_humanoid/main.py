import mujoco
import mujoco.viewer as viewer
import numpy as np
import os

# 步态参数 进一步减速
GAIT_FREQ = 0.03       # 步态节奏更慢
BASE_STEP_AMP = 0.15   # 更小步幅
BASE_ARM_AMP = 0.20    # 小幅摆臂
ELBOW_AMP = 0.3
KNEE_AMP = 0.35
BOUNCE_AMP = 0.02

# 移动边界
POS_LIMIT = 3.0

class HumanoidStableEnv:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.reset()
        self.standup_progress = 0.0  

        # 相机视角
        self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 5.0
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 90

    def reset(self):
        """复位：初始深度蹲下，重新开始起立"""
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[2] = 0.65   # 蹲得更深，视觉更明显
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[3] = 0.0
        self.data.qpos[4] = 0.0
        self.standup_progress = 0.0
        print("\n>>> 重置完成，开始【深度蹲下 → 超缓慢起立】<<<")

    def step(self, phase):
        # 越界自动复位
        if abs(self.data.qpos[0]) > POS_LIMIT or abs(self.data.qpos[1]) > POS_LIMIT:
            self.reset()

        # 起立阶段：总时长拉到 5000 步，过程极慢，方便观察
        if self.standup_progress < 1.0:
            self.standup_progress += 0.0002  # 单步增量极小，起立很慢
            target_height = 0.65 + 0.55 * self.standup_progress
            self.data.qpos[2] = target_height
            
            # 蹲下/起立过程：膝盖弯曲、肢体不动
            left_hip = 0
            right_hip = 0
            left_knee = 0.9
            right_knee = 0.9
            left_arm = 0
            right_arm = 0
            left_elbow = 0
            right_elbow = 0
            move_speed = 0.0  # 起立阶段完全不移动

        else:
            # 起立完成：超慢速行走
            left_hip = BASE_STEP_AMP * np.sin(phase)
            right_hip = BASE_STEP_AMP * np.sin(phase + np.pi)
            left_knee = KNEE_AMP * np.clip(np.sin(phase + np.pi/2), 0, 1)
            right_knee = KNEE_AMP * np.clip(np.sin(phase - np.pi/2), 0, 1)

            left_arm = BASE_ARM_AMP * np.sin(phase + np.pi)
            right_arm = BASE_ARM_AMP * np.sin(phase)
            left_elbow = ELBOW_AMP * np.abs(np.sin(phase + np.pi))
            right_elbow = ELBOW_AMP * np.abs(np.sin(phase))

            self.data.qpos[2] = 1.2 + BOUNCE_AMP * np.abs(np.cos(phase))
            move_speed = 0.0001  # 行走移动速度降到最低

        # 赋值关节
        self.data.qpos[self.model.joint("left_hip_pitch").qposadr] = left_hip
        self.data.qpos[self.model.joint("right_hip_pitch").qposadr] = right_hip
        self.data.qpos[self.model.joint("left_knee").qposadr] = left_knee
        self.data.qpos[self.model.joint("right_knee").qposadr] = right_knee

        self.data.qpos[self.model.joint("left_shoulder_pitch").qposadr] = left_arm
        self.data.qpos[self.model.joint("right_shoulder_pitch").qposadr] = right_arm
        self.data.qpos[self.model.joint("left_elbow").qposadr] = left_elbow
        self.data.qpos[self.model.joint("right_elbow").qposadr] = right_elbow

        # 向前移动
        self.data.qpos[1] += move_speed

        # 姿态限制，防止倒立、翻转
        self.data.qpos[self.model.joint("root").qposadr + 3] = np.clip(self.data.qpos[3], -0.2, 0.2)
        self.data.qpos[4] = np.clip(self.data.qpos[4], -1.5, 1.5)

        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

        return self.data.qpos[3], self.standup_progress

    def close(self):
        self.viewer.close()

def main():
    xml_path = os.path.join(os.path.dirname(__file__), "humanoid.xml")
    env = HumanoidStableEnv(xml_path)
    total_step = 0

    print("===== 深度蹲下 + 超长慢速起立 + 极慢行走 =====")
    print("流程：深蹲 → 缓慢站直 → 低速前行，全过程清晰可见\n")

    try:
        while env.viewer.is_running():
            total_step += 1
            t = total_step * env.model.opt.timestep
            phase = 2 * np.pi * GAIT_FREQ * t

            tilt, progress = env.step(phase)

            if total_step % 50 == 0:
                if progress < 1.0:
                    print(f"步数:{total_step:04d} | 起立进度:{progress:.1%}")
                else:
                    print(f"步数:{total_step:04d} | 正常慢速行走中")

    except KeyboardInterrupt:
        print("\n模拟终止")
    finally:
        env.close()
        print("环境关闭")

if __name__ == "__main__":
    main()