import mujoco
import mujoco.viewer as viewer
import numpy as np
import os

def main():
    xml_path = os.path.join(os.path.dirname(__file__), "humanoid.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 固定模型坐标，锁定在画面中心
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 1.0
    data.qpos[3] = 0.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0

    time = 0.0
    swing_freq = 0.15
    body_amp = 0.07   # 身体俯仰（走路）
    arm_amp = 0.06    # 左右转动（模拟摆手）

    v = viewer.launch_passive(model, data)
    # 相机居中对准模型，画面观感舒适
    v.cam.distance = 3.2
    v.cam.elevation = -15
    v.cam.azimuth = 90
    v.cam.lookat[:] = [0, 0, 0.8]  # 镜头焦点精准落在模型中心

    while v.is_running():
        time += model.opt.timestep
        phase = np.sin(2 * np.pi * swing_freq * time)

        # 走路+摆臂动作
        data.qpos[3] = body_amp * phase
        data.qpos[5] = arm_amp * phase

        # 强制锁死位置，保证始终在中间
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = 1.0
        data.qpos[4] = 0.0

        mujoco.mj_forward(model, data)
        v.sync()

if __name__ == "__main__":
    main()