import mujoco
import mujoco.viewer
import numpy as np

# 模型路径
model_path = "src/mujoco01/humanoid.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 关键：初始关键帧站姿，防止一开局躺地
mujoco.mj_resetDataKeyframe(model, data, 0)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 固定下肢+躯干，保持站立不倒下
    while viewer.is_running():
        t = data.time

        # 平缓抬手循环动作
        lift = 0.4 * (1 - np.cos(t * 0.7))

        # 只使用安全下标，绝不越界
        data.ctrl[19] = lift

        # 固定下肢关键电机，强制站稳
        data.ctrl[4]  = 0.0
        data.ctrl[9]  = 0.0
        data.ctrl[6]  = 0.0
        data.ctrl[11] = 0.0
        data.ctrl[0]  = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()