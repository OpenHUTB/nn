import mujoco
import mujoco.viewer
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)
nu = model.nu
print(f"模型控制维度 nu = {nu}")

# 关键：抬高初始高度，让机器人双脚扎实踩地，不悬空不下陷
data.qpos[2] = 1.4
mujoco.mj_forward(model, data)

# 高摩擦力，牢牢抓地
for i in range(model.ngeom):
    if "floor" in model.geom(i).name:
        model.geom(i).friction = [120, 0.1, 0.1]

with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running():
        dt = model.opt.timestep
        t += dt

        # 右手保持抬起、手肘极慢挥手（和你原代码完全一致）
        data.ctrl[1] = 0.3
        elbow_wave = np.sin(t * 0.08) * 0.35 + 0.35
        data.ctrl[2] = elbow_wave

        # 左手超慢速、极小幅度轻晃（保留原样）
        left_swing = np.sin(t * 0.1) * 0.06
        data.ctrl[3] = left_swing
        data.ctrl[4] = left_swing * 0.1

        # 头部保持固定
        data.ctrl[0] = 0

        mujoco.mj_step(model, data)
        viewer.sync()