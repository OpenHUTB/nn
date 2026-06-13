import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)

# 初始化姿态
data.qpos[:] = 0
data.qpos[2] = 0.9     # 初始高度，保证脚掌贴地
data.qpos[3] = 1.0     # 躯干直立，不翻倒
data.qpos[4] = 0.0
data.qpos[5] = 0.0
data.qpos[6] = 0.0
data.qvel[:] = 0
data.ctrl[:] = 0

# 走路参数：步频和幅度配合，产生足够前进动力
walk_freq = 0.04
leg_amp = 0.18    # 腿部幅度加大，产生足够推力
arm_amp = 0.08   # 手臂小幅度前后摆动，不交叉

with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running():
        dt = model.opt.timestep
        t += dt
        phase = t * walk_freq

        # 正常对侧联动（右手+左脚，左手+右脚）
        # 右臂 + 左腿
        data.ctrl[1] = np.sin(phase) * arm_amp
        data.ctrl[2] = np.sin(phase) * arm_amp * 0.4
        data.ctrl[8] = np.sin(phase) * leg_amp
        data.ctrl[9] = np.sin(phase) * leg_amp * 0.3

        # 左臂 + 右腿
        data.ctrl[3] = np.sin(phase + np.pi) * arm_amp
        data.ctrl[4] = np.sin(phase + np.pi) * arm_amp * 0.4
        data.ctrl[5] = np.sin(phase + np.pi) * leg_amp
        data.ctrl[6] = np.sin(phase + np.pi) * leg_amp * 0.3

        # 脚踝锁死，脚掌不悬空
        data.ctrl[7] = 0
        data.ctrl[10] = 0
        # 颈部固定
        data.ctrl[0] = 0

        # 只锁死高度和躯干角度，不清零速度，保留前进动力
        data.qpos[2] = 0.9
        data.qpos[3] = 1.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()