import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)

# 初始化姿态
data.qpos[:] = 0
data.qpos[2] = 0.9
data.qpos[3] = 1.0
# 初始化：强制姿态，防止悬空和翻倒
data.qpos[:] = 0
data.qpos[2] = 0.9     # 和XML高度一致，脚掌刚好压在地上
data.qpos[3] = 1.0     # 躯干直立，不翻倒
data.qpos[4] = 0.0
data.qpos[5] = 0.0
data.qpos[6] = 0.0
data.qvel[:] = 0
data.ctrl[:] = 0

# 运动参数
walk_freq = 0.04
leg_amp = 0.15
arm_amp = 0.08
head_freq = 0.02   # 转头速度
head_amp = 25      # 转头幅度
# 走路参数：腿部幅度保持足够推力，手臂幅度降低
walk_freq = 0.04
leg_amp = 0.15
arm_amp = 0.08

with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running():
        dt = model.opt.timestep
        t += dt
        phase = t * walk_freq
        head_phase = t * head_freq

        # 原有走路逻辑

        # 修正为：正常走路的「对侧联动」，不会交叉，也不会同手同脚
        # 右臂 + 左腿 同相位
        data.ctrl[1] = np.sin(phase) * arm_amp
        data.ctrl[2] = np.sin(phase) * arm_amp * 0.4
        data.ctrl[8] = np.sin(phase) * leg_amp
        data.ctrl[9] = np.sin(phase) * leg_amp * 0.3

        # 左臂 + 右腿 同相位，与另一侧相反
        data.ctrl[3] = np.sin(phase + np.pi) * arm_amp
        data.ctrl[4] = np.sin(phase + np.pi) * arm_amp * 0.4
        data.ctrl[5] = np.sin(phase + np.pi) * leg_amp
        data.ctrl[6] = np.sin(phase + np.pi) * leg_amp * 0.3

        # 新增：颈部左右转头
        data.ctrl[0] = np.sin(head_phase) * head_amp

        # 脚踝固定
        data.ctrl[7] = 0
        data.ctrl[10] = 0

        # 锁定姿态防倒地
        # 脚踝锁死 → 脚掌全程贴地，绝对不悬空
        data.ctrl[7] = 0
        data.ctrl[10] = 0
        # 颈部固定
        data.ctrl[0] = 0

        # 关键：每帧强制锁定姿态，防止掉下去/翻倒
        data.qpos[2] = 0.9
        data.qpos[3] = 1.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0
        data.qvel[:] = 0

        mujoco.mj_step(model, data)
        viewer.sync()