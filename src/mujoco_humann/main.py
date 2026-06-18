import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)

# 初始姿态
data.qpos[:] = 0
data.qpos[2] = 0.9   # 躯干高度
data.qpos[3] = 1.0   # 躯干俯仰
data.qpos[4] = 0.0
data.qpos[5] = 0.0
data.qpos[6] = 0.0
data.qvel[:] = 0
data.ctrl[:] = 0

# 步态参数【全部减速调整】
step_freq = 0.025       # 原0.05，减半，踏步节奏慢一倍
step_amp = 0.12
arm_amp = 0.03      
forward_speed = 0.007   # 原0.015，大幅降低平移速度

# 矩形路径参数
rect_length_x = 2.0    # 矩形长边
rect_length_y = 1.2    # 矩形短边
current_dir = 0        # 0:+X,1:+Y,2:-X,3:-Y 四个方向
segment_dist = 0.0     # 当前这条边已经走了的距离

with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running():
        dt = model.opt.timestep
        t += dt
        phase = t * step_freq

        # 腿部踏步控制
        data.ctrl[5] = np.sin(phase) * step_amp
        data.ctrl[6] = np.sin(phase) * step_amp * 0.3
        data.ctrl[8] = np.sin(phase + np.pi) * step_amp
        data.ctrl[9] = np.sin(phase + np.pi) * step_amp * 0.3

        # 小幅手臂摆动
        data.ctrl[1] = np.sin(phase + np.pi) * arm_amp
        data.ctrl[2] = np.sin(phase + np.pi) * arm_amp * 0.4
        data.ctrl[3] = np.sin(phase) * arm_amp
        data.ctrl[4] = np.sin(phase) * arm_amp * 0.4

        # 头部、脚踝固定不动
        data.ctrl[0] = 0
        data.ctrl[7] = 0
        data.ctrl[10] = 0

        # 矩形路径移动逻辑
        move_step = forward_speed * dt
        segment_dist += move_step

        if current_dir == 0:
            # 沿X正向走长边
            data.qpos[0] += move_step
            # 走完长边，切换向Y
            if segment_dist >= rect_length_x:
                current_dir = 1
                segment_dist = 0.0
        elif current_dir == 1:
            # 沿Y正向走短边
            data.qpos[1] += move_step
            if segment_dist >= rect_length_y:
                current_dir = 2
                segment_dist = 0.0
        elif current_dir == 2:
            # 沿X负向走长边
            data.qpos[0] -= move_step
            if segment_dist >= rect_length_x:
                current_dir = 3
                segment_dist = 0.0
        elif current_dir == 3:
            # 沿Y负向走短边
            data.qpos[1] -= move_step
            if segment_dist >= rect_length_y:
                current_dir = 0
                segment_dist = 0.0

        # 锁定躯干姿态防倒地
        data.qpos[2] = 0.9
        data.qpos[3] = 1.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0
        data.qvel[:] *= 0.98

        mujoco.mj_step(model, data)
        viewer.sync()