import mujoco
import mujoco.viewer
import numpy as np

# 直接调用 MuJoCo 官方自带的 humanoid 模型，绝对正确
model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)

# 用模型自带的关键帧初始化，直接就是标准站立姿态！
mujoco.mj_resetDataKeyframe(model, data, 0)

# PD控制参数（官方推荐的稳定参数）
kp = 1000.0
kd = 100.0

# 启动仿真
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 调整视角，方便观察
    viewer.cam.distance = 5
    viewer.cam.elevation = -25
    viewer.cam.lookat[:] = [0, 0, 1.2]

    while viewer.is_running():
        # 获取关节位置和速度
        q = data.qpos[7: 7 + model.nu]
        v = data.qvel[6: 6 + model.nu]

        # PD控制，目标就是初始的站立姿态
        data.ctrl[:] = kp * (np.zeros_like(q) - q) - kd * v

        mujoco.mj_step(model, data)
        viewer.sync()