import mujoco
import mujoco.viewer
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)
nu = model.nu
print(f"模型控制维度 nu = {nu}")

# 初始抬高，防止陷地
data.qpos[2] = 1.4
mujoco.mj_forward(model, data)

# 增加地面摩擦力，防止滑倒
for i in range(model.ngeom):
    if "floor" in model.geom(i).name:
        model.geom(i).friction = [10, 0.1, 0.1]

# PD控制参数
kp = 100
kd = 10

with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running():
        dt = model.opt.timestep
        t += dt

        # 手臂摆动幅度
        swing = np.sin(t * 1.2) * 0.3
        target = np.zeros(nu)

        # 适配nu=16的手臂关节索引
        if nu >= 16:
            # 右肩+右肘（索引10、11、12）
            target[10] = swing
            target[11] = swing
            target[12] = swing * 0.4
            # 左肩+左肘（索引13、14、15）
            target[13] = -swing
            target[14] = -swing
            target[15] = -swing * 0.4

        # 获取关节状态
        q = data.qpos[7:7+nu]
        v = data.qvel[6:6+nu]
        data.ctrl[:] = kp * (target - q) - kd * v

        mujoco.mj_step(model, data)
        viewer.sync()