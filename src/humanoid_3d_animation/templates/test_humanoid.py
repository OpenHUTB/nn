import mujoco
import mujoco.viewer as viewer
import time

# 加载模型
model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)


# 启动可视化界面
with viewer.launch_passive(model, data) as v:
    # 仿真循环
    while v.is_running():
        step_start = time.time()
        # 执行一步仿真
        mujoco.mj_step(model, data)
        # 同步帧率
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        # 更新视图
        v.sync()