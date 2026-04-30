import mujoco
import time
import numpy as np
import os

try:
    # 尝试使用较新版本的mujoco.viewer
    import mujoco.viewer

    use_new_viewer = True
except ImportError:
    # 如果没有mujoco.viewer，则使用glfw_viewer（如果可用）
    try:
        from mujoco import glfw_viewer

        use_new_viewer = False
    except ImportError:
        # 如果都不可用，给出提示
        print("未找到可用的可视化模块，请确认安装了正确的MuJoCo版本")
        exit(1)

# 读取模型文件内容
model_path = os.path.join(os.path.dirname(__file__), "arm.xml")
with open(model_path, 'r', encoding='utf-8') as f:
    model_xml = f.read()

# 从字符串加载模型
model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)

# 设置到home位置
data.qpos[:] = [0, 0.785398, 0, -1.5708, 0, 0, 0]
mujoco.mj_forward(model, data)


def move_arm_in_wave(data, t):
    """生成波动运动轨迹"""
    # 基础正弦波运动
    for i in range(7):
        data.ctrl[i] = np.sin(t + i * np.pi / 4) * 0.5


if use_new_viewer:
    # 使用新版本的viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("按ESC键退出")
        t = 0
        while viewer.is_running():
            step_start = time.time()

            # 更新控制信号，使机械臂运动
            move_arm_in_wave(data, t)
            t += 0.01

            mujoco.mj_step(model, data)
            viewer.sync()

            # 控制步频
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
else:
    # 使用glfw_viewer
    viewer = glfw_viewer.GlfwViewer(model, data)
    print("按ESC键退出")
    t = 0
    while viewer.is_running():
        step_start = time.time()

        # 更新控制信号，使机械臂运动
        move_arm_in_wave(data, t)
        t += 0.01

        mujoco.mj_step(model, data)
        viewer.render()

        # 控制步频
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    viewer.close()