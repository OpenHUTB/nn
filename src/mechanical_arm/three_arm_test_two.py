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
model_path = os.path.join(os.path.dirname(__file__), "three_fingered_arm.xml")
with open(model_path, 'r', encoding='utf-8') as f:
    model_xml = f.read()

# 从字符串加载模型
model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)

# 设置到home位置
data.qpos[:] = [0, 0.785398, 0, -1.5708, 0, 0, 0, 0, 0, 0]
mujoco.mj_forward(model, data)

if use_new_viewer:
    # 使用新版本的viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("按ESC键退出")
        print("演示三指夹爪开合动作...")

        # 记录初始时间用于动画控制
        start_time = time.time()

        while viewer.is_running():
            step_start = time.time()

            # 控制夹爪开合的简单动画
            elapsed = time.time() - start_time
            gripper_position = 0.09 * (1 + np.sin(elapsed))  # 增大运动幅度以适应更长的手指

            # 确保位置在有效范围内
            gripper_position = max(0, min(0.12, gripper_position))

            # 设置夹爪关节位置
            data.ctrl[7] = gripper_position  # finger_joint1
            data.ctrl[8] = gripper_position  # finger_joint2
            data.ctrl[9] = gripper_position  # finger_joint3

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
    print("演示三指夹爪开合动作...")

    # 记录初始时间用于动画控制
    start_time = time.time()

    while viewer.is_running():
        step_start = time.time()

        # 控制夹爪开合的简单动画
        elapsed = time.time() - start_time
        gripper_position = 0.02 * (1 + np.sin(elapsed))

        # 确保位置在有效范围内
        gripper_position = max(0, min(0.04, gripper_position))

        # 设置夹爪关节位置
        data.ctrl[7] = gripper_position  # finger_joint1
        data.ctrl[8] = gripper_position  # finger_joint2
        data.ctrl[9] = gripper_position  # finger_joint3

        mujoco.mj_step(model, data)
        viewer.render()

        # 控制步频
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    viewer.close()