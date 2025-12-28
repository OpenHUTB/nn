import mujoco
import mujoco.viewer
import numpy as np
import time
import glfw


def load_and_visualize():
    # 加载模型
    model = mujoco.MjModel.from_xml_path('new_arm.xml')
    data = mujoco.MjData(model)

    # 使用查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置相机视角
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]

        print("机械臂模型加载成功！")
        print(f"自由度数量: {model.nv}")
        print(f"关节数量: {model.njnt}")
        print(f"执行器数量: {model.nu}")

        # 设置控制模式
        print("\n控制模式:")
        print("1: 静止")
        print("2: 正弦运动")
        print("3: 随机运动")
        print("4: 夹爪开合")
        print("按 ESC 退出")

        mode = 1  # 默认静止模式
        start_time = time.time()

        while viewer.is_running():
            step_start = time.time()

            # 根据模式设置控制信号
            current_time = time.time() - start_time

            if mode == 1:  # 静止
                data.ctrl[:] = 0

            elif mode == 2:  # 正弦运动
                # 创建小幅度平滑的正弦运动，速度放慢10000倍
                for i in range(6):
                    amplitude = abs(0.1 + i * 0.05)  # 小幅度运动，确保幅度为正数
                    frequency = abs(0.5 + i * 0.1) / 10000.0  # 频率降低10000倍，确保为正数
                    phase = abs(i * 0.5)  # 确保相位为正数
                    # 添加偏置确保关节不会过度向下运动
                    if i == 1:  # 肩关节，避免向下过度运动
                        data.ctrl[i] = 0.1 + amplitude * np.sin(2 * np.pi * frequency * current_time + phase)
                    elif i == 2:  # 肘关节，避免向下过度运动
                        data.ctrl[i] = 0.1 + amplitude * np.sin(2 * np.pi * frequency * current_time + phase)
                    else:
                        data.ctrl[i] = amplitude * np.sin(2 * np.pi * frequency * current_time + phase)

                # 固定夹爪
                data.ctrl[6] = 0.02
                data.ctrl[7] = 0.02

            elif mode == 3:  # 随机运动
                if int(current_time * 2 / 10000.0) % 10 == 0:  # 更新频率降低10000倍
                    # 生成小幅度随机运动，但对关节2和3添加偏置避免向下过度运动
                    random_values = np.abs(np.random.uniform(-0.2, 0.2, 6))
                    # 确保肩关节和肘关节不会过度向下
                    random_values[1] = 0.1 + random_values[1] * 0.5  # 肩关节偏置
                    random_values[2] = 0.1 + random_values[2] * 0.5  # 肘关节偏置
                    data.ctrl[:6] = random_values

            elif mode == 4:  # 夹爪开合
                # 固定关节位置，确保不会低于底座
                data.ctrl[:6] = [0, 0.2, 0.1, 0, 0, 0]  # 调整关节位置，确保不会低于底座
                # 夹爪开合，速度放慢10000倍
                grip_pos = abs(0.01) * np.sin(2 * np.pi * abs(0.5) * current_time / 10000.0)
                data.ctrl[6] = grip_pos
                data.ctrl[7] = grip_pos

            # 执行物理仿真步
            mujoco.mj_step(model, data)

            # 同步查看器
            viewer.sync()

            # 处理按键输入
            # 注意：在新版本MuJoCo中，访问窗口上下文的方式已改变
            # 这里暂时注释掉按键检测，避免报错
            # if glfw.get_key(viewer._context.window, glfw.KEY_1) == glfw.PRESS:
            #     mode = 1
            #     print("切换到模式1: 静止")
            # elif glfw.get_key(viewer._context.window, glfw.KEY_2) == glfw.PRESS:
            #     mode = 2
            #     print("切换到模式2: 正弦运动")
            # elif glfw.get_key(viewer._context.window, glfw.KEY_3) == glfw.PRESS:
            #     mode = 3
            #     print("切换到模式3: 随机运动")
            # elif glfw.get_key(viewer._context.window, glfw.KEY_4) == glfw.PRESS:
            #     mode = 4
            #     print("切换到模式4: 夹爪开合")

            # 控制帧率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def simulate_trajectory():
    """演示一个预设的轨迹"""
    model = mujoco.MjModel.from_xml_path('new_arm.xml')
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]

        print("执行预设轨迹...")
        print("1. 移动到初始位置")
        print("2. 执行拾取动作")
        print("3. 返回初始位置")

        steps = 0
        trajectory_phase = 0
        start_time = time.time()

        while viewer.is_running():
            current_time = time.time() - start_time

            # 定义轨迹
            if trajectory_phase == 0:  # 初始位置
                target_pos = [0, 0, 0, 0, 0, 0, 0.02, 0.02]
                if steps > 100:
                    trajectory_phase = 1
                    steps = 0

            elif trajectory_phase == 1:  # 拾取位置 - 确保不低于底座
                target_pos = [0.1, 0.15, 0.05, 0.05, 0, 0, 0.02, 0.02]
                if steps > 150:
                    trajectory_phase = 2
                    steps = 0

            elif trajectory_phase == 2:  # 夹取物体 - 确保不低于底座
                target_pos = [0.1, 0.15, 0.05, 0.05, 0, 0, -0.01, -0.01]
                if steps > 50:
                    trajectory_phase = 3
                    steps = 0

            elif trajectory_phase == 3:  # 抬起
                target_pos = [0.1, 0.15, 0.15, 0.05, 0, 0, -0.01, -0.01]
                if steps > 100:
                    trajectory_phase = 4
                    steps = 0

            elif trajectory_phase == 4:  # 返回
                target_pos = [0, 0, 0, 0, 0, 0, -0.01, -0.01]
                if steps > 150:
                    trajectory_phase = 5
                    steps = 0

            elif trajectory_phase == 5:  # 释放
                target_pos = [0, 0, 0, 0, 0, 0, 0.02, 0.02]
                if steps > 50:
                    # 重新开始
                    trajectory_phase = 0
                    steps = 0

            # 使用PD控制器达到目标位置
            kp = 10.0  # 比例增益
            for i in range(8):
                error = target_pos[i] - data.qpos[i]
                data.ctrl[i] = kp * error

            mujoco.mj_step(model, data)
            viewer.sync()

            steps += 1
            time.sleep(0.01)


def get_robot_info():
    """获取机械臂信息"""
    model = mujoco.MjModel.from_xml_path('new_arm.xml')

    print("=" * 50)
    print("机械臂模型信息")
    print("=" * 50)
    print(f"模型名称: {model.names}")
    print(f"自由度数量: {model.nv}")
    print(f"关节数量: {model.njnt}")
    print(f"执行器数量: {model.nu}")
    print(f"几何体数量: {model.ngeom}")
    print(f"体数量: {model.nbody}")

    print("\n关节列表:")
    for i in range(model.njnt):
        jnt_type = model.jnt_type[i]
        print(f"  关节{i}: {model.joint(i).name}, 类型: {jnt_type}")

    print("\n执行器列表:")
    for i in range(model.nu):
        actuator_name = model.actuator(i).name
        joint_id = model.actuator_trnid[i, 0]
        joint_name = model.joint(joint_id).name
        print(f"  执行器{i}: {actuator_name}, 控制的关节: {joint_name}")

    print("\n体列表:")
    for i in range(model.nbody):
        body_name = model.body(i).name
        if body_name:
            print(f"  体{i}: {body_name}")


if __name__ == "__main__":
    print("MuJoCo 机械臂可视化")
    print("=" * 50)

    # 显示模型信息
    get_robot_info()

    print("\n选择运行模式:")
    print("1: 交互式可视化")
    print("2: 预设轨迹演示")

    try:
        choice = int(input("请输入选择 (1 或 2): "))

        if choice == 1:
            load_and_visualize()
        elif choice == 2:
            simulate_trajectory()
        else:
            print("无效选择，使用默认模式...")
            load_and_visualize()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")