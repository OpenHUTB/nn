#!/usr/bin/env python3
"""
完整版三指夹爪控制演示
此脚本展示了多种控制三指夹爪的方法：
1. 同步开合
2. 逐个手指控制
3. 波浪式动作
"""

import mujoco
import mujoco.viewer
import numpy as np
import time


def main():
    # 加载模型
    model_path = "three_fingered_arm.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 获取夹爪关节的ID
    finger_joint1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    finger_joint2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    finger_joint3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint3")

    # 弯曲关节
    finger_bend1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger1_bend")
    finger_bend2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger2_bend")
    finger_bend3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger3_bend")

    # 获取夹爪关节在qpos数组中的位置
    finger_joint1_qpos_addr = model.jnt_qposadr[finger_joint1_id]
    finger_joint2_qpos_addr = model.jnt_qposadr[finger_joint2_id]
    finger_joint3_qpos_addr = model.jnt_qposadr[finger_joint3_id]

    # 弯曲关节位置
    finger_bend1_qpos_addr = model.jnt_qposadr[finger_bend1_id]
    finger_bend2_qpos_addr = model.jnt_qposadr[finger_bend2_id]
    finger_bend3_qpos_addr = model.jnt_qposadr[finger_bend3_id]

    # 创建可视化环境
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("开始三指夹爪完整控制演示...")
        print("演示包括三种不同的控制模式")
        print("按Ctrl+C退出程序")

        try:
            start_time = time.time()
            while viewer.is_running():
                current_time = time.time() - start_time

                # 每隔6秒切换一种模式
                mode = int(current_time / 6) % 3

                if mode == 0:
                    # 模式1: 同步开合
                    if int(current_time) % 6 == 0:  # 每轮开始时打印
                        print("模式1: 三指同步开合...")

                    # 正弦波控制开合 (周期3秒)
                    open_close_value = 0.06 * (1 + np.sin(2 * np.pi * current_time / 3))

                    # 设置三个手指的基础关节
                    data.qpos[finger_joint1_qpos_addr] = open_close_value
                    data.qpos[finger_joint2_qpos_addr] = open_close_value
                    data.qpos[finger_joint3_qpos_addr] = open_close_value

                    # 微微弯曲指尖
                    bend_value = 0.2 * np.sin(2 * np.pi * current_time / 3)
                    data.qpos[finger_bend1_qpos_addr] = bend_value
                    data.qpos[finger_bend2_qpos_addr] = bend_value
                    data.qpos[finger_bend3_qpos_addr] = bend_value

                elif mode == 1:
                    # 模式2: 顺序动作
                    if int(current_time) % 6 == 0:  # 每轮开始时打印
                        print("模式2: 手指顺序动作...")

                    # 每个手指独立控制，相位差120度
                    phase1 = 0.06 * (1 + np.sin(2 * np.pi * current_time / 3))
                    phase2 = 0.06 * (1 + np.sin(2 * np.pi * current_time / 3 + 2 * np.pi / 3))
                    phase3 = 0.06 * (1 + np.sin(2 * np.pi * current_time / 3 + 4 * np.pi / 3))

                    data.qpos[finger_joint1_qpos_addr] = phase1
                    data.qpos[finger_joint2_qpos_addr] = phase2
                    data.qpos[finger_joint3_qpos_addr] = phase3

                elif mode == 2:
                    # 模式3: 波浪式动作
                    if int(current_time) % 6 == 0:  # 每轮开始时打印
                        print("模式3: 波浪式动作...")

                    # 创建波浪效果
                    wave1 = 0.06 * (1 + np.sin(4 * np.pi * current_time / 3))
                    wave2 = 0.06 * (1 + np.sin(4 * np.pi * current_time / 3 + np.pi / 2))
                    wave3 = 0.06 * (1 + np.sin(4 * np.pi * current_time / 3 + np.pi))

                    data.qpos[finger_joint1_qpos_addr] = wave1
                    data.qpos[finger_joint2_qpos_addr] = wave2
                    data.qpos[finger_joint3_qpos_addr] = wave3

                    # 同时控制弯曲关节产生波浪效果
                    bend1 = 0.3 * np.sin(4 * np.pi * current_time / 3)
                    bend2 = 0.3 * np.sin(4 * np.pi * current_time / 3 + np.pi / 2)
                    bend3 = 0.3 * np.sin(4 * np.pi * current_time / 3 + np.pi)

                    data.qpos[finger_bend1_qpos_addr] = bend1
                    data.qpos[finger_bend2_qpos_addr] = bend2
                    data.qpos[finger_bend3_qpos_addr] = bend3

                # 运行仿真步骤
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n程序已退出")


if __name__ == "__main__":
    main()