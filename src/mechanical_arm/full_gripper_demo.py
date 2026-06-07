#!/usr/bin/env python3
"""
完整版三指夹爪控制演示（优化版）
展示三种控制模式：
1. 同步开合
2. 逐个手指顺序动作
3. 波浪式动作
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

def get_joint_qpos_addrs(model, joint_names: list[str]) -> list[int]:
    """根据关节名称列表获取 qpos 地址列表"""
    return [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in joint_names]

def compute_mode_actions(mode: int, t: float) -> tuple[list[float], list[float]]:
    """根据模式和当前时间计算三指夹爪基础关节和弯曲关节动作"""
    if mode == 0:  # 同步开合
        base = 0.06 * (1 + np.sin(2 * np.pi * t / 3))
        bend = 0.2 * np.sin(2 * np.pi * t / 3)
        base_values = [base] * 3
        bend_values = [bend] * 3
    elif mode == 1:  # 顺序动作
        phases = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        base_values = [0.06 * (1 + np.sin(2 * np.pi * t / 3 + p)) for p in phases]
        bend_values = [0.0, 0.0, 0.0]  # 手指单独控制不弯曲
    else:  # 波浪式动作
        phases = [0, np.pi / 2, np.pi]
        base_values = [0.06 * (1 + np.sin(4 * np.pi * t / 3 + p)) for p in phases]
        bend_values = [0.3 * np.sin(4 * np.pi * t / 3 + p) for p in phases]
    return base_values, bend_values

def main():
    # 加载模型
    model_path = "three_fingered_arm.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 关节名称
    finger_joints = ["finger_joint1", "finger_joint2", "finger_joint3"]
    finger_bends = ["finger1_bend", "finger2_bend", "finger3_bend"]

    # 获取 qpos 地址
    finger_qpos = get_joint_qpos_addrs(model, finger_joints)
    bend_qpos = get_joint_qpos_addrs(model, finger_bends)

    dt = 0.01  # 时间步长

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("开始三指夹爪完整控制演示...")
        print("演示包括三种不同的控制模式")
        print("按Ctrl+C退出程序")

        start_time = time.time()
        try:
            while viewer.is_running():
                t = time.time() - start_time
                mode = int(t / 6) % 3  # 每6秒切换模式

                base_vals, bend_vals = compute_mode_actions(mode, t)

                # 更新 qpos
                for i in range(3):
                    data.qpos[finger_qpos[i]] = base_vals[i]
                    data.qpos[bend_qpos[i]] = bend_vals[i]

                # 模式切换打印
                if int(t) % 6 == 0:
                    mode_names = ["同步开合", "手指顺序动作", "波浪式动作"]
                    print(f"模式{mode + 1}: {mode_names[mode]}...")

                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(dt)

        except KeyboardInterrupt:
            print("\n程序已退出")


if __name__ == "__main__":
    main()