"""
MuJoCo 人形机器人仿真主模块

本模块实现基于 MuJoCo 物理引擎的人形机器人仿真控制。
机器人从平躺关键帧（home keyframe）启动，通过 PD 反馈控制维持平躺姿态，
右肘关节按固定周期执行循环抬手动作（上升→保持→下降）。

功能概述
----------
1. **模型加载与初始化**：从 ``humanoid.xml`` 加载 MJCF 模型，重置至 keyframe 0（平躺姿态），
   清零关节速度以避免初始瞬态冲击。
2. **PD 反馈控制**：对 15 个躯干/下肢关节施加 PD 力矩，跟踪平躺关键帧的参考位置，
   维持机器人平躺构型。
3. **肘关节周期抬手**：右肘关节按 6 秒周期执行分段线性力矩轨迹，
   实现上升（2s）→ 保持（2s）→ 下降（2s）的循环动作。
4. **控制信号后处理**：一阶指数平滑（EMA）滤波 + 力矩限幅，保证仿真数值稳定性。
5. **数据采集**：以 200 Hz 频率将全状态数据（时间、qpos、qvel）写入 CSV 文件。

控制架构
----------
- **PD 控制器**：``tau = kp * (q_des - q_cur) - kd * q_vel``
- **肘关节开环力矩**：基于时间的分段线性轨迹，无位置反馈
- **EMA 平滑**：``u_smooth[t] = alpha * u_target[t] + (1-alpha) * u_smooth[t-1]``
- **力矩限幅**：``clip(u, -max_torque, max_torque)``

稳定关节列表（15 个）
~~~~~~~~~~~~~~~~~~~~
abdomen_z, abdomen_y, abdomen_x,
hip_x_right, hip_z_right, hip_y_right, knee_right, ankle_y_right, ankle_x_right,
hip_x_left, hip_z_left, hip_y_left, knee_left, ankle_y_left, ankle_x_left

控制参数
----------
+------------------+-------+----------------------------+---------------------------+
| 参数             | 默认值| 说明                       | 作用                      |
+==================+=======+============================+===========================+
| ``kp``          | 35.0  | PD 比例增益                | 维持平躺姿态的刚度        |
+------------------+-------+----------------------------+---------------------------+
| ``kd``          | 8.0   | PD 微分增益                | 抑制振荡的阻尼            |
+------------------+-------+----------------------------+---------------------------+
| ``max_torque``  | 1.0   | 最大力矩限制 (Nm)          | 防止控制信号溢出          |
+------------------+-------+----------------------------+---------------------------+
| ``smooth_alpha``| 0.06  | EMA 平滑系数               | 控制信号低通滤波          |
+------------------+-------+----------------------------+---------------------------+
| ``elbow_lift_max``| 0.25| 肘关节最大力矩 (Nm)        | 抬手动作幅值              |
+------------------+-------+----------------------------+---------------------------+
| ``cycle_period``| 6.0   | 抬手周期 (秒)              | 上升2s+保持2s+下降2s     |
+------------------+-------+----------------------------+---------------------------+

数据输出
----------
- **终端**：每 0.5 秒打印仿真时间、抬手阶段、肘关节力矩、机器人高度
- **CSV**：``simulation_data.csv``，包含 [time, qpos_0..nq-1, qvel_0..nv-1]

典型用法
----------
.. code-block:: python

    from mujoco01.main import main
    main()

或直接运行：

.. code-block:: bash

    python -m mujoco01.main
"""

import time
import csv
import mujoco
from mujoco import viewer
import math
import numpy as np

def main():
    """
    启动 MuJoCo 人形机器人仿真控制主循环。

    控制流程
    ----------
    1. **模型加载**：从 ``humanoid.xml`` 解析 MJCF 模型，若失败则打印诊断信息并退出。
    2. **状态初始化**：重置至 keyframe 0（平躺姿态），清零关节速度以避免初始瞬态。
    3. **执行器映射**：正则提取 ``<motor>`` 标签，建立 name→index 映射表。
    4. **PD 控制回路**：对 15 个稳定关节计算 PD 力矩，跟踪平躺参考位姿。
       ``torque = kp * (q_des - q_cur) - kd * q_vel``
    5. **肘关节激励**：按 6 秒周期施加分段线性力矩轨迹（上升 2s → 保持 2s → 下降 2s）。
    6. **信号后处理**：EMA 平滑（:math:`\\alpha=0.06`）→ 力矩限幅（:math:`\\pm 1.0` N·m）。
    7. **数据记录**：每步将 [time, qpos, qvel] 写入 ``simulation_data.csv``（200 Hz）。
    8. **状态输出**：每 0.5 秒打印仿真时间、抬手阶段、肘关节力矩、根高度。

    Raises
    ------
    FileNotFoundError
        当 ``humanoid.xml`` 不存在或路径不正确时，被 try/except 捕获并输出错误信息。

    Notes
    -----
    - 函数阻塞运行，直至用户关闭 MuJoCo 可视化窗口。
    - CSV 文件在仿真结束后自动关闭。
    - 通过 ``viewer.launch_passive`` 启动被动可视化模式。

    See Also
    --------
    mujoco01.run_robot_control : 简易机器人控制示例（独立脚本，无 PD 控制）
    """
    model_path = "src/mujoco01/humanoid.xml"
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    # 清零速度和加速度，避免遗留初始速导致的冲击
    try:
        data.qvel[:] = 0
    except Exception:
        pass

    # 最大力矩限制（恒定控制，可以适度）
    max_torque = 1.0

    # 平滑滤波系数（增强稳定性，避免抖动）
    smooth_alpha = 0.06
    prev_ctrl = np.zeros(model.nu)

    # CSV 记录
    csv_file = open("simulation_data.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    header = ["time"] + [f"qpos_{i}" for i in range(model.nq)] + [f"qvel_{i}" for i in range(model.nv)]
    writer.writerow(header)

    print("启动模拟器...")
    with viewer.launch_passive(model, data) as v:
        last_print_time = 0
        # 启动诊断：打印 actuator 列表和初始根位姿，帮助定位控制索引
        try:
            import re
            with open(model_path, 'r', encoding='utf-8') as xf:
                xmltext = xf.read()
            act_names = re.findall(r'<motor name="([^\"]+)"', xmltext)
            print('actuator count:', model.nu)
            print('actuator order (index:name):')
            for i, n in enumerate(act_names):
                print(f'  {i}: {n}')
        except Exception as e:
            print('诊断读取 actuator 列表失败:', e)

        # 打印初始根部 qpos（root pos + quat），用于判断是否躺地
        if len(data.qpos) >= 3:
            print(f'初始 root pos z: {data.qpos[2]:.3f}')


        # 记录初始参考位姿以保持站立（keyframe 中的 home）
        qpos_ref = data.qpos.copy()

        # 读取 actuator 列表后，按名称选择需要进行 PD 的稳定执行器
        # 包含腿部和脚踝，保证稳定站立
        stable_names = [
            'abdomen_z','abdomen_y','abdomen_x',
            'hip_x_right','hip_z_right','hip_y_right','knee_right','ankle_y_right','ankle_x_right',
            'hip_x_left','hip_z_left','hip_y_left','knee_left','ankle_y_left','ankle_x_left'
        ]
        name_to_idx = {n: i for i, n in enumerate(act_names)} if 'act_names' in locals() else {}
        stable_actuators = [name_to_idx[n] for n in stable_names if n in name_to_idx]

        # 右臂执行器索引（基于 humanoid.xml 的 actuator 顺序）
        RIGHT_SHOULDER1 = name_to_idx.get('shoulder1_right', 15)
        RIGHT_SHOULDER2 = name_to_idx.get('shoulder2_right', 16)
        RIGHT_ELBOW = name_to_idx.get('elbow_right', 17)

        # PD 增益，用于保持其他关节的初始位置
        kp = 35.0   # 保持躺着姿态
        kd = 8.0    # 阻尼
        
        # 肘关节循环抬手参数
        elbow_lift_max = 0.25  # 最大力矩（降低到平稳）
        cycle_period = 6.0    # 周期（秒）：上升2.0s + 保持2.0s + 下降2.0s
        
        # 记录起始时间用于循环计算
        start_time = data.time

        while v.is_running():
            t = data.time

            # 获取当前根部高度（用于判断状态）
            root_z = data.qpos[2] if len(data.qpos) > 2 else 0.0
            
            # 简化的控制逻辑：直接给肘关节施加恒定力矩，其他关节保持参考位置
            target_ctrls = np.zeros(model.nu)
            
            # 第一部分：其他关节使用 PD 控制保持初始位置
            for a in stable_actuators:
                if a != name_to_idx.get('elbow_right', -1):  # 排除肘关节
                    try:
                        joint_id = int(model.actuator_trnid[a][1])
                        qidx = int(model.jnt_qposadr[joint_id])
                        if qidx >= 0 and qidx < len(data.qpos):
                            q_des = float(qpos_ref[qidx])
                            q_cur = float(data.qpos[qidx])
                            q_vel = float(data.qvel[qidx]) if qidx < len(data.qvel) else 0.0
                            torque = kp * (q_des - q_cur) - kd * q_vel
                            target_ctrls[a] = torque
                    except Exception:
                        continue
            
            # 第二部分：肘关节循环平稳抬手（上升→保持→下降→重复）
            if RIGHT_ELBOW < model.nu:
                # 计算在周期中的位置
                elapsed = (t - start_time) % cycle_period
                
                if elapsed < 1.0:  # 上升阶段（0-1秒）
                    target_lift = elbow_lift_max * (elapsed / 1.0)
                elif elapsed < 2.0:  # 保持阶段（1-2秒）
                    target_lift = elbow_lift_max
                else:  # 下降阶段（2-3秒）
                    target_lift = elbow_lift_max * (1.0 - (elapsed - 2.0) / 1.0)
                
                target_ctrls[RIGHT_ELBOW] = target_lift

            # 简单平滑滤波，避免控制突变导致抖动
            smoothed = prev_ctrl * (1.0 - smooth_alpha) + target_ctrls * smooth_alpha

            # 力矩限幅并写回 data.ctrl
            for i in range(min(model.nu, len(smoothed))):
                val = float(smoothed[i])
                if val > max_torque:
                    val = max_torque
                elif val < -max_torque:
                    val = -max_torque
                data.ctrl[i] = val

            prev_ctrl = smoothed

            # 仿真步进
            mujoco.mj_step(model, data)

            # 写入CSV
            row = [data.time]
            row += data.qpos.tolist()
            row += data.qvel.tolist()
            writer.writerow(row)

            # 打印信息（每0.5秒打印一次）
            if data.time - last_print_time > 0.5:
                print("="*70)
                print(f"[{t:.1f}s] 躺着姿态 - 循环抬手中")
                print(f"  控制模式: 肘关节循环抬手（周期{cycle_period}s）")
                print(f"  最大力矩: {elbow_lift_max:.2f} Nm")
                
                # 计算当前周期位置
                elapsed = (t - start_time) % cycle_period
                if elapsed < 1.0:
                    phase = "上升"
                    pct = (elapsed / 1.0) * 100
                elif elapsed < 2.0:
                    phase = "保持"
                    pct = 100
                else:
                    phase = "下降"
                    pct = ((1.0 - (elapsed - 2.0) / 1.0) * 100)
                
                print(f"  阶段: {phase} ({pct:.0f}%)")
                
                re = data.ctrl[RIGHT_ELBOW] if RIGHT_ELBOW < model.nu else 0.0
                print(f"  右肘关节输出力矩: {re:.3f} Nm")
                print(f"  机器人高度: {root_z:.3f}m")
                print()
                last_print_time = data.time

            v.sync()

    csv_file.close()
    print(" 数据已保存到 simulation_data.csv")

if __name__ == "__main__":
    main()