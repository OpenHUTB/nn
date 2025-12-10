import mujoco
from mujoco import viewer
import time
import numpy as np


def control_robot(model_path):
    """
    控制DeepMind Humanoid模型行走（修复索引越界+Windows适配）
    """
    # 加载模型和数据
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 打印电机数量（调试用，确认ctrl长度）
    print(f"模型电机数量：{model.nu}，data.ctrl长度：{len(data.ctrl)}")

    # -------------------------- 修复：兼容低版本MuJoCo的ID查询 --------------------------
    wall_id = -1
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name == "wall":
            wall_id = i
            break
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    # 避障参数配置
    avoid_obstacle = False
    obstacle_distance_threshold = 1.5
    obstacle_avoidance_time = 0
    obstacle_avoidance_duration = 4.0
    turn_direction = -1

    # 步态控制参数
    gait_period = 2.0
    swing_gain = 1.2
    stance_gain = 0.8

    # 姿态稳定参数
    torso_pitch_target = 0.0
    torso_roll_target = 0.0
    balance_kp = 80.0
    balance_kd = 10.0

    # 启动可视化器
    mujoco.set_mjcb_control(None)
    with viewer.launch_passive(model, data) as viewer_instance:
        print("DeepMind Humanoid仿真启动！（修复索引越界版）")
        print("控制逻辑：正常行走 → 检测障碍 → 左转避障 → 恢复行走")
        start_time = time.time()

        try:
            while True:
                if not viewer_instance.is_running():
                    break

                # -------------------------- 1. 障碍检测 --------------------------
                distance_to_wall = 0.0
                if wall_id != -1 and torso_id != -1:
                    torso_pos = data.xpos[torso_id]
                    wall_pos = data.xpos[wall_id]
                    distance_to_wall = np.linalg.norm(torso_pos[:2] - wall_pos[:2])

                    if distance_to_wall < obstacle_distance_threshold and not avoid_obstacle:
                        avoid_obstacle = True
                        obstacle_avoidance_time = time.time()
                        print(f"\n检测到墙壁障碍！距离：{distance_to_wall:.2f}米，开始左转避障...")

                    if avoid_obstacle and (time.time() - obstacle_avoidance_time) > obstacle_avoidance_duration:
                        avoid_obstacle = False
                        print(f"避障完成（耗时{obstacle_avoidance_duration}秒），恢复正常行走")

                # -------------------------- 2. 步态周期计算 --------------------------
                elapsed_time = time.time() - start_time
                cycle = elapsed_time % gait_period
                phase = cycle / gait_period

                # -------------------------- 3. 关节控制核心逻辑 --------------------------
                data.ctrl[:] = 0.0  # 重置控制指令

                if not avoid_obstacle:
                    # ========== 正常行走模式 ==========
                    for side, sign in [("right", 1), ("left", -1)]:
                        swing_phase = (phase + 0.5 * sign) % 1.0

                        # 1. 查询关节ID（仅查询<actuator>下的电机ID，而非关节ID）
                        # 注意：这里改为查询ACTUATOR（电机）ID，而非JOINT ID！
                        hip_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_x_{side}")
                        hip_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_z_{side}")
                        hip_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                        knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                        ankle_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_y_{side}")
                        ankle_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_x_{side}")

                        # 2. 核心修复：仅当ID有效（< len(data.ctrl)）时才赋值
                        if 0 <= hip_x_act_id < len(data.ctrl):
                            data.ctrl[hip_x_act_id] = swing_gain * np.sin(2 * np.pi * swing_phase)
                        if 0 <= hip_z_act_id < len(data.ctrl):
                            data.ctrl[hip_z_act_id] = stance_gain * np.cos(2 * np.pi * swing_phase) * 0.3
                        if 0 <= hip_y_act_id < len(data.ctrl):
                            data.ctrl[hip_y_act_id] = -1.2 * np.sin(2 * np.pi * swing_phase) - 0.5
                        if 0 <= knee_act_id < len(data.ctrl):
                            data.ctrl[knee_act_id] = 1.5 * np.sin(2 * np.pi * swing_phase) + 0.8
                        if 0 <= ankle_y_act_id < len(data.ctrl):
                            data.ctrl[ankle_y_act_id] = 0.4 * np.cos(2 * np.pi * swing_phase)
                        if 0 <= ankle_x_act_id < len(data.ctrl):
                            data.ctrl[ankle_x_act_id] = 0.2 * np.sin(2 * np.pi * swing_phase)

                    # 躯干稳定控制
                    abdomen_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
                    abdomen_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_y")
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_x")

                    if torso_id != -1 and 0 <= abdomen_x_act_id < len(data.ctrl):
                        torso_quat = data.xquat[torso_id]
                        pitch = 2 * (torso_quat[1] * torso_quat[3] - torso_quat[0] * torso_quat[2])
                        roll = 2 * (torso_quat[0] * torso_quat[1] + torso_quat[2] * torso_quat[3])

                        if 0 <= abdomen_x_act_id < len(data.ctrl):
                            data.ctrl[abdomen_x_act_id] = balance_kp * (torso_roll_target - roll) - balance_kd * \
                                                          data.qvel[abdomen_x_act_id]
                        if 0 <= abdomen_y_act_id < len(data.ctrl):
                            data.ctrl[abdomen_y_act_id] = balance_kp * (torso_pitch_target - pitch) - balance_kd * \
                                                          data.qvel[abdomen_y_act_id]
                        if 0 <= abdomen_z_act_id < len(data.ctrl):
                            data.ctrl[abdomen_z_act_id] = 0.1 * np.sin(elapsed_time * 0.5)

                    # 手臂控制（仅控制存在的电机）
                    for side, sign in [("right", 1), ("left", -1)]:
                        shoulder1_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"shoulder1_{side}")
                        shoulder2_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"shoulder2_{side}")
                        elbow_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"elbow_{side}")

                        shoulder1_cmd = 0.3 * np.sin(2 * np.pi * (phase + 0.5 * sign))
                        shoulder2_cmd = 0.2 * np.cos(2 * np.pi * (phase + 0.5 * sign))
                        elbow_cmd = -0.5 * np.sin(2 * np.pi * (phase + 0.5 * sign)) - 0.3

                        if 0 <= shoulder1_act_id < len(data.ctrl):
                            data.ctrl[shoulder1_act_id] = shoulder1_cmd
                        if 0 <= shoulder2_act_id < len(data.ctrl):
                            data.ctrl[shoulder2_act_id] = shoulder2_cmd
                        if 0 <= elbow_act_id < len(data.ctrl):
                            data.ctrl[elbow_act_id] = elbow_cmd

                else:
                    # ========== 避障模式 ==========
                    avoid_phase = (time.time() - obstacle_avoidance_time) / obstacle_avoidance_duration
                    turn_speed = 1.2 * np.sin(avoid_phase * np.pi)

                    # 转向控制（仅控制存在的电机）
                    hip_z_right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_right")
                    hip_z_left_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_left")
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")

                    if 0 <= hip_z_right_act_id < len(data.ctrl):
                        data.ctrl[hip_z_right_act_id] = turn_direction * turn_speed * 0.8
                    if 0 <= hip_z_left_act_id < len(data.ctrl):
                        data.ctrl[hip_z_left_act_id] = -turn_direction * turn_speed * 0.8
                    if 0 <= abdomen_z_act_id < len(data.ctrl):
                        data.ctrl[abdomen_z_act_id] = turn_direction * turn_speed * 1.5

                    # 平衡控制
                    hip_y_right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_y_right")
                    hip_y_left_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_y_left")
                    knee_right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_right")
                    knee_left_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_left")

                    if 0 <= hip_y_right_act_id < len(data.ctrl):
                        data.ctrl[hip_y_right_act_id] = -0.8
                    if 0 <= hip_y_left_act_id < len(data.ctrl):
                        data.ctrl[hip_y_left_act_id] = -0.8
                    if 0 <= knee_right_act_id < len(data.ctrl):
                        data.ctrl[knee_right_act_id] = 1.0
                    if 0 <= knee_left_act_id < len(data.ctrl):
                        data.ctrl[knee_left_act_id] = 1.0

                # -------------------------- 4. 仿真推进 --------------------------
                mujoco.mj_step(model, data)
                viewer_instance.sync()

                # 实时输出（避免未定义变量）
                if wall_id != -1 and torso_id != -1 and int(elapsed_time * 2) % 2 == 0:
                    status = "避障中" if avoid_obstacle else "正常行走"
                    print(f"\r时间：{elapsed_time:.1f}s | 距离墙壁：{distance_to_wall:.2f}m | 状态：{status}", end="")

                time.sleep(model.opt.timestep * 2)

        except KeyboardInterrupt:
            print("\n\n仿真被用户中断")
        except IndexError as e:
            print(f"\n\n索引越界错误：{e}")
            print(f"当前data.ctrl长度：{len(data.ctrl)}，出错时的ID：{elbow_id if 'elbow_id' in locals() else '未知'}")


if __name__ == "__main__":
    model_file = "Robot_move_straight.xml"
    control_robot(model_file)