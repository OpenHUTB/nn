import mujoco
from mujoco import viewer
import time
import numpy as np
import random  # 新增：随机转向控制


def control_robot(model_path):
    """
    控制DeepMind Humanoid模型：向前行走 → 检测所有障碍 → 随机转向避障最近障碍 → 回归路径 → 停止
    适配多障碍物（固定+随机）场景
    """
    # 加载模型和数据
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # -------------------------- 关键修复：动态生成随机障碍位置 --------------------------
    # 找到wall2的body ID并修改其位置
    wall2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wall2")
    if wall2_id != -1:
        # 随机生成wall2位置：x∈[2.5,4.0], y∈[-1.5,1.5], z固定0.5
        random_x = 2.5 + random.random() * 1.5
        random_y = -1.5 + random.random() * 3.0
        # 修改模型中wall2的位置（mj_setGeomPos需要用data，或直接修改model.body_pos）
        model.body_pos[wall2_id][0] = random_x
        model.body_pos[wall2_id][1] = random_y
        model.body_pos[wall2_id][2] = 0.5
        print(f"已生成随机障碍wall2位置：x={random_x:.2f}, y={random_y:.2f}, z=0.5")

    # 打印电机数量（调试用）
    print(f"模型电机数量：{model.nu}，data.ctrl长度：{len(data.ctrl)}")

    # -------------------------- 兼容低版本MuJoCo的ID查询 --------------------------
    # 关键修改：查询所有以"wall"开头的障碍物ID（支持wall1/wall2/更多）
    wall_ids = []
    wall_names = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name and body_name.startswith("wall"):  # 匹配所有障碍体
            wall_ids.append(i)
            wall_names.append(body_name)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    # 打印检测到的障碍物
    if wall_ids:
        print(f"检测到障碍物：{wall_names}（ID：{wall_ids}）")
    else:
        print("警告：未检测到任何障碍物！")

    # -------------------------- 核心参数配置 --------------------------
    # 避障参数
    avoid_obstacle = False
    obstacle_distance_threshold = 1.5  # 触发避障距离
    obstacle_avoidance_time = 0
    obstacle_avoidance_duration = 4.0  # 转向避障持续时间
    turn_direction = 0  # 0:未选择, 1:右转, -1:左转（随机赋值）
    return_to_path = False  # 是否进入回归路径阶段
    return_time = 0
    return_duration = 3.0  # 回归路径持续时间
    stop_walk = False  # 是否停止行走
    closest_wall_id = -1  # 最近障碍物ID

    # 步态控制参数
    gait_period = 2.0
    swing_gain = 0.9  # 进一步降低增益，提升多障碍场景稳定性
    stance_gain = 0.8
    forward_speed = 0.4  # 降低前进速度，预留更多避障反应时间

    # 姿态稳定参数
    torso_pitch_target = 0.0
    torso_roll_target = 0.0
    torso_yaw_target = 0.0  # 偏航角目标（x轴正方向）
    balance_kp = 90.0  # 提高平衡增益，防摔倒
    balance_kd = 12.0
    yaw_kp = 70.0  # 提高偏航回正增益，精准回归路径

    # 启动可视化器
    mujoco.set_mjcb_control(None)
    with viewer.launch_passive(model, data) as viewer_instance:
        print("\nDeepMind Humanoid仿真启动！")
        print("控制逻辑：向前行走 → 检测最近障碍 → 随机转向避障 → 回归路径 → 停止")
        start_time = time.time()

        try:
            while True:
                if not viewer_instance.is_running():
                    break

                # -------------------------- 1. 多障碍检测与状态切换 --------------------------
                distance_to_closest_wall = float('inf')
                closest_wall_name = ""

                # 计算到所有障碍物的距离，找到最近的那个
                if wall_ids and torso_id != -1 and not stop_walk:
                    torso_pos = data.xpos[torso_id]
                    for idx, wall_id in enumerate(wall_ids):
                        wall_pos = data.xpos[wall_id]
                        distance = np.linalg.norm(torso_pos[:2] - wall_pos[:2])
                        if distance < distance_to_closest_wall:
                            distance_to_closest_wall = distance
                            closest_wall_id = wall_id
                            closest_wall_name = wall_names[idx]

                # 状态切换逻辑（基于最近障碍物）
                if closest_wall_id != -1 and torso_id != -1 and not stop_walk:
                    # 1.1 触发避障（随机选择转向方向）
                    if (distance_to_closest_wall < obstacle_distance_threshold and
                            not avoid_obstacle and not return_to_path):
                        avoid_obstacle = True
                        obstacle_avoidance_time = time.time()
                        turn_direction = random.choice([-1, 1])  # 随机左转/右转
                        dir_name = "左转" if turn_direction == -1 else "右转"
                        print(
                            f"\n检测到最近障碍【{closest_wall_name}】！距离：{distance_to_closest_wall:.2f}米，开始{dir_name}避障...")

                    # 1.2 避障完成，进入回归路径阶段
                    if avoid_obstacle and (time.time() - obstacle_avoidance_time) > obstacle_avoidance_duration:
                        avoid_obstacle = False
                        return_to_path = True
                        return_time = time.time()
                        print(f"{dir_name}避障完成，开始回归原前进方向...")

                    # 1.3 回归路径完成，停止行走
                    if return_to_path and (time.time() - return_time) > return_duration:
                        return_to_path = False
                        stop_walk = True
                        torso_pos = data.xpos[torso_id]
                        print(f"\n已回归原前进方向，停止行走！最终位置：x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f}")

                # -------------------------- 2. 步态周期计算 --------------------------
                elapsed_time = time.time() - start_time
                cycle = elapsed_time % gait_period
                phase = cycle / gait_period

                # -------------------------- 3. 关节控制核心逻辑 --------------------------
                data.ctrl[:] = 0.0  # 重置控制指令

                if stop_walk:
                    # 4. 停止状态：所有关节归零，保持站立
                    continue

                elif return_to_path:
                    # 3. 回归路径模式：反向转向，回到x轴正方向
                    return_phase = (time.time() - return_time) / return_duration
                    # 余弦减速回归，避免过冲（更平滑）
                    return_speed = 1.2 * np.cos(return_phase * np.pi)

                    # 3.1 躯干偏航角回正（关键：回到x轴正方向）
                    if torso_id != -1:
                        torso_quat = data.xquat[torso_id]
                        # 四元数转偏航角（yaw）- 修正计算方式，提升精度
                        yaw = np.arctan2(2 * (torso_quat[2] * torso_quat[3] - torso_quat[0] * torso_quat[1]),
                                         torso_quat[0] ** 2 - torso_quat[1] ** 2 - torso_quat[2] ** 2 + torso_quat[
                                             3] ** 2)
                        yaw_error = torso_yaw_target - yaw

                        # 3.2 转向回正控制（多关节协同）
                        abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
                        hip_z_right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_right")
                        hip_z_left_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_left")

                        if 0 <= abdomen_z_act_id < len(data.ctrl):
                            data.ctrl[abdomen_z_act_id] = yaw_kp * yaw_error * return_speed
                        if 0 <= hip_z_right_act_id < len(data.ctrl):
                            data.ctrl[hip_z_right_act_id] = -yaw_error * return_speed * 0.6
                        if 0 <= hip_z_left_act_id < len(data.ctrl):
                            data.ctrl[hip_z_left_act_id] = yaw_error * return_speed * 0.6

                    # 3.3 保持基本站立姿态（增强平衡）
                    for side in ["right", "left"]:
                        hip_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                        knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                        ankle_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_y_{side}")
                        if 0 <= hip_y_act_id < len(data.ctrl):
                            data.ctrl[hip_y_act_id] = -0.6
                        if 0 <= knee_act_id < len(data.ctrl):
                            data.ctrl[knee_act_id] = 0.9
                        if 0 <= ankle_y_act_id < len(data.ctrl):
                            data.ctrl[ankle_y_act_id] = 0.2

                elif avoid_obstacle:
                    # 2. 避障模式：转向绕开最近障碍
                    avoid_phase = (time.time() - obstacle_avoidance_time) / obstacle_avoidance_duration
                    # 正弦曲线控制转向速度（先加速后减速，更自然）
                    turn_speed = 1.3 * np.sin(avoid_phase * np.pi)

                    # 2.1 转向控制（增强转向力度，确保绕开障碍）
                    hip_z_right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_right")
                    hip_z_left_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_z_left")
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")

                    if 0 <= hip_z_right_act_id < len(data.ctrl):
                        data.ctrl[hip_z_right_act_id] = turn_direction * turn_speed * 0.9
                    if 0 <= hip_z_left_act_id < len(data.ctrl):
                        data.ctrl[hip_z_left_act_id] = -turn_direction * turn_speed * 0.9
                    if 0 <= abdomen_z_act_id < len(data.ctrl):
                        data.ctrl[abdomen_z_act_id] = turn_direction * turn_speed * 1.6

                    # 2.2 保持平衡（防摔倒）
                    for side in ["right", "left"]:
                        hip_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                        knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                        ankle_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_x_{side}")
                        if 0 <= hip_y_act_id < len(data.ctrl):
                            data.ctrl[hip_y_act_id] = -0.9
                        if 0 <= knee_act_id < len(data.ctrl):
                            data.ctrl[knee_act_id] = 1.1
                        if 0 <= ankle_x_act_id < len(data.ctrl):
                            data.ctrl[ankle_x_act_id] = 0.1  # 小幅脚踝调整，增强平衡

                else:
                    # 1. 正常向前行走模式（适配多障碍场景，更稳健）
                    for side, sign in [("right", 1), ("left", -1)]:
                        swing_phase = (phase + 0.5 * sign) % 1.0

                        # 查询电机ID
                        hip_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_x_{side}")
                        hip_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_z_{side}")
                        hip_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hip_y_{side}")
                        knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"knee_{side}")
                        ankle_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_y_{side}")
                        ankle_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_x_{side}")

                        # 腿部关节控制（向前行走，降低幅度增强稳定）
                        if 0 <= hip_x_act_id < len(data.ctrl):
                            data.ctrl[hip_x_act_id] = swing_gain * np.sin(2 * np.pi * swing_phase) * forward_speed
                        if 0 <= hip_z_act_id < len(data.ctrl):
                            data.ctrl[hip_z_act_id] = stance_gain * np.cos(2 * np.pi * swing_phase) * 0.25
                        if 0 <= hip_y_act_id < len(data.ctrl):
                            data.ctrl[hip_y_act_id] = -1.0 * np.sin(2 * np.pi * swing_phase) - 0.45
                        if 0 <= knee_act_id < len(data.ctrl):
                            data.ctrl[knee_act_id] = 1.3 * np.sin(2 * np.pi * swing_phase) + 0.75
                        if 0 <= ankle_y_act_id < len(data.ctrl):
                            data.ctrl[ankle_y_act_id] = 0.35 * np.cos(2 * np.pi * swing_phase)
                        if 0 <= ankle_x_act_id < len(data.ctrl):
                            data.ctrl[ankle_x_act_id] = 0.18 * np.sin(2 * np.pi * swing_phase)

                    # 躯干稳定控制（防摔倒+保持前进方向）
                    abdomen_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_z")
                    abdomen_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_y")
                    abdomen_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_x")

                    if torso_id != -1:
                        torso_quat = data.xquat[torso_id]
                        # 四元数转欧拉角（俯仰/侧翻）
                        pitch = 2 * (torso_quat[1] * torso_quat[3] - torso_quat[0] * torso_quat[2])
                        roll = 2 * (torso_quat[0] * torso_quat[1] + torso_quat[2] * torso_quat[3])

                        # 平衡补偿（提高增益，增强稳定）
                        if 0 <= abdomen_x_act_id < len(data.ctrl):
                            data.ctrl[abdomen_x_act_id] = balance_kp * (torso_roll_target - roll) - balance_kd * \
                                                          data.qvel[abdomen_x_act_id]
                        if 0 <= abdomen_y_act_id < len(data.ctrl):
                            data.ctrl[abdomen_y_act_id] = balance_kp * (torso_pitch_target - pitch) - balance_kd * \
                                                          data.qvel[abdomen_y_act_id]
                        if 0 <= abdomen_z_act_id < len(data.ctrl):
                            data.ctrl[abdomen_z_act_id] = 0.08 * np.sin(elapsed_time * 0.5)  # 减小扭腰幅度

                    # 手臂自然摆动（降低幅度，增强平衡）
                    for side, sign in [("right", 1), ("left", -1)]:
                        shoulder1_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"shoulder1_{side}")
                        shoulder2_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"shoulder2_{side}")
                        elbow_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"elbow_{side}")

                        shoulder1_cmd = 0.25 * np.sin(2 * np.pi * (phase + 0.5 * sign))
                        shoulder2_cmd = 0.18 * np.cos(2 * np.pi * (phase + 0.5 * sign))
                        elbow_cmd = -0.45 * np.sin(2 * np.pi * (phase + 0.5 * sign)) - 0.28

                        if 0 <= shoulder1_act_id < len(data.ctrl):
                            data.ctrl[shoulder1_act_id] = shoulder1_cmd
                        if 0 <= shoulder2_act_id < len(data.ctrl):
                            data.ctrl[shoulder2_act_id] = shoulder2_cmd
                        if 0 <= elbow_act_id < len(data.ctrl):
                            data.ctrl[elbow_act_id] = elbow_cmd

                # -------------------------- 4. 仿真推进 --------------------------
                mujoco.mj_step(model, data)
                viewer_instance.sync()

                # 实时状态输出（适配多障碍）
                if torso_id != -1 and int(elapsed_time * 2) % 2 == 0:
                    if stop_walk:
                        status = "已停止"
                        dist_info = "—"
                    elif return_to_path:
                        status = "回归路径中"
                        dist_info = f"{distance_to_closest_wall:.2f}m（最近障碍）"
                    elif avoid_obstacle:
                        status = f"避障中（{dir_name}）"
                        dist_info = f"{distance_to_closest_wall:.2f}m（{closest_wall_name}）"
                    else:
                        status = "正常行走"
                        dist_info = f"{distance_to_closest_wall:.2f}m（最近障碍）"

                    torso_pos = data.xpos[torso_id]
                    print(
                        f"\r时间：{elapsed_time:.1f}s | 位置：x={torso_pos[0]:.2f}, y={torso_pos[1]:.2f} | 距离：{dist_info} | 状态：{status}",
                        end="")

                time.sleep(model.opt.timestep * 2)

        except KeyboardInterrupt:
            print("\n\n仿真被用户中断")
        except Exception as e:
            print(f"\n\n运行错误：{e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    model_file = "Robot_move_straight.xml"  # 你的模型文件名
    control_robot(model_file)