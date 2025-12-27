"""MuJoCo部署脚本 - 使用游戏手柄控制机器人"""
import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import pygame

# 导入功能模块
from utils.math_utils import get_gravity_orientation, pd_control
from utils.gamepad_utils import (
    init_gamepad, update_cmd_from_gamepad, joystick,
    get_current_mode
)
from utils.mode_utils import mode_names
from utils.disturbance_utils import apply_disturbance_force, set_disturbance_body_id
from utils.plot_utils import (
    recording, record_start_time, record_duration,
    tau_history, dqj_history, time_history, plot_data, reset_recording
)
import utils.plot_utils as plot_utils  # 用于修改全局变量
import utils.viewer_utils as viewer_utils  # 用于访问和修改viewer相关全局变量

# ==================== 主程序 ====================
if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, nargs='?', default="e3.yaml", 
                        help="config file name in the config folder (default: e3.yaml)")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)
        print("default_angles = ", default_angles)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        # 如果有相位，在基础观测维度上加2
        num_obs = config["num_obs"]
        print("num_actions = ", num_actions)
        print("num_obs = ", num_obs)
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    print("xml_path = ", xml_path)
    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    # 保存初始状态用于reset
    initial_qpos = d.qpos.copy()
    initial_qvel = d.qvel.copy()
    initial_cmd = cmd.copy()  # 保存初始速度命令

    # load policy
    policy = torch.jit.load(policy_path)
    init_gamepad(config)  # 初始化游戏手柄（传入配置）
    
    # 查找pelvis body的ID
    pelvis_body_id = None
    try:
        pelvis_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    except Exception:
        try:
            pelvis_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        except Exception:
            print("警告: 未找到pelvis_link或base_link，将使用body 0")
            pelvis_body_id = 0
    
    if pelvis_body_id >= 0:
        print(f"Pelvis body ID: {pelvis_body_id}")
    else:
        print("警告: 未找到pelvis body")
        pelvis_body_id = 0
    
    # 初始化扰动力body ID
    set_disturbance_body_id(m)
    
    phase = 0.0
    last_gamepad_update = time.time()
    gamepad_update_interval = 0.02  # 游戏手柄更新间隔（50Hz）
    
    current_mode = get_current_mode()
    print(f"\n当前模式: {mode_names[current_mode]} (模式 {current_mode})")
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # 处理reset请求
            if viewer_utils.reset_requested:
                # 重置机器人状态
                d.qpos[:] = initial_qpos
                d.qvel[:] = initial_qvel
                # 重置速度命令
                cmd[:] = initial_cmd
                # 重置动作和目标位置
                action[:] = 0.0
                target_dof_pos[:] = default_angles.copy()
                # 重置相位
                phase = 0.0
                # 重置计数器
                counter = 0
                # 重置数据记录
                reset_recording()
                # 清除外力
                d.xfrc_applied[:] = 0.0
                # 重置reset标志
                viewer_utils.reset_requested = False
                print("机器人状态已重置")
                # 执行一步仿真以应用重置
                mujoco.mj_step(m, d)
                viewer.sync()
                continue
            
            # 定期更新游戏手柄输入
            current_time = time.time()
            if current_time - last_gamepad_update >= gamepad_update_interval:
                cmd = update_cmd_from_gamepad(cmd)
                last_gamepad_update = current_time
            
            # 施加扰动力（如果处于抗扰动模式）
            force_base, torso_pos = apply_disturbance_force(m, d)
            
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # 执行一步仿真
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9:9 + num_actions] = qj
                obs[9 + num_actions:9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions:9 + 3 * num_actions] = action
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

                # 如果正在记录数据且在5秒之内，则记录 tau 和 dqj 数据
                if plot_utils.recording:
                    current_time = time.time() - plot_utils.record_start_time
                    if current_time <= plot_utils.record_duration:
                        time_history.append(current_time)
                        tau_history.append(tau.copy())
                        dqj_history.append((dqj / dof_vel_scale).copy())
                    else:
                        # 达到5秒后停止记录并绘图
                        plot_utils.recording = False
                        print("Recording finished, plotting data...")
                        plot_data()

            # 更新viewer设置
            viewer_utils.update_viewer_settings(viewer, pelvis_body_id, d)
            
            # 更新viewer，捕捉 GUI 改变和干扰
            viewer.sync()

            # 控制步长
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # 清理 pygame
    if joystick is not None:
        joystick.quit()
    pygame.quit()
