import mujoco
import numpy as np
from mujoco import viewer
import time
import os


class HumanoidStabilizer:
    """专注于机器人稳定站立的控制器"""

    def __init__(self, model_path):
        # 类型检查
        if not isinstance(model_path, str):
            raise TypeError(f"模型路径必须是字符串，当前是 {type(model_path)} 类型")

        # 模型加载
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}\n请检查：1.路径是否为字符串 2.文件是否存在 3.文件是否完整")

        # 仿真参数
        self.sim_duration = 60.0  # 延长仿真时间，测试长时间站立
        self.dt = self.model.opt.timestep
        self.init_wait_time = 2.0  # 延长初始等待，让模型稳定

        # 站立控制参数（针对重心不佳优化）
        self.kp_root = 120.0  # 躯干姿态比例增益（增强直立控制）
        self.kd_root = 15.0  # 躯干阻尼增益（抑制晃动）
        self.kp_legs = 200.0  # 腿部关节比例增益（增强支撑）
        self.kd_legs = 20.0  # 腿部关节阻尼增益
        self.hip_bias = 0.1  # 髋关节偏置，调整重心前后位置
        self.knee_bias = -0.2  # 膝关节偏置，微屈以降低重心

        # 状态变量
        self.prev_root_rot = np.zeros(3)  # 上一帧躯干欧拉角
        self.leg_joint_targets = None  # 腿部关节目标角度
        self.torso_target_euler = np.zeros(3)  # 躯干目标欧拉角（直立）

        # 初始化稳定姿态
        self._init_stable_pose()

    def _init_stable_pose(self):
        """初始化稳定的站立姿态，调整重心位置"""
        # 重置模型到初始状态
        mujoco.mj_resetData(self.model, self.data)

        # 手动调整初始位置和姿态，降低重心并调整到双脚中心
        self.data.qpos[2] = 1.1  # 降低躯干高度（原可能过高，调整重心）
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 躯干直立（四元数，w,x,y,z顺序）
        self.data.qvel[:] = 0.0  # 初始速度归零，防止模型飞起

        # 初始化腿部关节目标角度（微屈，形成稳定支撑）
        num_actuators = self.model.nu
        self.leg_joint_targets = np.zeros(num_actuators)

        # 假设腿部关节索引：0-5为右腿，6-11为左腿（根据常见humanoid.xml结构）
        leg_joint_count = min(12, num_actuators)
        for i in range(leg_joint_count):
            if i % 6 == 0:  # 髋x
                self.leg_joint_targets[i] = self.hip_bias
            elif i % 6 == 3:  # 膝y
                self.leg_joint_targets[i] = self.knee_bias

        # 前向计算更新状态
        mujoco.mj_forward(self.model, self.data)

    def _get_root_euler(self):
        """提取躯干欧拉角（roll, pitch, yaw），用于姿态控制"""
        # 修正mju_quat2Mat的参数格式（需要一维数组，且内存连续）
        rot_mat = np.zeros(9, dtype=np.float64)  # 一维数组存储9个元素
        quat = self.data.qpos[3:7].astype(np.float64).copy()  # 确保float64且连续
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)  # 转为3x3矩阵

        # 计算欧拉角（XYZ顺序）
        euler = np.zeros(3, dtype=np.float64)
        mujoco.mju_mat2Euler(euler, rot_mat.flatten(), 1)  # 第二个参数为一维数组
        return euler

    def _calculate_stabilizing_torques(self):
        """计算维持站立的稳定力矩，重点补偿重心偏移"""
        num_actuators = self.model.nu
        torques = np.zeros(num_actuators, dtype=np.float64)

        # 1. 躯干姿态控制（保持直立）
        try:
            root_euler = self._get_root_euler()
        except:
            root_euler = np.zeros(3)
        root_euler_error = self.torso_target_euler - root_euler
        root_angular_vel = self.data.qvel[3:6].copy()  # 躯干角速度
        root_angular_vel = np.clip(root_angular_vel, -5.0, 5.0)  # 限制角速度

        # 躯干稳定力矩（通过根关节虚拟力矩，实际作用于腿部支撑调整）
        torso_torque = self.kp_root * root_euler_error - self.kd_root * root_angular_vel
        torso_torque = np.clip(torso_torque, -20.0, 20.0)  # 限制躯干力矩

        # 2. 腿部关节控制（刚性支撑+重心补偿）
        leg_joint_count = min(12, num_actuators)
        leg_joint_indices = list(range(leg_joint_count))

        # 获取当前腿部关节位置（确保数组长度匹配）
        current_joints = self.data.qpos[7:] if len(self.data.qpos) > 7 else np.zeros(num_actuators)
        current_joints = current_joints[:num_actuators].astype(np.float64)

        # 获取当前腿部关节速度（确保数组长度匹配）
        current_vel = self.data.qvel[6:] if len(self.data.qvel) > 6 else np.zeros(num_actuators)
        current_vel = current_vel[:num_actuators].astype(np.float64)
        current_vel = np.clip(current_vel, -10.0, 10.0)  # 限制关节速度

        # 腿部关节误差（加入躯干姿态补偿，调整支撑脚位置）
        leg_joint_error = self.leg_joint_targets.copy()
        if len(leg_joint_indices) >= 2:
            leg_joint_error[leg_joint_indices[0]] += torso_torque[0] * 0.05  # 左髋补偿roll
            leg_joint_error[leg_joint_indices[6]] -= torso_torque[0] * 0.05  # 右髋补偿roll
        if len(leg_joint_indices) >= 1:
            leg_joint_error[leg_joint_indices[1]] += torso_torque[1] * 0.05  # 髋补偿pitch

        # 腿部力矩计算（高刚性+阻尼）
        torque_error = leg_joint_error - current_joints
        torque_error = np.clip(torque_error, -0.5, 0.5)  # 限制误差范围，防止力矩突变
        torques = self.kp_legs * torque_error - self.kd_legs * current_vel

        # 3. 力矩限幅（防止过载）
        torque_limit = 50.0  # 增大力矩上限，提供足够支撑力
        torques = np.clip(torques, -torque_limit, torque_limit)

        # 4. 重力补偿（针对重心不佳的模型，添加固定偏置力矩）
        if leg_joint_count > 0:
            torques[leg_joint_indices] += np.sign(leg_joint_error[leg_joint_indices]) * 2.0  # 小幅重力补偿

        return torques

    def simulate_stable_standing(self):
        """启动稳定站立仿真"""
        # 关闭重力扰动（可选，若模型仍飞起）
        self.model.opt.gravity[2] = -9.81  # 标准重力，防止重力设置错误
        self.model.opt.timestep = 0.005  # 合理的时间步长，提高仿真稳定性

        with viewer.launch_passive(self.model, self.data) as v:
            print("可视化窗口已启动，机器人将尝试稳定站立...")
            print("操作：鼠标拖动旋转视角，滚轮缩放，W/A/S/D平移，关闭窗口结束")

            # 初始等待（让模型稳定落地）
            start_time = time.time()
            while time.time() - start_time < self.init_wait_time:
                # 初始阶段施加零力矩，让模型自然落地
                self.data.ctrl[:] = 0.0
                mujoco.mj_step(self.model, self.data)
                # 强制速度归零，防止初始抖动
                self.data.qvel[:] = 0.0
                v.sync()
                time.sleep(0.01)

            # 主仿真循环（稳定站立控制）
            while self.data.time < self.sim_duration:
                # 计算稳定力矩
                torques = self._calculate_stabilizing_torques()
                self.data.ctrl[:] = torques

                # 步进仿真
                mujoco.mj_step(self.model, self.data)

                # 实时监测重心位置（调试用）
                if self.data.time % 5 < 0.1:  # 每5秒打印一次
                    com = self.data.subtree_com[0]  # 整体重心（第一个子树为根节点）
                    print(
                        f"仿真时间: {self.data.time:.1f}s | 重心位置(x,y,z): {com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}")

                # 可视化同步
                v.sync()
                time.sleep(0.001)

                # 紧急停止（若跌倒严重）
                com = self.data.subtree_com[0]
                if com[2] < 0.5:  # 重心过低判定为跌倒
                    print(f"机器人跌倒！仿真时间: {self.data.time:.1f}s")
                    break

        print("仿真完成！")


if __name__ == "__main__":
    # 模型路径处理
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(current_directory, "humanoid.xml")

    print(f"当前脚本所在目录：{current_directory}")
    print(f"模型文件完整路径：{model_file_path}")

    # 检查模型文件存在性
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(
            f"模型文件不存在！\n查找路径：{model_file_path}\n"
            f"请确认 'humanoid.xml' 文件放在以下目录中：{current_directory}"
        )

    # 启动稳定站立仿真
    try:
        stabilizer = HumanoidStabilizer(model_file_path)
        print("\n开始稳定站立仿真...")
        stabilizer.simulate_stable_standing()
    except Exception as e:
        print(f"\n仿真过程中发生错误：{e}")