"""
人形机器人 MuJoCo 仿真程序
功能：站立保持/行走/手势控制/平衡测试
特性：简化模型、清晰控制、视觉反馈
"""
import os
os.environ['MUJOCO_LOG_DIR'] = os.devnull

import mujoco
import mujoco.viewer
import numpy as np
import time
import logging
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Optional


# ======================== 配置 ========================
@dataclass
class HumanoidConfig:
    """人形机器人配置"""
    model_path: str = "model/humanoid.xml"
    fps: int = 60
    sim_duration: float = 60.0
    enable_vision: bool = True
    gravity_z: float = -9.81  # 重力加速度 (可以调小用于测试)


def setup_logger(name: str = "HumanoidSim") -> logging.Logger:
    """日志配置"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    return logger


logger = setup_logger()


# ======================== 控制器 ========================
class HumanoidController:
    """人形机器人控制器"""

    # 站立姿势定义 (双腿完全对称伸直)
    # 格式: [躯干, 颈部, 左肩, 左肘, 左腕, 右肩, 右肘, 右腕, 左髋, 左膝, 左踝, 右髋, 右膝, 右踝]
    STANDING_POSE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 关节列表 (按 XML 顺序)
    JOINT_NAMES = [
        "torso_joint", "neck_joint",
        "left_shoulder_y", "left_elbow_y", "left_wrist_y",
        "right_shoulder_y", "right_elbow_y", "right_wrist_y",
        "left_hip_y", "left_knee_x", "left_ankle_y",
        "right_hip_y", "right_knee_x", "right_ankle_y",
    ]

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)

        # 获取关节和执行器 ID
        self.joint_ids = {}
        self.actuator_ids = {}
        for i, name in enumerate(self.JOINT_NAMES):
            self.joint_ids[name] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.actuator_ids[name] = i  # actuator 按顺序排列

        # 获取pelvis freejoint ID (用于平衡控制)
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.pelvis_qposadr = model.body_jntadr[self.pelvis_id]  # freejoint在qpos中的起始地址

        # 站立姿势
        self.standing_pose = np.zeros(len(self.JOINT_NAMES))  # 所有关节目标为0
        self.target_qpos = self.standing_pose.copy()

        # 初始位置和姿态 (用于保持平衡)
        self.initial_pelvis_pos = np.array([0, 0, 1.0])  # 初始位置
        self.initial_pelvis_quat = np.array([1, 0, 0, 0])  # 初始姿态 (w,x,y,z)

        # PD 控制参数 (关节控制)
        # 躯干 颈部 左肩 左肘 左腕 右肩 右肘 右腕 左髋 左膝 左踝 右髋 右膝 右踝
        kp = np.array([80, 50, 80, 60, 40, 80, 60, 40, 100, 120, 60, 100, 120, 60])
        kd = np.array([25, 15, 25, 20, 12, 25, 20, 12, 35, 40, 20, 35, 40, 20])
        self.pid_kp = kp
        self.pid_kd = kd

        # 平衡控制参数
        self.balance_kp_pos = 300   # 位置控制刚度
        self.balance_kd_pos = 80   # 位置控制阻尼
        self.balance_kp_rot = 150   # 姿态控制刚度
        self.balance_kd_rot = 40   # 姿态控制阻尼

    def set_target(self, joint_name: str, angle: float):
        """设置单个关节目标角度 (度)"""
        if joint_name in self.joint_ids:
            idx = self.JOINT_NAMES.index(joint_name)
            self.target_qpos[idx] = np.radians(angle)

    def set_targets_all(self, angles_deg: List[float]):
        """批量设置所有关节目标角度"""
        if len(angles_deg) == len(self.JOINT_NAMES):
            self.target_qpos = np.radians(angles_deg)

    def compute(self, dt: float = 0.002) -> np.ndarray:
        """计算 PD 控制力矩"""
        current_qpos = np.array([self.data.joint(jid).qpos[0] for jid in self.joint_ids.values()])
        current_qvel = np.array([self.data.joint(jid).qvel[0] for jid in self.joint_ids.values()])

        # 误差计算 (处理角度环绕)
        error = self.target_qpos - current_qpos
        error = np.arctan2(np.sin(error), np.cos(error))  # 环绕处理

        # 误差限幅
        error = np.clip(error, -0.5, 0.5)

        # PD 控制
        torque = self.pid_kp * error - self.pid_kd * current_qvel

        # 添加平衡控制 (通过髋关节调整)
        balance_torque = self._compute_balance_control()
        torque += balance_torque

        return torque

    def _compute_balance_control(self) -> np.ndarray:
        """计算平衡控制力矩"""
        # 获取pelvis位置和姿态
        pelvis_pos = self.data.body(self.pelvis_id).xpos.copy()
        pelvis_quat = self.data.body(self.pelvis_id).xquat.copy()  # w,x,y,z格式

        # 计算位置误差
        pos_error = pelvis_pos - self.initial_pelvis_pos
        pos_vel = self.data.body(self.pelvis_id).cvel[3:6]  # 线速度

        # 计算姿态误差 (使用四元数)
        quat_error = self._quat_diff(self.initial_pelvis_quat, pelvis_quat)
        ang_vel = self.data.body(self.pelvis_id).cvel[0:3]  # 角速度

        # 平衡力矩
        balance_moment = self.balance_kp_rot * quat_error - self.balance_kd_rot * ang_vel

        # 将力矩分配到关节 (躯干和髋关节负责平衡)
        # torso: 0, neck: 1, 左髋: 8, 右髋: 11
        balance_torque = np.zeros(len(self.JOINT_NAMES))
        balance_torque[0] = balance_moment[2] * 0.5  # 躯干yaw
        balance_torque[8] = -quat_error[0] * 50 - pos_error[0] * 100  # 左髋
        balance_torque[11] = quat_error[0] * 50 + pos_error[0] * 100  # 右髋

        # X方向倾斜控制
        balance_torque[8] += -quat_error[1] * 50 - pos_error[1] * 80
        balance_torque[11] += -quat_error[1] * 50 - pos_error[1] * 80

        return balance_torque

    def _quat_diff(self, quat1: np.ndarray, quat2: np.ndarray) -> np.ndarray:
        """计算两个四元数之间的误差角 (绕x,y,z轴)"""
        # 四元数乘法: q1 * conj(q2)
        w1, x1, y1, z1 = quat1
        w2, x2, y2, z2 = quat2

        # conj(quat2)
        w2_c, x2_c, y2_c, z2_c = w2, -x2, -y2, -z2

        # q = q1 * conj(q2)
        w = w1*w2_c - x1*x2_c - y1*y2_c - z1*z2_c
        x = w1*x2_c + x1*w2_c + y1*z2_c - z1*y2_c
        y = w1*y2_c - x1*z2_c + y1*w2_c + z1*x2_c
        z = w1*z2_c + x1*y2_c - y1*x2_c + z1*w2_c

        # 转换为欧拉角 (简化的yaw-pitch-roll)
        # 从四元数提取角度
        error_angle_x = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        error_angle_y = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        error_angle_z = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

        return np.array([error_angle_x, error_angle_y, error_angle_z])

    def reset(self):
        """重置控制器"""


# ======================== 仿真器 ========================
class HumanoidSimulator:
    """人形机器人仿真器"""

    # 站立待机姿势 (关键：腿部分开，重心在中间)
    STANDING_POSE = HumanoidController.STANDING_POSE

    PRESETS = {
        "idle": {
            "name": "站立待机",
            "angles": STANDING_POSE,
        },
        "wave_left": {
            "name": "左手挥手",
            "angles": [0, 0, -70, -45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
        "wave_right": {
            "name": "右手挥手",
            "angles": [0, 0, 0, 0, 0, 70, 45, 0, 0, 0, 0, 0, 0, 0],
        },
        "arms_up": {
            "name": "双臂举起",
            "angles": [0, 0, -90, -80, 0, 90, 80, 0, 0, 0, 0, 0, 0, 0],
        },
        "squat": {
            "name": "蹲下",
            "angles": [0, 0, 0, 0, 0, 0, 0, 0, 0, -45, -15, 0, 45, 15],
        },
        "tilt_left": {
            "name": "身体左倾",
            "angles": [15, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, -10, 0, 0],
        },
        "tilt_right": {
            "name": "身体右倾",
            "angles": [-15, 0, 0, 0, 0, 0, 0, 0, -10, 0, 0, 10, 0, 0],
        },
    }

    def __init__(self, config: HumanoidConfig):
        self.config = config
        self.running = False
        self.cmd_queue = queue.Queue(maxsize=20)

        # 加载模型
        self._load_model()
        # 初始化控制器
        self.controller = HumanoidController(self.model)
        # 初始化站立姿势
        self._init_standing_pose()

        logger.info(f"人形机器人加载成功 | 关节数: {len(self.controller.JOINT_NAMES)}")

    def _load_model(self):
        """加载 MuJoCo 模型"""
        model_path = self.config.model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 应用重力设置
        self.model.opt.gravity[2] = self.config.gravity_z

    def _init_standing_pose(self):
        """初始化站立姿势"""
        standing_rad = np.radians(self.STANDING_POSE)
        for i, name in enumerate(self.controller.JOINT_NAMES):
            jid = self.controller.joint_ids[name]
            self.data.joint(jid).qpos[0] = standing_rad[i]
            self.data.joint(jid).qvel[0] = 0
        # 更新前向运动学
        mujoco.mj_forward(self.model, self.data)

        # 记录pelvis初始位置和姿态
        self.controller.initial_pelvis_pos = self.data.body(self.controller.pelvis_id).xpos.copy()
        self.controller.initial_pelvis_quat = self.data.body(self.controller.pelvis_id).xquat.copy()

        # 确保初始控制目标与初始姿势一致
        self.controller.target_qpos = self.controller.standing_pose.copy()

    def get_joint_positions(self) -> List[float]:
        """获取所有关节当前角度 (度)"""
        return [
            round(np.degrees(self.data.joint(jid).qpos[0]), 1)
            for jid in self.controller.joint_ids.values()
        ]

    def get_com_position(self) -> np.ndarray:
        """获取质心位置"""
        mujoco.mj_comPos(self.model, self.data)
        return self.data.subtree_com[0].copy()

    def get_balance_state(self) -> Dict:
        """获取平衡状态"""
        com = self.get_com_position()
        foot_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "l_foot_geom"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "r_foot_geom"),
        ]
        foot_z = min(self.data.geom(fid).xpos[2] for fid in foot_ids)

        return {
            "com_x": com[0],
            "com_y": com[1],
            "com_z": com[2],
            "stability": 1.0 if abs(com[0]) < 0.15 and abs(com[1]) < 0.1 else 0.0,
            "ground_clearance": com[2] - foot_z,
        }

    def apply_pose(self, preset_name: str, transition_time: float = 1.0):
        """应用预设姿势 (平滑过渡)"""
        if preset_name not in self.PRESETS:
            logger.warning(f"未知预设: {preset_name}")
            return

        preset = self.PRESETS[preset_name]
        target_angles = preset["angles"]

        # 保存当前角度用于平滑过渡 (从当前角度过渡到目标角度)
        self._start_angles = self.controller.target_qpos.copy()
        self._target_angles = np.radians(target_angles)
        self._transition_start = time.time()
        self._transition_duration = max(transition_time, 0.3)

        logger.info(f"切换到姿势: {preset['name']}")

    def _update_pose_transition(self):
        """更新姿势过渡"""
        if hasattr(self, '_target_angles') and hasattr(self, '_start_angles'):
            elapsed = time.time() - self._transition_start
            t = min(elapsed / self._transition_duration, 1.0)
            # 使用 smoothstep 曲线平滑过渡
            t_smooth = t * t * (3 - 2 * t)

            # 从当前角度平滑过渡到目标角度
            self.controller.target_qpos = self._start_angles * (1 - t_smooth) + self._target_angles * t_smooth

            if t >= 1.0:
                del self._target_angles
                del self._start_angles

    # 演示模式的姿势序列
    DEMO_PRESETS = ["idle", "wave_left", "idle", "arms_up", "idle", "squat", "idle", "wave_right", "idle"]

    def run(self, mode: str = "interactive"):
        """运行仿真"""
        self.running = True
        self.mode = mode
        start_time = time.time()
        dt = 1.0 / self.config.fps
        self.demo_idx = 0
        self.last_demo_switch = time.time()
        self.demo_interval = 3.0

        # 初始姿势 (使用站立姿势，已包含右腿方向修正)
        self.controller.target_qpos = self.controller.standing_pose.copy()
        # 重置调试计数器
        self.controller._debug_counter = 0

        # 启动命令监听线程
        if mode == "interactive":
            threading.Thread(target=self._command_listener, daemon=True).start()

        logger.info(f"启动仿真 | 帧率: {self.config.fps} | 模式: {mode}")

        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self._init_viewer(viewer)

                while self.running and (time.time() - start_time) < self.config.sim_duration:
                    loop_start = time.time()

                    # 模式更新
                    self._update_pose_transition()
                    if mode == "interactive":
                        self._process_commands()
                    elif mode == "demo":
                        self._update_demo_mode()

                    # 计算控制
                    torque = self.controller.compute(dt)

                    # 施加控制力
                    for i, name in enumerate(self.controller.JOINT_NAMES):
                        act_id = self.controller.actuator_ids[name]
                        self.data.ctrl[act_id] = np.clip(torque[i], -150, 150)

                    # 仿真一步
                    mujoco.mj_step(self.model, self.data)

                    # 渲染
                    viewer.sync()

                    # 打印状态 (每2秒一次)
                    if int(time.time() - start_time) % 2 == 0 and abs(time.time() - loop_start) < 0.1:
                        self._print_status()

                    # 帧率控制
                    elapsed = time.time() - loop_start
                    if elapsed < dt:
                        time.sleep(dt - elapsed)

        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            self.running = False

    def _init_viewer(self, viewer):
        """初始化视角"""
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.lookat = np.array([0, 0, 0.8])

    def _print_status(self):
        """打印状态"""
        joints = self.get_joint_positions()
        balance = self.get_balance_state()
        status = "✓ 平衡" if balance["stability"] > 0.5 else "✗ 失衡"
        print(f"\r[状态] {status} | COM: ({balance['com_x']:.2f}, {balance['com_y']:.2f}) | 高度: {balance['com_z']:.2f}m", end="", flush=True)

    def _command_listener(self):
        """命令监听"""
        print("\n" + "="*50)
        print("         人形机器人控制指令")
        print("="*50)
        print("姿势切换:")
        for key, preset in self.PRESETS.items():
            print(f"  {key:12} - {preset['name']}")
        print("\n控制指令:")
        print("  reset      - 重置控制器")
        print("  quit       - 退出程序")
        print("="*50 + "\n")

        while self.running:
            try:
                cmd = input("> ").strip().lower()
                if cmd == "quit":
                    self.running = False
                elif cmd:
                    self.cmd_queue.put(cmd, timeout=0.1)
            except EOFError:
                break
            except Exception:
                pass

    def _process_commands(self):
        """处理命令队列"""
        try:
            while True:
                cmd = self.cmd_queue.get_nowait()
                if cmd == "reset":
                    self.controller.reset()
                    logger.info("控制器已重置")
                elif cmd in self.PRESETS:
                    self.apply_pose(cmd)
                else:
                    logger.warning(f"未知指令: {cmd}")
        except queue.Empty:
            pass

    def _update_demo_mode(self):
        """更新演示模式"""
        now = time.time()
        if now - self.last_demo_switch >= self.demo_interval:
            preset = self.DEMO_PRESETS[self.demo_idx % len(self.DEMO_PRESETS)]
            self.apply_pose(preset, transition_time=1.0)
            self.demo_idx += 1
            self.last_demo_switch = now
            print(f"\r[演示] {self.PRESETS[preset]['name']}", end="", flush=True)


# ======================== 主函数 ========================
def main():
    """主函数"""
    print("\n" + "="*50)
    print("      人形机器人 MuJoCo 仿真器")
    print("="*50)
    print("重力设置:")
    print("  1. 正常重力 (9.81 m/s²)")
    print("  2. 低重力 (3.0 m/s²) - 更容易保持平衡")
    print("  3. 零重力 (0.0 m/s²) - 仅测试关节控制")
    print("\n运行模式:")
    print("  a. 交互模式 (命令行控制)")
    print("  b. 演示模式 (自动循环姿势)")
    print("="*50)

    gravity_choice = input("\n选择重力 [1/2/3]: ").strip()
    gravity_map = {"1": -9.81, "2": -3.0, "3": 0.0}
    gravity = gravity_map.get(gravity_choice, -9.81)

    mode_choice = input("选择模式 [a/b]: ").strip()
    mode = "demo" if mode_choice == "b" else "interactive"

    config = HumanoidConfig(
        model_path=os.path.join(os.path.dirname(__file__), "model/humanoid.xml"),
        fps=60,
        sim_duration=120.0,
        gravity_z=gravity,
    )

    print(f"\n启动仿真 | 重力: {abs(gravity):.2f} m/s² | 模式: {mode}\n")

    simulator = HumanoidSimulator(config)
    simulator.run(mode=mode)


if __name__ == "__main__":
    main()
