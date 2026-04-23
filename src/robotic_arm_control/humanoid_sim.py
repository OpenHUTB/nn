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

        # 初始化目标角度
        self.target_qpos = np.zeros(len(self.JOINT_NAMES))
        self.pid_kp = np.array([150, 50, 80, 50, 20, 80, 50, 20, 100, 80, 50, 100, 80, 50])
        self.pid_kd = np.array([15, 5, 8, 5, 2, 8, 5, 2, 10, 8, 5, 10, 8, 5])

        # 积分项 (防饱和)
        self.integral = np.zeros(len(self.JOINT_NAMES))
        self.last_error = np.zeros(len(self.JOINT_NAMES))

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
        """计算 PID 控制力矩"""
        current_qpos = np.array([self.data.joint(jid).qpos[0] for jid in self.joint_ids.values()])
        current_qvel = np.array([self.data.joint(jid).qvel[0] for jid in self.joint_ids.values()])

        # 误差计算 (处理角度环绕)
        error = self.target_qpos - current_qpos
        error = np.arctan2(np.sin(error), np.cos(error))  # 环绕处理

        # PID 计算
        self.integral += error * dt
        self.integral = np.clip(self.integral, -0.5, 0.5)  # 积分限幅
        derivative = (error - self.last_error) / dt if dt > 1e-6 else 0
        self.last_error = error.copy()

        torque = self.pid_kp * error + self.pid_kd * derivative
        return torque

    def reset(self):
        """重置控制器"""
        self.integral.fill(0)
        self.last_error.fill(0)


# ======================== 仿真器 ========================
class HumanoidSimulator:
    """人形机器人仿真器"""

    # 预设姿势
    PRESETS = {
        "idle": {
            "name": "站立待机",
            "angles": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
        "wave_left": {
            "name": "左手挥手",
            "angles": [0, 0, -80, -60, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
        "wave_right": {
            "name": "右手挥手",
            "angles": [0, 0, 0, 0, 0, 80, 60, -20, 0, 0, 0, 0, 0, 0],
        },
        "arms_up": {
            "name": "双臂举起",
            "angles": [0, 0, -120, -90, 0, 120, 90, 0, 0, 0, 0, 0, 0, 0],
        },
        "squat": {
            "name": "蹲下",
            "angles": [0, 0, 0, 0, 0, 0, 0, 0, 0, 80, -30, 0, 80, -30],
        },
        "tilt_left": {
            "name": "身体左倾",
            "angles": [20, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, -10, 0, 0],
        },
        "tilt_right": {
            "name": "身体右倾",
            "angles": [-20, 0, 0, 0, 0, 0, 0, 0, -10, 0, 0, 10, 0, 0],
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

        logger.info(f"人形机器人加载成功 | 关节数: {len(self.controller.JOINT_NAMES)}")

    def _load_model(self):
        """加载 MuJoCo 模型"""
        model_path = self.config.model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

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

        # 保存目标用于平滑过渡
        self._target_angles = np.array(target_angles, dtype=np.float64)
        self._transition_start = time.time()
        self._transition_duration = transition_time

        logger.info(f"切换到姿势: {preset['name']}")

    def _update_pose_transition(self):
        """更新姿势过渡"""
        if hasattr(self, '_target_angles'):
            elapsed = time.time() - self._transition_start
            t = min(elapsed / self._transition_duration, 1.0)
            # 使用 smoothstep 曲线平滑过渡
            t_smooth = t * t * (3 - 2 * t)

            current = np.array(self.get_joint_positions())
            target = self._target_angles
            interpolated = current + (target - current) * t_smooth * 0.3
            self.controller.set_targets_all(interpolated.tolist())

            if t >= 1.0:
                del self._target_angles

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

        # 初始姿势
        self.apply_pose("idle", transition_time=0.5)

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
                        self.data.ctrl[act_id] = np.clip(torque[i], -50, 50)

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
    config = HumanoidConfig(
        model_path=os.path.join(os.path.dirname(__file__), "model/humanoid.xml"),
        fps=60,
        sim_duration=120.0,
    )

    print("\n" + "="*50)
    print("      人形机器人 MuJoCo 仿真器")
    print("="*50)
    print("1. 交互模式 (命令行控制)")
    print("2. 演示模式 (自动循环姿势)")
    print("="*50)

    choice = input("\n选择模式 [1/2]: ").strip()

    simulator = HumanoidSimulator(config)

    if choice == "2":
        simulator.run(mode="demo")
    else:
        simulator.run(mode="interactive")


if __name__ == "__main__":
    main()
