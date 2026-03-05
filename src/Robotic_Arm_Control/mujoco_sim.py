import mujoco
import mujoco.viewer
import numpy as np
import time
import logging
import os
import threading
import sys
import queue
import json

# 配置日志（精简格式）
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 修复循环导入
try:
    from core.kinematics import RoboticArmKinematics
    from core.arm_functions import ArmFunctions
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from core.kinematics import RoboticArmKinematics
    from core.arm_functions import ArmFunctions


# ======================== 核心扩展类（内联，不新增文件） ========================
class PIDController:
    """轻量级PID控制器"""

    def __init__(self, kp=1500, ki=0.1, kd=10):
        self.kp = np.array([kp] * 6, dtype=np.float64)
        self.ki = np.array([ki] * 6, dtype=np.float64)
        self.kd = np.array([kd] * 6, dtype=np.float64)
        self.error_sum = np.zeros(6)
        self.last_error = np.zeros(6)
        self.dt = 1 / 30  # 匹配仿真帧率

    def compute(self, current_joints, target_joints):
        """计算PID输出（弧度）"""
        current = np.radians(current_joints)
        target = np.radians(target_joints)
        error = target - current

        # 积分项（带饱和）
        self.error_sum += error * self.dt
        self.error_sum = np.clip(self.error_sum, -1.0, 1.0)

        # 微分项
        error_diff = (error - self.last_error) / self.dt
        self.last_error = error

        # PID输出
        output = self.kp * error + self.ki * self.error_sum + self.kd * error_diff
        return output.tolist()


class TrajectoryIO:
    """轨迹保存/加载工具"""

    def __init__(self):
        self.traj_dir = "trajectories"
        os.makedirs(self.traj_dir, exist_ok=True)
        self.default_path = os.path.join(self.traj_dir, "arm_traj.json")

    def save(self, trajectory):
        """保存轨迹到JSON"""
        if not trajectory:
            logger.warning("无轨迹可保存")
            return False
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trajectory": trajectory
        }
        with open(self.default_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.info(f"轨迹已保存至：{self.default_path}")
        return True

    def load(self):
        """加载轨迹"""
        if not os.path.exists(self.default_path):
            logger.error(f"轨迹文件不存在：{self.default_path}")
            return []
        with open(self.default_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"加载轨迹：{len(data['trajectory'])}个点")
        return data['trajectory']


class TargetVisualizer:
    """目标点可视化（轻量级）"""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.target_pos = np.array([0.1, 0.0, 0.3])
        # 创建红色球形目标点
        self.geom = mujoco.MjGeom()
        mujoco.mj_initGeom(
            self.geom, mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0.02, 0.02],
            np.zeros(3), np.eye(3).flatten(), [1.0, 0.0, 0.0, 1.0]  # 红色
        )

    def update(self, pos):
        """更新目标点位置"""
        self.target_pos = np.array(pos) if len(pos) == 3 else self.target_pos

    def render(self, viewer):
        """在Viewer中绘制目标点"""
        self.geom.pos = self.target_pos
        mujoco.mjv_geom(viewer.vopt, self.model, self.data, [self.geom], 1)


# ======================== 主仿真类 ========================
class MuJoCoArmSim:
    """优化版机械臂仿真主类"""

    def __init__(self, model_path="model/six_axis_arm.xml"):
        # 1. 初始化基础配置
        self.model_path = os.path.abspath(model_path)
        self._load_mujoco_model()

        # 2. 初始化核心模块（复用现有类）
        self.kinematics = RoboticArmKinematics()
        self.arm_funcs = ArmFunctions(self.kinematics)
        self.pid = PIDController()
        self.traj_io = TrajectoryIO()
        self.target_vis = TargetVisualizer(self.model, self.data)

        # 3. 缓存常用ID和参数（减少重复查询）
        self.joint_names = [f"joint{i + 1}" for i in range(6)]
        self.actuator_names = [f"act{i + 1}" for i in range(6)]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in
                             self.actuator_names]

        # 4. 仿真参数（精简）
        self.fps = 30
        self.dt = 1 / self.fps
        self.joint_speed_limits = [10.0, 8.0, 8.0, 15.0, 15.0, 20.0]  # 度/秒

        # 5. 运行状态（统一管理）
        self.running = True
        self.key_queue = queue.Queue()
        self.current_traj_idx = 0
        self.trajectory = []

    def _load_mujoco_model(self):
        """精简版模型加载（带校验）"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在：{self.model_path}")
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            logger.info(f"MuJoCo版本：{mujoco.__version__} | 模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败：{e}")
            raise

    def _clip_joint_speed(self, current, target):
        """关节速度限制（核心优化）"""
        delta = np.array(target) - np.array(current)
        max_delta = np.array(self.joint_speed_limits) * self.dt
        return (np.array(current) + np.clip(delta, -max_delta, max_delta)).tolist()

    def get_joint_angles(self):
        """精简版关节角读取（带滤波）"""
        raw_angles = np.degrees([self.data.joint(jid).qpos[0] for jid in self.joint_ids])
        # 轻量级滤波（归一化+防抖）
        raw_angles = np.clip(raw_angles % 360, -180, 180)
        return [round(angle, 2) for angle in raw_angles]

    def set_joint_angles(self, joint_angles, use_pid=False):
        """精简版关节角设置（支持PID）"""
        joint_angles = self.kinematics._clip_joint_angles(joint_angles)
        if use_pid:
            # PID控制输出（直接设置到ctrl）
            current_angles = self.get_joint_angles()
            pid_output = self.pid.compute(current_angles, joint_angles)
            for i, act_id in enumerate(self.actuator_ids):
                self.data.ctrl[act_id] = pid_output[i]
        else:
            # 普通位置控制
            joint_radians = np.radians(joint_angles)
            for i, act_id in enumerate(self.actuator_ids):
                self.data.ctrl[act_id] = joint_radians[i]

    # ======================== 交互控制 ========================
    def _manual_control_listener(self):
        """优化版手动控制监听（指令更清晰）"""
        logger.info("\n===== 手动控制指令 =====")
        logger.info("1. 关节控制：j1+ / j1- / j2+ / j2- ... / j6+ / j6- (步长1度)")
        logger.info("2. 轨迹控制：save_traj (保存) | load_traj (加载)")
        logger.info("3. 目标控制：set_target x,y,z (例：set_target 0.2,0,0.4)")
        logger.info("4. 退出：stop / quit")
        logger.info("=======================\n")

        while self.running:
            try:
                cmd = input("输入指令：").strip()
                if cmd == 'quit':
                    self.running = False
                    break
                self.key_queue.put(cmd)
            except:
                continue

    def _process_manual_cmd(self, cmd, current_joints):
        """精简版指令处理"""
        new_joints = current_joints.copy()

        # 1. 基础关节控制
        joint_map = {f"j{i + 1}+": i for i in range(6)} | {f"j{i + 1}-": i for i in range(6)}
        if cmd in joint_map:
            idx = joint_map[cmd]
            step = 1.0 if '+' in cmd else -1.0
            new_joints[idx] = np.clip(new_joints[idx] + step,
                                      self.kinematics.joint_limits[self.joint_names[idx]][0],
                                      self.kinematics.joint_limits[self.joint_names[idx]][1])

        # 2. 轨迹控制
        elif cmd == 'save_traj':
            self.traj_io.save(self.trajectory)
        elif cmd == 'load_traj':
            self.trajectory = self.traj_io.load()
            self.current_traj_idx = 0

        # 3. 目标点控制
        elif cmd.startswith('set_target'):
            try:
                pos = [float(x) for x in cmd.split(' ')[1].split(',')]
                self.target_vis.update(pos)
                logger.info(f"目标点更新为：{pos}")
            except:
                logger.error("目标点格式错误！示例：set_target 0.2,0,0.4")

        return new_joints

    # ======================== 仿真主逻辑 ========================
    def run_mode(self, mode):
        """按模式运行仿真（精简核心循环）"""
        start_time = time.time()
        frame_count = 0
        self.trajectory = []

        # 初始化模式
        if mode == "trajectory":
            # 生成轨迹（速度限制）
            start_joints = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
            target_joints = self.kinematics.inverse_kinematics([0.15, 0.0, 0.35, 0, 0, 0])
            self.trajectory = self.arm_funcs.generate_linear_trajectory(start_joints, target_joints, 100)
            logger.info("轨迹规划模式启动（PID控制）")

        elif mode == "manual":
            # 启动手动控制线程（延迟启动，避免输入冲突）
            threading.Thread(target=self._manual_control_listener, daemon=True).start()
            logger.info("手动控制模式启动")

        elif mode == "follow":
            logger.info("目标跟随模式启动（三维动态目标）")

        # 仿真主循环（大幅精简）
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 初始化视角
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -15
            viewer.cam.lookat = [0.2, 0, 0.5]

            while viewer.is_running() and self.running and (time.time() - start_time) < 30:
                frame_start = time.time()
                current_joints = self.get_joint_angles()
                new_joints = current_joints

                # 按模式更新关节角
                if mode == "trajectory" and self.current_traj_idx < len(self.trajectory):
                    target_joints = self.trajectory[self.current_traj_idx]
                    new_joints = self._clip_joint_speed(current_joints, target_joints)
                    self.set_joint_angles(new_joints, use_pid=True)  # PID控制
                    self.current_traj_idx += 1

                elif mode == "manual":
                    # 处理手动指令（非阻塞）
                    try:
                        cmd = self.key_queue.get_nowait()
                        new_joints = self._process_manual_cmd(cmd, current_joints)
                        self.set_joint_angles(new_joints)
                    except queue.Empty:
                        pass

                elif mode == "follow":
                    # 动态目标点（三维正弦运动）
                    t = time.time()
                    target_pos = [
                        0.1 + 0.05 * np.sin(t),
                        0.0 + 0.03 * np.cos(t),
                        0.3 + 0.04 * np.sin(t / 2)
                    ]
                    self.target_vis.update(target_pos)
                    new_joints = self.arm_funcs.follow_moving_target(current_joints, target_pos)
                    new_joints = self._clip_joint_speed(current_joints, new_joints)
                    self.set_joint_angles(new_joints)

                # 核心仿真步骤（精简）
                mujoco.mj_step(self.model, self.data)
                self.target_vis.render(viewer)  # 绘制目标点
                viewer.sync()

                # 状态打印（每5帧一次，减少IO）
                if frame_count % 5 == 0:
                    current_pose = self.kinematics.forward_kinematics(current_joints)
                    print(f"\r仿真时长：{time.time() - start_time:.1f}s | 末端位姿：{current_pose}", end="")

                # 帧率控制
                time.sleep(max(0, self.dt - (time.time() - frame_start)))
                frame_count += 1

        logger.info(f"\n仿真结束 | 总帧数：{frame_count} | 平均帧率：{frame_count / (time.time() - start_time):.1f}FPS")


# ======================== 主函数 ========================
def main():
    """优化版主函数（交互更友好）"""
    try:
        sim = MuJoCoArmSim()

        # 清晰的模式选择（带输入校验）
        print("\n========================")
        print("      优化版机械臂仿真      ")
        print("========================")
        print("模式选择：")
        print("1 - 轨迹规划（PID精准控制）")
        print("2 - 手动控制（增强指令）")
        print("3 - 目标跟随（可视化目标点）")
        print("========================")

        while True:
            choice = input("输入模式编号（1/2/3）：").strip()
            if choice in ["1", "2", "3"]:
                break
            print("输入错误！请输入 1、2 或 3")

        # 运行对应模式
        mode_map = {"1": "trajectory", "2": "manual", "3": "follow"}
        sim.run_mode(mode_map[choice])

    except Exception as e:
        logger.error(f"程序异常：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()