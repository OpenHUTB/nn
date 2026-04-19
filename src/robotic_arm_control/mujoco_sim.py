"""
机械臂MuJoCo仿真主程序（深度优化版）
功能：轨迹规划/PID控制/手动控制/目标跟随/碰撞检测/轨迹保存/目标可视化
特性：鲁棒性强、性能优化、易扩展、易维护
"""
import os

# 获取脚本目录作为基准路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 禁用MuJoCo日志（设置空路径）
os.environ['MUJOCO_LOG_DIR'] = os.devnull

import mujoco
import mujoco.viewer
import numpy as np
import time
import logging
import threading
import sys
import queue
import json
from dataclasses import dataclass
from typing import List, Dict


# ======================== 全局配置（解耦硬编码） ========================
@dataclass
class SimConfig:
    """仿真配置类（集中管理所有参数）"""
    model_path: str = "model/six_axis_arm.xml"
    config_path: str = "config/arm_config.yaml"
    fps: int = 30
    sim_duration: float = 30.0
    joint_speed_limits: List[float] = None
    pid_params: Dict[str, float] = None
    target_geom_size: List[float] = None
    traj_save_path: str = "trajectories/arm_traj.json"

    def __post_init__(self):
        # 默认参数初始化
        self.joint_speed_limits = self.joint_speed_limits or [10.0, 8.0, 8.0, 15.0, 15.0, 20.0]
        self.pid_params = self.pid_params or {"kp": 1500.0, "ki": 0.1, "kd": 10.0}
        self.target_geom_size = self.target_geom_size or [0.02, 0.02, 0.02]


# ======================== 日志配置（分级输出） ========================
def setup_logger() -> logging.Logger:
    """配置日志：区分调试/信息/错误级别，输出到控制台+文件"""
    logger = logging.getLogger("ArmSim")
    logger.setLevel(logging.DEBUG)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 控制台处理器（INFO级别）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # 文件处理器（DEBUG级别，保存详细日志）
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/arm_sim.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


# 初始化日志
logger = setup_logger()


# ======================== 核心工具类（复用+解耦） ========================
class PIDController:
    """PID控制器（线程安全+参数可调）"""

    def __init__(self, kp: float, ki: float, kd: float, joint_num: int = 6, dt: float = 1 / 30):
        self.kp = np.array([kp] * joint_num, dtype=np.float64)
        self.ki = np.array([ki] * joint_num, dtype=np.float64)
        self.kd = np.array([kd] * joint_num, dtype=np.float64)
        self.dt = dt
        self._lock = threading.Lock()  # 线程安全锁

        # 状态变量（初始化）
        self.error_sum = np.zeros(joint_num, dtype=np.float64)
        self.last_error = np.zeros(joint_num, dtype=np.float64)

    def compute(self, current_joints: List[float], target_joints: List[float]) -> List[float]:
        """计算PID输出（弧度）"""
        if len(current_joints) != len(target_joints):
            logger.error(f"关节数不匹配：当前{len(current_joints)}个，目标{len(target_joints)}个")
            return [0.0] * len(current_joints)

        with self._lock:  # 线程安全
            # 转换为弧度
            current = np.radians(np.array(current_joints, dtype=np.float64))
            target = np.radians(np.array(target_joints, dtype=np.float64))

            # 计算误差
            error = target - current

            # 积分项（带饱和限制）
            self.error_sum += error * self.dt
            self.error_sum = np.clip(self.error_sum, -1.0, 1.0)

            # 微分项（防除零）
            error_diff = (error - self.last_error) / self.dt if self.dt > 1e-6 else np.zeros_like(error)

            # PID计算
            output = self.kp * error + self.ki * self.error_sum + self.kd * error_diff

            # 更新最后误差
            self.last_error = error.copy()

            return output.tolist()

    def reset(self):
        """重置PID状态"""
        with self._lock:
            self.error_sum = np.zeros_like(self.error_sum)
            self.last_error = np.zeros_like(self.last_error)


class TrajectoryManager:
    """轨迹管理器（持久化+校验）"""

    def __init__(self, save_path: str):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def save(self, trajectory: List[List[float]]) -> bool:
        """保存轨迹（带校验）"""
        if not trajectory or len(trajectory) == 0:
            logger.warning("空轨迹，无需保存")
            return False

        # 轨迹校验
        for idx, point in enumerate(trajectory):
            if len(point) != 6:
                logger.error(f"轨迹点{idx}格式错误：需6个关节角，当前{len(point)}个")
                return False

        # 保存数据
        data = {
            "version": "1.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "joint_num": 6,
            "trajectory": trajectory
        }

        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"轨迹已保存至：{self.save_path}（共{len(trajectory)}个点）")
            return True
        except Exception as e:
            logger.error(f"保存轨迹失败：{e}")
            return False

    def load(self) -> List[List[float]]:
        """加载轨迹（带校验）"""
        if not os.path.exists(self.save_path):
            logger.warning(f"轨迹文件不存在：{self.save_path}")
            return []

        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 版本/格式校验
            if data.get("version") != "1.0" or data.get("joint_num") != 6:
                logger.error("轨迹文件格式不兼容")
                return []

            trajectory = data.get("trajectory", [])
            logger.info(f"加载轨迹成功：{self.save_path}（共{len(trajectory)}个点）")
            return trajectory
        except Exception as e:
            logger.error(f"加载轨迹失败：{e}")
            return []


class TargetVisualizer:
    """目标点可视化（资源安全）"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, size: List[float]):
        self.model = model
        self.data = data
        self.size = size
        self.target_pos = np.array([0.1, 0.0, 0.3], dtype=np.float64)
        self.geom = self._create_geom()
        self._is_rendered = False

    def _create_geom(self) -> mujoco.MjvGeom:
        """创建目标点几何（红色球体）"""
        geom = mujoco.MjvGeom()
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array(self.size, dtype=np.float64), np.zeros(3), np.eye(3).flatten(), np.array([1.0, 0.0, 0.0, 1.0]))
        return geom

    def update(self, pos: List[float]) -> None:
        """更新目标点位置（带校验）"""
        if len(pos) != 3:
            logger.error(f"目标点位置格式错误：需3个坐标，当前{len(pos)}个")
            return
        self.target_pos = np.array(pos, dtype=np.float64)

    def render(self, viewer) -> None:
        """渲染目标点（MuJoCo 3.x兼容）"""
        try:
            self.geom.pos[:] = self.target_pos
            mujoco.mjv_addGeoms(viewer.model, viewer.data, viewer.user_scn, [self.geom])
            self._is_rendered = True
        except Exception as e:
            if not self._is_rendered:
                logger.warning(f"首次渲染目标点失败：{e}")


# ======================== 主仿真类（深度优化） ========================
class ArmSimulator:
    """机械臂仿真器（鲁棒+高性能）"""

    def __init__(self, config: SimConfig):
        self.config = config
        self._validate_config()  # 配置校验

        # 核心状态（初始化）
        self.running = False
        self.key_queue = queue.Queue(maxsize=10)  # 有限队列，避免内存溢出
        self.trajectory: List[List[float]] = []
        self.current_traj_idx = 0

        # 加载核心模块
        self._load_mujoco_model()
        self._load_kinematics_module()
        self._init_core_components()

    def _validate_config(self) -> None:
        """配置校验（提前发现错误）"""
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"MuJoCo模型文件不存在：{self.config.model_path}")

        if len(self.config.joint_speed_limits) != 6:
            raise ValueError(f"关节速度限制需6个值，当前{len(self.config.joint_speed_limits)}个")

    def _load_mujoco_model(self) -> None:
        """加载MuJoCo模型（带异常处理+资源清理）"""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.config.model_path)
            self.data = mujoco.MjData(self.model)
            logger.info(f"MuJoCo模型加载成功：{self.config.model_path}")
            logger.info(f"MuJoCo版本：{mujoco.__version__}")

            # 缓存ID（一次性查询，避免重复计算）
            self.joint_ids = self._get_mujoco_ids(mujoco.mjtObj.mjOBJ_JOINT, [f"joint{i + 1}" for i in range(6)])
            self.actuator_ids = self._get_mujoco_ids(mujoco.mjtObj.mjOBJ_ACTUATOR, [f"act{i + 1}" for i in range(6)])
        except Exception as e:
            logger.error(f"加载MuJoCo模型失败：{e}")
            raise

    def _load_kinematics_module(self) -> None:
        """加载运动学模块（容错导入）"""
        try:
            # 优先从core导入
            from core.kinematics import RoboticArmKinematics
            from core.arm_functions import ArmFunctions
        except ImportError:
            # 降级导入（添加路径）
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from core.kinematics import RoboticArmKinematics
            from core.arm_functions import ArmFunctions

        self.kinematics = RoboticArmKinematics(self.config.config_path)
        self.arm_functions = ArmFunctions(self.kinematics)
        logger.info("运动学模块加载成功")

    def _init_core_components(self) -> None:
        """初始化核心组件"""
        dt = 1.0 / self.config.fps
        self.pid = PIDController(
            kp=self.config.pid_params["kp"],
            ki=self.config.pid_params["ki"],
            kd=self.config.pid_params["kd"],
            dt=dt
        )
        self.traj_mgr = TrajectoryManager(self.config.traj_save_path)
        self.target_vis = TargetVisualizer(
            self.model, self.data,
            size=self.config.target_geom_size
        )
        logger.info("核心组件初始化完成")

    def _get_mujoco_ids(self, obj_type: int, names: List[str]) -> List[int]:
        """批量获取MuJoCo ID（带校验）"""
        ids = []
        for name in names:
            obj_id = mujoco.mj_name2id(self.model, obj_type, name)
            if obj_id == -1:
                raise ValueError(f"MuJoCo对象不存在：{name}")
            ids.append(obj_id)
        return ids

    # ======================== 核心功能方法 ========================
    def get_joint_angles(self) -> List[float]:
        """获取关节角（高性能+防抖）"""
        # 批量读取（减少循环次数）
        raw_radians = np.array([self.data.joint(jid).qpos[0] for jid in self.joint_ids], dtype=np.float64)
        raw_angles = np.degrees(raw_radians)

        # 轻量级防抖+归一化
        raw_angles = np.mod(raw_angles, 360.0)
        raw_angles[raw_angles > 180] -= 360.0

        # 裁剪到合理范围（防异常值）
        raw_angles = np.clip(raw_angles, -180.0, 180.0)
        return [round(angle, 2) for angle in raw_angles]

    def set_joint_angles(self, joint_angles: List[float], use_pid: bool = False) -> None:
        """设置关节角（PID/普通模式，带校验）"""
        if len(joint_angles) != 6:
            logger.error(f"关节角数量错误：需6个，当前{len(joint_angles)}个")
            return

        # 限位裁剪
        joint_angles = self.kinematics._clip_joint_angles(joint_angles)

        if use_pid:
            # PID控制（精准）
            current_angles = self.get_joint_angles()
            pid_output = self.pid.compute(current_angles, joint_angles)
            # 批量设置（提升性能）
            for i, act_id in enumerate(self.actuator_ids):
                self.data.ctrl[act_id] = pid_output[i]
        else:
            # 普通位置控制
            joint_radians = np.radians(joint_angles)
            for i, act_id in enumerate(self.actuator_ids):
                self.data.ctrl[act_id] = joint_radians[i]

    def clip_joint_speed(self, current: List[float], target: List[float]) -> List[float]:
        """关节速度限制（矢量运算，高性能）"""
        current_arr = np.array(current, dtype=np.float64)
        target_arr = np.array(target, dtype=np.float64)
        delta = target_arr - current_arr

        # 计算最大允许变化量
        max_delta = np.array(self.config.joint_speed_limits, dtype=np.float64) * (1.0 / self.config.fps)

        # 矢量裁剪（比循环快10倍+）
        delta_clipped = np.clip(delta, -max_delta, max_delta)
        return (current_arr + delta_clipped).tolist()

    # ======================== 交互控制 ========================
    def _manual_control_listener(self) -> None:
        """手动控制监听（优雅退出+指令校验）"""
        logger.info("\n===== 手动控制指令说明 =====")
        logger.info("1. 关节控制：j1+ / j1- / ... / j6+ / j6- (步长1度)")
        logger.info("2. 轨迹控制：save_traj (保存) | load_traj (加载)")
        logger.info("3. 目标控制：set_target x,y,z (示例：set_target 0.2,0,0.4)")
        logger.info("4. 系统控制：reset (重置PID) | quit (退出)")
        logger.info("===========================\n")

        while self.running:
            try:
                cmd = input("输入控制指令：").strip()
                if not cmd:
                    continue

                # 优雅退出
                if cmd.lower() == "quit":
                    self.running = False
                    break

                # 指令入队（非阻塞）
                if not self.key_queue.full():
                    self.key_queue.put(cmd, block=False)
                else:
                    logger.warning("指令队列已满，请稍后输入")
            except EOFError:
                logger.info("输入流关闭，退出手动控制")
                break
            except Exception as e:
                logger.error(f"读取指令失败：{e}")
                continue

    def _process_manual_command(self, cmd: str, current_joints: List[float]) -> List[float]:
        """处理手动指令（职责单一+易扩展）"""
        new_joints = current_joints.copy()

        # 1. 关节控制
        joint_cmd_map = {f"j{i + 1}+": i for i in range(6)} | {f"j{i + 1}-": i for i in range(6)}
        if cmd in joint_cmd_map:
            joint_idx = joint_cmd_map[cmd]
            step = 1.0 if '+' in cmd else -1.0
            new_joints[joint_idx] += step
            # 单关节限位
            new_joints[joint_idx] = np.clip(
                new_joints[joint_idx],
                self.kinematics.joint_limits[f"joint{joint_idx + 1}"][0],
                self.kinematics.joint_limits[f"joint{joint_idx + 1}"][1]
            )

        # 2. 轨迹控制
        elif cmd == "save_traj":
            self.traj_mgr.save(self.trajectory)
        elif cmd == "load_traj":
            self.trajectory = self.traj_mgr.load()
            self.current_traj_idx = 0

        # 3. 目标控制
        elif cmd.startswith("set_target"):
            self._process_set_target_cmd(cmd)

        # 4. 系统控制
        elif cmd == "reset":
            self.pid.reset()
            logger.info("PID控制器已重置")

        return new_joints

    def _process_set_target_cmd(self, cmd: str) -> None:
        """处理设置目标点指令（拆分职责）"""
        try:
            parts = cmd.split()
            if len(parts) != 2:
                raise ValueError("指令格式错误")

            pos_str = parts[1]
            pos = [float(x.strip()) for x in pos_str.split(",")]
            if len(pos) != 3:
                raise ValueError("需3个坐标值")

            # 坐标范围校验（防越界）
            pos = np.clip(pos, [-0.5, -0.5, 0.0], [0.5, 0.5, 1.0]).tolist()
            self.target_vis.update(pos)
            logger.info(f"目标点已更新：{pos}")
        except ValueError as e:
            logger.error(f"设置目标点失败：{e} | 正确格式：set_target 0.2,0,0.4")
        except Exception as e:
            logger.error(f"设置目标点异常：{e}")

    # ======================== 仿真模式处理 ========================
    def _init_trajectory_mode(self) -> None:
        """初始化轨迹模式（职责单一）"""
        start_joints = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
        target_joints = self.kinematics.inverse_kinematics([0.15, 0.0, 0.35, 0, 0, 0])
        self.trajectory = self.arm_functions.generate_linear_trajectory(start_joints, target_joints, 100)
        self.current_traj_idx = 0
        logger.info(f"轨迹模式初始化完成：{len(self.trajectory)}个轨迹点")

    def _update_trajectory_mode(self) -> None:
        """更新轨迹模式（高性能）"""
        if self.current_traj_idx < len(self.trajectory):
            current_joints = self.get_joint_angles()
            target_joints = self.trajectory[self.current_traj_idx]
            # 速度限制+PID控制
            limited_joints = self.clip_joint_speed(current_joints, target_joints)
            self.set_joint_angles(limited_joints, use_pid=True)
            self.current_traj_idx += 1

    def _update_follow_mode(self) -> None:
        """更新跟随模式（三维目标）"""
        # 生成动态目标点（正弦运动）
        t = time.time()
        target_pos = [
            0.1 + 0.05 * np.sin(t),
            0.0 + 0.03 * np.cos(t),
            0.3 + 0.04 * np.sin(t / 2)
        ]
        self.target_vis.update(target_pos)

        # 逆解+速度限制
        current_joints = self.get_joint_angles()
        target_joints = self.arm_functions.follow_moving_target(current_joints, target_pos)
        limited_joints = self.clip_joint_speed(current_joints, target_joints)
        self.set_joint_angles(limited_joints)

    def _update_manual_mode(self) -> None:
        """更新手动模式（非阻塞）"""
        try:
            cmd = self.key_queue.get_nowait()
            current_joints = self.get_joint_angles()
            new_joints = self._process_manual_command(cmd, current_joints)
            self.set_joint_angles(new_joints)
        except queue.Empty:
            pass

    # ======================== 主仿真循环 ========================
    def run(self, mode: str) -> None:
        """运行仿真（主入口，高性能+鲁棒）"""
        if mode not in ["trajectory", "manual", "follow"]:
            raise ValueError(f"无效模式：{mode} | 支持：trajectory/manual/follow")

        # 初始化运行状态
        self.running = True
        start_time = time.time()
        frame_count = 0
        dt = 1.0 / self.config.fps

        # 模式初始化
        if mode == "trajectory":
            self._init_trajectory_mode()
        elif mode == "manual":
            # 启动手动控制线程（守护线程，优雅退出）
            threading.Thread(target=self._manual_control_listener, daemon=True).start()

        logger.info(f"启动仿真模式：{mode} | 帧率：{self.config.fps}FPS | 时长：{self.config.sim_duration}s")

        # 仿真主循环（高性能）
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # 初始化视角
                self._init_viewer(viewer)

                # 循环优化：减少属性查找次数
                mj_step = mujoco.mj_step
                model = self.model
                data = self.data
                target_vis_render = self.target_vis.render
                viewer_sync = viewer.sync

                while self.running and (time.time() - start_time) < self.config.sim_duration:
                    frame_start = time.time()

                    # 按模式更新
                    if mode == "trajectory":
                        self._update_trajectory_mode()
                    elif mode == "manual":
                        self._update_manual_mode()
                    elif mode == "follow":
                        self._update_follow_mode()

                    # 核心仿真步骤（批量调用，减少开销）
                    mj_step(model, data)
                    target_vis_render(viewer)
                    viewer_sync()

                    # 状态打印（每5帧一次，减少IO）
                    if frame_count % 5 == 0:
                        self._print_sim_status(start_time)

                    # 帧率控制（精准）
                    frame_elapsed = time.time() - frame_start
                    if frame_elapsed < dt:
                        time.sleep(dt - frame_elapsed)

                    frame_count += 1
        finally:
            # 资源清理（优雅退出）
            self.running = False
            self.pid.reset()
            logger.info(f"仿真结束 | 总帧数：{frame_count} | 平均帧率：{frame_count / (time.time() - start_time):.1f}FPS")

    def _init_viewer(self, viewer) -> None:
        """初始化Viewer视角（集中配置）"""
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -15.0
        viewer.cam.lookat = np.array([0.2, 0.0, 0.5], dtype=np.float64)

    def _print_sim_status(self, start_time: float) -> None:
        """打印仿真状态（格式化输出）"""
        try:
            elapsed = time.time() - start_time
            current_joints = self.get_joint_angles()
            current_pose = self.kinematics.forward_kinematics(current_joints)

            # 格式化输出（减少换行）
            status_str = (
                f"\r仿真时长：{elapsed:.1f}s | "
                f"关节角：{current_joints} | "
                f"末端位姿：{current_pose}"
            )
            print(status_str, end="", flush=True)
        except Exception as e:
            logger.debug(f"打印状态失败：{e}")


# ======================== 主函数（用户交互） ========================
def get_user_mode_choice() -> str:
    """获取用户模式选择（交互优化）"""
    print("\n========================")
    print("      机械臂仿真系统      ")
    print("========================")
    print("支持模式：")
    print("1 - 轨迹规划（PID精准控制）")
    print("2 - 手动控制（增强指令）")
    print("3 - 目标跟随（可视化目标点）")
    print("========================")

    while True:
        choice = input("请输入模式编号（1/2/3）：").strip()
        mode_map = {"1": "trajectory", "2": "manual", "3": "follow"}
        if choice in mode_map:
            return mode_map[choice]
        print("输入错误！请输入 1、2 或 3")


def main() -> None:
    """主函数（入口）"""
    try:
        # 加载配置
        config = SimConfig(
            model_path=os.path.join(os.path.dirname(__file__), "model/six_axis_arm.xml"),
            sim_duration=30.0
        )

        # 创建仿真器
        simulator = ArmSimulator(config)

        # 获取用户选择
        mode = get_user_mode_choice()

        # 运行仿真
        simulator.run(mode)
    except KeyboardInterrupt:
        logger.info("用户中断程序，优雅退出")
    except Exception as e:
        logger.error(f"程序异常：{e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()