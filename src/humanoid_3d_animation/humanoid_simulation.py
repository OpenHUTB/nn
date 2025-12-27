import mujoco
import mujoco.viewer as viewer
import os
import time
import math
import threading
import signal
import sys
import random
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Optional, Tuple
import logging

# ====================== 日志配置 ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ====================== 全局配置 ======================
# 全局运行状态
sim_running = True
# 线程安全锁
data_lock = threading.Lock()


# ====================== 配置类（增强版） ======================
@dataclass
class SimConfig:
    """仿真配置类：集中管理所有可配置参数"""
    # 仿真核心参数
    timestep: float = 0.005
    sim_frequency: float = 2.0
    state_print_interval: float = 1.0

    # 相机参数
    cam_distance: float = 2.0
    cam_azimuth: float = 45.0
    cam_elevation: float = -20.0

    # 关节运动幅度配置
    joint_amplitudes: Dict[str, float] = field(default_factory=lambda: {
        "left_shoulder": 1.2, "right_shoulder": 1.2,
        "left_elbow": 1.0, "right_elbow": 1.0,
        "left_hip": 1.0, "right_hip": 1.0,
        "left_knee": 1.2, "right_knee": 1.2
    })

    # 控制模式
    default_mode: str = "walk"
    supported_modes: List[str] = field(default_factory=lambda: ["walk", "wave", "sin", "random", "stop"])

    # 可视化配置
    plot_update_interval: int = 50
    max_plot_points: int = 200
    plot_refresh_ms: int = 50

    # 动作参数
    walk_stride: float = 0.8
    wave_frequency: float = 1.5
    smooth_factor: float = 0.05  # 控制信号平滑因子

    # 性能配置
    max_fps: int = 60  # 最大帧率限制
    step_sleep: float = 0.001  # 步长休眠时间


# ====================== 信号处理 ======================
def signal_handler(sig: int, frame) -> None:
    """优雅处理中断信号"""
    global sim_running
    sim_running = False
    logger.warning("收到中断信号，正在优雅退出仿真...")


signal.signal(signal.SIGINT, signal_handler)


# ====================== 核心仿真类 ======================
class HumanoidSimulator:
    def __init__(self, config: SimConfig):
        self.config = config
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None

        # 关节相关
        self.joint_names: List[str] = list(config.joint_amplitudes.keys())
        self.joint_ctrl_ids: Dict[str, int] = {}
        self.joint_qpos_indices: Dict[str, int] = {}
        self.joint_limits: Dict[str, Tuple[float, float]] = {}  # 关节限位

        # 控制状态
        self.current_mode: str = config.default_mode
        self.last_ctrl_signals: Dict[str, float] = {name: 0.0 for name in self.joint_names}

        # 动作状态
        self.walk_phase: float = 0.0
        self.wave_arm: str = "right"

        # 可视化
        self.plot_data: Dict[str, List[float]] = {name: [] for name in self.joint_names}
        self.time_data: List[float] = []
        self.frame_counter: int = 0

        # 性能监控
        self.last_print_time: float = 0.0
        self.frame_count: int = 0
        self.start_time: float = 0.0
        self.fps: float = 0.0

        # 绘图对象
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.lines: Dict[str, plt.Line2D] = {}
        self.ani: Optional[FuncAnimation] = None

    def load_model(self) -> None:
        """加载并验证MuJoCo模型"""
        xml_content = self._get_robot_xml()

        try:
            self.model = mujoco.MjModel.from_xml_string(xml_content)
            self.data = mujoco.MjData(self.model)
            logger.info("✅ 模型加载成功")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
            sys.exit(1)

        # 初始化关节信息
        self._init_joint_info()

        # 验证控制数组
        logger.info(f"📊 控制信号数组长度: {len(self.data.ctrl)}")
        logger.info(f"📊 关节位置数组长度: {len(self.data.qpos)}")

    def _get_robot_xml(self) -> str:
        """返回优化后的机器人XML描述"""
        return """<mujoco model="optimized_humanoid">
  <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" iterations="100" tolerance="1e-6"/>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1" specular="0.1 0.1 0.1"/>
    <body name="ground" pos="0 0 0">
      <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1" condim="3"/>
    </body>

    <!-- 优化的机器人结构 -->
    <body name="pelvis" pos="0 0 1.0">
      <joint name="root" type="free"/>
      <geom name="pelvis_geom" type="capsule" size="0.1" fromto="0 0 0 0 0 0.2" rgba="0.5 0.5 0.9 1" mass="5"/>

      <body name="torso" pos="0 0 0.2">
        <geom name="torso_geom" type="capsule" size="0.1" fromto="0 0 0 0 0 0.3" rgba="0.5 0.5 0.9 1" mass="8"/>

        <body name="head" pos="0 0 0.3">
          <geom name="head_geom" type="sphere" size="0.15" pos="0 0 0" rgba="0.8 0.5 0.5 1" mass="3"/>
        </body>

        <!-- 左手臂 -->
        <body name="left_arm" pos="0.15 0 0.15">
          <joint name="left_shoulder" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="0.5"/>
          <geom name="left_upper_arm" type="capsule" size="0.05" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1" mass="1"/>
          <body name="left_forearm" pos="0 0 0.2">
            <joint name="left_elbow" type="hinge" axis="1 0 0" range="-1.57 0" damping="0.5"/>
            <geom name="left_forearm_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1" mass="0.5"/>
          </body>
        </body>

        <!-- 右手臂 -->
        <body name="right_arm" pos="-0.15 0 0.15">
          <joint name="right_shoulder" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="0.5"/>
          <geom name="right_upper_arm" type="capsule" size="0.05" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1" mass="1"/>
          <body name="right_forearm" pos="0 0 0.2">
            <joint name="right_elbow" type="hinge" axis="1 0 0" range="-1.57 0" damping="0.5"/>
            <geom name="right_forearm_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1" mass="0.5"/>
          </body>
        </body>

        <!-- 左腿部 -->
        <body name="left_leg" pos="0.05 0 -0.2">
          <joint name="left_hip" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="0.8"/>
          <geom name="left_thigh" type="capsule" size="0.06" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.5 1" mass="2"/>
          <body name="left_calf" pos="0 0 -0.3">
            <joint name="left_knee" type="hinge" axis="1 0 0" range="0 1.57" damping="0.8"/>
            <geom name="left_calf_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.5 1" mass="1"/>
          </body>
        </body>

        <!-- 右腿部 -->
        <body name="right_leg" pos="-0.05 0 -0.2">
          <joint name="right_hip" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="0.8"/>
          <geom name="right_thigh" type="capsule" size="0.06" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.5 1" mass="2"/>
          <body name="right_calf" pos="0 0 -0.3">
            <joint name="right_knee" type="hinge" axis="1 0 0" range="0 1.57" damping="0.8"/>
            <geom name="right_calf_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.5 1" mass="1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- 优化的执行器 -->
  <actuator>
    <motor name="left_shoulder" joint="left_shoulder" ctrlrange="-1.57 1.57" gear="20" ctrllimited="true"/>
    <motor name="right_shoulder" joint="right_shoulder" ctrlrange="-1.57 1.57" gear="20" ctrllimited="true"/>
    <motor name="left_elbow" joint="left_elbow" ctrlrange="-1.57 0" gear="15" ctrllimited="true"/>
    <motor name="right_elbow" joint="right_elbow" ctrlrange="-1.57 0" gear="15" ctrllimited="true"/>
    <motor name="left_hip" joint="left_hip" ctrlrange="-1.57 1.57" gear="25" ctrllimited="true"/>
    <motor name="right_hip" joint="right_hip" ctrlrange="-1.57 1.57" gear="25" ctrllimited="true"/>
    <motor name="left_knee" joint="left_knee" ctrlrange="0 1.57" gear="20" ctrllimited="true"/>
    <motor name="right_knee" joint="right_knee" ctrlrange="0 1.57" gear="20" ctrllimited="true"/>
  </actuator>
</mujoco>"""

    def _init_joint_info(self) -> None:
        """初始化关节ID、限位等信息"""
        logger.info("\n🔍 关节信息初始化:")
        for name in self.joint_names:
            # 获取控制ID
            ctrl_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.joint_ctrl_ids[name] = ctrl_id

            # 获取关节ID和位置索引
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id != -1:
                self.joint_qpos_indices[name] = self.model.jnt_qposadr[joint_id]
                # 获取关节限位
                self.joint_limits[name] = (
                    self.model.jnt_range[joint_id][0],
                    self.model.jnt_range[joint_id][1]
                )
            else:
                self.joint_qpos_indices[name] = -1
                self.joint_limits[name] = (-np.pi, np.pi)

            logger.info(
                f"  {name}: ctrl_id={ctrl_id}, qpos_idx={self.joint_qpos_indices[name]}, "
                f"limits={self.joint_limits[name]}"
            )

    def get_joint_ctrl_signal(self, name: str, t: float) -> float:
        """生成关节控制信号（统一入口）"""
        if self.current_mode not in self.config.supported_modes:
            logger.warning(f"未知模式 {self.current_mode}，切换到默认模式")
            self.current_mode = self.config.default_mode

        # 根据模式生成信号
        if self.current_mode == "walk":
            signal = self._get_walk_action(name, t)
        elif self.current_mode == "wave":
            signal = self._get_wave_action(name, t)
        elif self.current_mode == "sin":
            signal = self._get_sin_action(name, t)
        elif self.current_mode == "random":
            signal = self._get_random_action(name, t)
        else:  # stop
            signal = 0.0

        # 平滑过渡和限位
        signal = self._smooth_and_limit_signal(name, signal)
        return signal

    def _get_walk_action(self, name: str, t: float) -> float:
        """生成行走动作控制信号"""
        amplitude = self.config.joint_amplitudes[name]
        stride = self.config.walk_stride

        # 更新行走相位（优化计算）
        self.walk_phase = (self.walk_phase + 0.01) % (2 * math.pi)

        if "hip" in name:
            phase_offset = math.pi if "right" in name else 0
            signal = math.sin(self.walk_phase + phase_offset) * amplitude * stride
        elif "knee" in name:
            phase_offset = math.pi if "right" in name else 0
            signal = math.cos(self.walk_phase + phase_offset) * amplitude * stride * 1.2
        elif "shoulder" in name:
            phase_offset = 0 if "right" in name else math.pi
            signal = math.sin(self.walk_phase + phase_offset) * amplitude * 0.5
        elif "elbow" in name:
            phase_offset = 0 if "right" in name else math.pi
            signal = -math.fabs(math.sin(self.walk_phase + phase_offset)) * amplitude * 0.6
        else:
            signal = 0.0

        return signal

    def _get_wave_action(self, name: str, t: float) -> float:
        """生成挥手动作控制信号"""
        amplitude = self.config.joint_amplitudes[name]
        freq = self.config.wave_frequency

        # 优化手臂切换逻辑
        self.wave_arm = "right" if (int(t) % 2 == 0) else "left"

        if f"{self.wave_arm}_shoulder" == name:
            signal = math.sin(t * freq) * amplitude * 1.2
        elif f"{self.wave_arm}_elbow" == name:
            signal = -math.fabs(math.sin(t * freq)) * amplitude * 1.0
        elif "shoulder" in name:
            signal = -0.2
        elif "elbow" in name:
            signal = -0.8
        else:
            signal = 0.0

        return signal

    def _get_sin_action(self, name: str, t: float) -> float:
        """生成正弦运动信号"""
        amplitude = self.config.joint_amplitudes[name]
        if "left" in name:
            return math.sin(t * self.config.sim_frequency) * amplitude
        else:
            return -math.sin(t * self.config.sim_frequency) * amplitude

    def _get_random_action(self, name: str, t: float) -> float:
        """生成随机运动信号"""
        # 优化随机数生成（减少抖动）
        if int(t * 10) % 2 == 0:  # 每0.2秒更新一次随机值
            self.last_ctrl_signals[name] = (random.random() * 2 - 1) * self.config.joint_amplitudes[name]
        return self.last_ctrl_signals[name]

    def _smooth_and_limit_signal(self, name: str, signal: float) -> float:
        """平滑控制信号并限制在关节范围内"""
        # 指数平滑
        smoothed = (1 - self.config.smooth_factor) * self.last_ctrl_signals[name] + \
                   self.config.smooth_factor * signal

        # 关节限位
        min_limit, max_limit = self.joint_limits[name]
        limited = np.clip(smoothed, min_limit, max_limit)

        # 更新最后信号值
        self.last_ctrl_signals[name] = limited

        return limited

    def update_joint_controls(self) -> None:
        """更新关节控制信号（优化版）"""
        t = self.data.time
        for name in self.joint_names:
            ctrl_id = self.joint_ctrl_ids[name]
            if ctrl_id == -1 or ctrl_id >= len(self.data.ctrl):
                continue

            try:
                ctrl_signal = self.get_joint_ctrl_signal(name, t)
                self.data.ctrl[ctrl_id] = ctrl_signal
            except Exception as e:
                logger.error(f"⚠️ 关节 {name} 控制失败: {e}")

    def collect_plot_data(self) -> None:
        """优化的绘图数据收集（减少锁竞争）"""
        self.frame_counter += 1
        if self.frame_counter % self.config.plot_update_interval != 0:
            return

        with data_lock:
            current_time = self.data.time
            self.time_data.append(current_time)

            # 批量更新数据
            for name in self.joint_names:
                qpos_idx = self.joint_qpos_indices[name]
                if 0 <= qpos_idx < len(self.data.qpos):
                    self.plot_data[name].append(self.data.qpos[qpos_idx])
                else:
                    self.plot_data[name].append(0.0)

            # 限制数据长度（优化切片操作）
            if len(self.time_data) > self.config.max_plot_points:
                excess = len(self.time_data) - self.config.max_plot_points
                self.time_data = self.time_data[excess:]
                for name in self.joint_names:
                    self.plot_data[name] = self.plot_data[name][excess:]

    def init_plot(self) -> None:
        """初始化优化的绘图界面"""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig, self.ax = plt.subplots(figsize=(12, 8))

        # 设置图表属性
        self.ax.set_xlabel('Time (s)', fontsize=12)
        self.ax.set_ylabel('Joint Angle (rad)', fontsize=12)
        self.ax.set_title('Real-time Joint Angle Monitoring', fontsize=14, fontweight='bold')
        self.ax.set_ylim(-2, 2)
        self.ax.grid(True, alpha=0.3)

        # 优化的颜色和线型方案
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

        # 创建线条（预分配）
        for i, name in enumerate(self.joint_names):
            line, = self.ax.plot([], [], label=name,
                                 color=colors[i % len(colors)],
                                 linestyle=linestyles[i % len(linestyles)],
                                 linewidth=2, alpha=0.8)
            self.lines[name] = line

        self.ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        plt.tight_layout()

        # 禁用matplotlib交互模式的自动更新
        plt.ioff()
        logger.info("📊 关节角度可视化图表已创建")

    def update_plot(self, frame) -> List[plt.Line2D]:
        """优化的绘图更新（减少重绘）"""
        with data_lock:
            if not self.time_data:
                return list(self.lines.values())

            # 批量更新数据
            for name, line in self.lines.items():
                if len(self.plot_data[name]) == len(self.time_data):
                    line.set_data(self.time_data, self.plot_data[name])

            # 智能更新X轴范围
            x_min = max(0, self.time_data[-1] - 10)
            x_max = self.time_data[-1] + 1
            self.ax.set_xlim(x_min, x_max)

        return list(self.lines.values())

    def print_robot_state(self) -> None:
        """优化的状态打印（减少IO操作）"""
        current_time = self.data.time

        # 初始化性能监控
        if not hasattr(self, 'start_time'):
            self.start_time = current_time
            self.frame_count = 0
            self.fps = 0.0

        self.frame_count += 1
        elapsed_time = current_time - self.start_time

        # 计算FPS（避免除以零）
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time

        # 按时间间隔打印
        if current_time - self.last_print_time >= self.config.state_print_interval:
            logger.info(
                f"\n===== 机器人状态 | 时间: {current_time:.2f}s | 帧率: {self.fps:.1f} FPS | 模式: {self.current_mode} ====="
            )

            # 批量打印关节状态
            joint_states = []
            for name in self.joint_names:
                ctrl_id = self.joint_ctrl_ids[name]
                qpos_idx = self.joint_qpos_indices[name]

                if 0 <= ctrl_id < len(self.data.ctrl) and 0 <= qpos_idx < len(self.data.qpos):
                    joint_states.append(
                        f"{name}: 位置={self.data.qpos[qpos_idx]:.2f}rad, 控制={self.data.ctrl[ctrl_id]:.2f}"
                    )

            logger.info("\n".join(joint_states))
            self.last_print_time = current_time

    def reset_robot(self) -> None:
        """优化的机器人重置"""
        with data_lock:
            mujoco.mj_resetData(self.model, self.data)
            # 优化初始位置设置
            self.data.qpos[:7] = [0, 0, 1.0, 1, 0, 0, 0]

            # 重置控制状态
            for name in self.joint_names:
                self.last_ctrl_signals[name] = 0.0
                ctrl_id = self.joint_ctrl_ids[name]
                if 0 <= ctrl_id < len(self.data.ctrl):
                    self.data.ctrl[ctrl_id] = 0.0

            # 重置动作状态
            self.walk_phase = 0.0
            self.wave_arm = "right"

            # 清空绘图数据
            self.plot_data = {name: [] for name in self.joint_names}
            self.time_data = []
            self.frame_counter = 0

            # 重置性能监控
            self.frame_count = 0
            self.start_time = self.data.time
            self.fps = 0.0

        logger.info("🔄 机器人已重置到初始状态")

    def _get_user_input(self) -> Optional[str]:
        """跨平台用户输入获取"""
        if sys.platform == 'win32':
            try:
                import msvcrt
                if msvcrt.kbhit():
                    return msvcrt.readline().decode().strip().lower()
            except ImportError:
                pass
        else:
            # Unix系统非阻塞输入
            try:
                import select
                if select.select([sys.stdin], [], [], 0)[0]:
                    return sys.stdin.readline().strip().lower()
            except:
                pass
        return None

    def process_user_input(self) -> None:
        """处理用户输入（优化版）"""
        user_input = self._get_user_input()
        if not user_input:
            return

        command_map = {
            'r': self.reset_robot,
            'q': lambda: globals().update(sim_running=False),
            'clear': self._clear_plot_data
        }

        # 处理模式切换
        if user_input in self.config.supported_modes:
            self.current_mode = user_input
            mode_descriptions = {
                "walk": "👣 行走模式：机器人进行自然行走动作",
                "wave": "✋ 挥手模式：机器人交替挥动手臂",
                "sin": "📈 正弦模式：关节做正弦规律运动",
                "random": "🎲 随机模式：关节做随机运动",
                "stop": "🛑 停止模式：所有关节停止运动"
            }
            logger.info(f"\n🔄 运动模式切换为: {user_input}")
            logger.info(mode_descriptions.get(user_input, ""))
        # 处理其他命令
        elif user_input in command_map:
            command_map[user_input]()
            if user_input == 'q':
                logger.info("\n📤 收到退出指令，仿真将结束...")
        else:
            self._print_help()

    def _clear_plot_data(self) -> None:
        """清空绘图数据"""
        with data_lock:
            self.plot_data = {name: [] for name in self.joint_names}
            self.time_data = []
        logger.info("🧹 绘图数据已清空")

    def _print_help(self) -> None:
        """打印优化的帮助信息"""
        help_text = """
❓ 支持的指令：
  - r         : 重置机器人到初始状态
  - walk      : 行走模式（自然行走动作）
  - wave      : 挥手模式（交替挥动手臂）
  - sin       : 正弦模式（关节正弦运动）
  - random    : 随机模式（关节随机运动）
  - stop      : 停止模式（所有关节停止）
  - clear     : 清空绘图数据
  - q         : 退出仿真
  - help      : 显示此帮助信息
"""
        logger.info(help_text)

    def run_simulation(self) -> None:
        """优化的仿真主循环"""
        # 初始化
        self.load_model()
        self.init_plot()

        # 启动动画
        self.ani = FuncAnimation(
            self.fig, self.update_plot,
            interval=self.config.plot_refresh_ms,
            blit=True,
            cache_frame_data=False
        )

        # 显示绘图窗口
        plt.show(block=False)

        # 启动MuJoCo可视化
        with viewer.launch_passive(self.model, self.data) as v:
            # 配置相机
            self._setup_camera(v)

            # 打印操作提示
            self._print_help()
            logger.info(f"\n🚀 仿真开始（默认模式：{self.config.default_mode}）")

            # 帧率控制
            frame_interval = 1.0 / self.config.max_fps
            last_step_time = time.perf_counter()

            # 主循环
            while sim_running and v.is_running():
                current_time = time.perf_counter()

                # 处理用户输入
                self.process_user_input()

                # 固定步长执行仿真
                if current_time - last_step_time >= frame_interval:
                    try:
                        # 更新控制并执行仿真步
                        self.update_joint_controls()
                        mujoco.mj_step(self.model, self.data)

                        # 更新可视化
                        v.sync()

                        # 数据收集和状态打印
                        self.collect_plot_data()
                        self.print_robot_state()

                        last_step_time = current_time
                    except Exception as e:
                        logger.error(f"⚠️ 仿真步执行失败: {e}", exc_info=True)
                        self.reset_robot()

                # 处理matplotlib事件（减少CPU占用）
                plt.pause(self.config.step_sleep)

        # 清理资源
        self._cleanup()

    def _setup_camera(self, v) -> None:
        """配置相机参数"""
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_id != -1:
            v.cam.trackbodyid = pelvis_id
        v.cam.distance = self.config.cam_distance
        v.cam.azimuth = self.config.cam_azimuth
        v.cam.elevation = self.config.cam_elevation

    def _cleanup(self) -> None:
        """优雅清理资源"""
        if self.fig:
            plt.close(self.fig)
        logger.info("\n🏁 仿真结束，资源已清理")


# ====================== 程序入口 ======================
def main() -> None:
    """程序主入口"""
    # 设置matplotlib后端
    import matplotlib
    matplotlib.use('TkAgg')

    # Windows编码修复
    if sys.platform == 'win32':
        try:
            import subprocess
            subprocess.call('chcp 65001', shell=True, stdout=subprocess.DEVNULL)
        except:
            pass

    # 初始化配置和仿真器
    config = SimConfig()
    simulator = HumanoidSimulator(config)

    # 运行仿真
    try:
        simulator.run_simulation()
    except KeyboardInterrupt:
        global sim_running
        sim_running = False
        logger.warning("\n⚠️ 用户中断，正在退出...")
    except Exception as e:
        logger.error(f"\n❌ 程序异常: {e}", exc_info=True)
    finally:
        plt.close('all')
        sys.exit(0)


if __name__ == "__main__":
    main()