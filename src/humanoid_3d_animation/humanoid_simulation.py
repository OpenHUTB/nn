import mujoco
import mujoco.viewer as viewer
import os
import time
import math
import threading
import signal
import sys
import random
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ====================== 配置抽离 ======================
@dataclass
class SimConfig:
    """仿真配置类：集中管理所有可配置参数"""
    # 仿真参数
    timestep: float = 0.005
    sim_frequency: float = 2.0
    state_print_interval: float = 1.0
    # 相机参数
    cam_distance: float = 2.0
    cam_azimuth: float = 45.0
    cam_elevation: float = -20.0
    # 关节运动幅度配置（针对不同动作优化）
    joint_amplitudes = {
        "left_shoulder": 1.2, "right_shoulder": 1.2,
        "left_elbow": 1.0, "right_elbow": 1.0,
        "left_hip": 1.0, "right_hip": 1.0,
        "left_knee": 1.2, "right_knee": 1.2
    }
    # 控制模式（新增行走和挥手动作）
    default_mode: str = "walk"
    # 可视化配置
    plot_update_interval: int = 50  # 绘图更新间隔（帧数）
    max_plot_points: int = 200  # 图表最大显示数据点
    # 动作参数
    walk_stride: float = 0.8  # 行走步幅
    wave_frequency: float = 1.5  # 挥手频率


# 全局变量
sim_running = True
# 用于线程间数据共享的锁
data_lock = threading.Lock()


def signal_handler(sig, frame):
    """处理Ctrl+C中断信号"""
    global sim_running
    sim_running = False
    print("\n⚠️ 收到中断信号，正在退出仿真...")


signal.signal(signal.SIGINT, signal_handler)


# ====================== 核心功能类 ======================
class HumanoidSimulator:
    def __init__(self, config: SimConfig):
        self.config = config
        self.model = None
        self.data = None
        self.joint_names = list(config.joint_amplitudes.keys())
        self.joint_ctrl_ids = {}
        self.joint_qpos_indices = {}
        self.current_mode = config.default_mode
        self.last_ctrl_signals = {}

        # 新增：动作状态变量
        self.walk_phase = 0.0  # 行走相位
        self.wave_arm = "right"  # 当前挥动手臂

        # 可视化相关变量
        self.plot_data = {name: [] for name in self.joint_names}
        self.time_data = []
        self.frame_counter = 0

        # 绘图相关
        self.fig, self.ax = None, None
        self.lines = {}
        self.ani = None

    def load_model(self):
        """加载MuJoCo模型（完全修复XML格式）"""
        # 完全兼容所有MuJoCo版本的XML
        xml_content = """<mujoco model="simple_humanoid">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1" specular="0.1 0.1 0.1"/>

    <!-- 地面body（用于约束） -->
    <body name="ground" pos="0 0 0">
      <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
    </body>

    <!-- 机器人主体 -->
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

  <!-- 执行器定义（简化命名，与关节名一致） -->
  <actuator>
    <!-- 手臂关节电机 -->
    <motor name="left_shoulder" joint="left_shoulder" ctrlrange="-1.57 1.57" gear="20"/>
    <motor name="right_shoulder" joint="right_shoulder" ctrlrange="-1.57 1.57" gear="20"/>
    <motor name="left_elbow" joint="left_elbow" ctrlrange="-1.57 0" gear="15"/>
    <motor name="right_elbow" joint="right_elbow" ctrlrange="-1.57 0" gear="15"/>

    <!-- 腿部关节电机 -->
    <motor name="left_hip" joint="left_hip" ctrlrange="-1.57 1.57" gear="25"/>
    <motor name="right_hip" joint="right_hip" ctrlrange="-1.57 1.57" gear="25"/>
    <motor name="left_knee" joint="left_knee" ctrlrange="0 1.57" gear="20"/>
    <motor name="right_knee" joint="right_knee" ctrlrange="0 1.57" gear="20"/>
  </actuator>

  <!-- 可选：移除weld约束，让机器人可以自由运动 -->
  <!-- <equality>
    <weld body1="ground" body2="pelvis"/>
  </equality> -->
</mujoco>"""

        try:
            # 直接从XML字符串加载模型
            self.model = mujoco.MjModel.from_xml_string(xml_content)
            self.data = mujoco.MjData(self.model)
            print("✅ 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # 映射关节ID
        print("\n🔍 关节ID映射结果：")
        for name in self.joint_names:
            # 直接使用关节名作为电机名
            ctrl_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.joint_ctrl_ids[name] = ctrl_id

            # 获取关节ID和位置索引
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id != -1:
                self.joint_qpos_indices[name] = self.model.jnt_qposadr[joint_id]
            else:
                self.joint_qpos_indices[name] = -1

            self.last_ctrl_signals[name] = 0.0

            # 打印详细信息
            print(f"  {name}: ctrl_id={ctrl_id}, qpos_idx={self.joint_qpos_indices[name]}")

        # 验证控制信号数组
        print(f"\n📊 控制信号数组长度：{len(self.data.ctrl)}")
        print(f"📊 关节位置数组长度：{len(self.data.qpos)}")

    def get_walk_action(self, name, t):
        """生成行走动作控制信号"""
        amplitude = self.config.joint_amplitudes[name]
        stride = self.config.walk_stride

        # 更新行走相位
        self.walk_phase = (self.walk_phase + 0.01) % (2 * math.pi)

        if "hip" in name:
            # 髋关节交替摆动
            if "left" in name:
                signal = math.sin(self.walk_phase) * amplitude * stride
            else:
                signal = math.sin(self.walk_phase + math.pi) * amplitude * stride
        elif "knee" in name:
            # 膝关节配合髋关节运动
            if "left" in name:
                signal = math.cos(self.walk_phase) * amplitude * stride * 1.2
            else:
                signal = math.cos(self.walk_phase + math.pi) * amplitude * stride * 1.2
        elif "shoulder" in name:
            # 手臂自然摆动（与对侧腿相反）
            if "left" in name:
                signal = math.sin(self.walk_phase + math.pi) * amplitude * 0.5
            else:
                signal = math.sin(self.walk_phase) * amplitude * 0.5
        elif "elbow" in name:
            # 肘部轻微弯曲
            if "left" in name:
                signal = -math.fabs(math.sin(self.walk_phase + math.pi)) * amplitude * 0.6
            else:
                signal = -math.fabs(math.sin(self.walk_phase)) * amplitude * 0.6
        else:
            signal = 0.0

        return signal

    def get_wave_action(self, name, t):
        """生成挥手动作控制信号"""
        amplitude = self.config.joint_amplitudes[name]
        freq = self.config.wave_frequency

        # 每2秒切换一次挥动手臂
        if int(t) % 2 == 0:
            self.wave_arm = "right"
        else:
            self.wave_arm = "left"

        signal = 0.0

        # 挥动手臂的肩部和肘部运动
        if f"{self.wave_arm}_shoulder" == name:
            # 肩部上下摆动
            signal = math.sin(t * freq) * amplitude * 1.2
        elif f"{self.wave_arm}_elbow" == name:
            # 肘部配合弯曲
            signal = -math.fabs(math.sin(t * freq)) * amplitude * 1.0
        # 另一只手臂保持自然下垂
        elif ("shoulder" in name and self.wave_arm not in name):
            signal = -0.2
        elif ("elbow" in name and self.wave_arm not in name):
            signal = -0.8
        # 腿部保持稳定
        elif "hip" in name or "knee" in name:
            signal = 0.0

        return signal

    def get_joint_ctrl_signal(self, name, t):
        """生成关节控制信号（支持多种动作模式）"""
        # 根据当前模式选择动作
        if self.current_mode == "walk":
            signal = self.get_walk_action(name, t)
        elif self.current_mode == "wave":
            signal = self.get_wave_action(name, t)
        elif self.current_mode == "sin":
            # 原有正弦运动模式
            if "left" in name:
                signal = math.sin(t * self.config.sim_frequency) * self.config.joint_amplitudes[name]
            else:
                signal = -math.sin(t * self.config.sim_frequency) * self.config.joint_amplitudes[name]
        elif self.current_mode == "random":
            # 随机运动模式
            signal = (random.random() * 2 - 1) * self.config.joint_amplitudes[name]
        elif self.current_mode == "stop":
            # 停止模式
            signal = 0.0
        else:
            signal = 0.0

        # 平滑过渡
        smooth_factor = 0.05
        self.last_ctrl_signals[name] = (1 - smooth_factor) * self.last_ctrl_signals[name] + smooth_factor * signal

        # 限制信号范围在关节限位内
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id != -1:
            jnt_range = self.model.jnt_range[joint_id]
            self.last_ctrl_signals[name] = np.clip(self.last_ctrl_signals[name], jnt_range[0], jnt_range[1])

        return self.last_ctrl_signals[name]

    def update_joint_controls(self):
        """更新关节控制信号"""
        t = self.data.time
        for name in self.joint_names:
            ctrl_id = self.joint_ctrl_ids[name]
            if ctrl_id == -1:
                continue

            try:
                ctrl_signal = self.get_joint_ctrl_signal(name, t)
                if 0 <= ctrl_id < len(self.data.ctrl):
                    self.data.ctrl[ctrl_id] = ctrl_signal
                else:
                    print(f"⚠️ 关节 {name} 控制ID {ctrl_id} 超出范围（最大：{len(self.data.ctrl) - 1}）")
            except Exception as e:
                print(f"⚠️ 关节 {name} 控制失败：{e}")

    def collect_plot_data(self):
        """收集绘图数据（线程安全）"""
        self.frame_counter += 1
        if self.frame_counter % self.config.plot_update_interval != 0:
            return

        with data_lock:
            # 添加时间数据
            current_time = self.data.time
            self.time_data.append(current_time)

            # 添加各关节角度数据
            for name in self.joint_names:
                qpos_idx = self.joint_qpos_indices[name]
                if qpos_idx != -1 and 0 <= qpos_idx < len(self.data.qpos):
                    angle = self.data.qpos[qpos_idx]
                    self.plot_data[name].append(angle)
                else:
                    self.plot_data[name].append(0.0)

            # 限制数据点数量
            if len(self.time_data) > self.config.max_plot_points:
                self.time_data.pop(0)
                for name in self.joint_names:
                    if len(self.plot_data[name]) > 0:
                        self.plot_data[name].pop(0)

    def init_plot(self):
        """初始化绘图界面"""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlabel('Time (s)', fontsize=12)
        self.ax.set_ylabel('Joint Angle (rad)', fontsize=12)
        self.ax.set_title('Real-time Joint Angle Monitoring', fontsize=14, fontweight='bold')

        # 定义颜色方案
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

        # 创建线条对象
        for i, name in enumerate(self.joint_names):
            line, = self.ax.plot([], [], label=name, color=colors[i % len(colors)],
                                 linestyle=linestyles[i % len(linestyles)], linewidth=2)
            self.lines[name] = line

        self.ax.legend(loc='upper right', fontsize=10)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(-2, 2)

        plt.tight_layout()
        print("📊 关节角度可视化图表已创建！")

    def update_plot(self, frame):
        """更新绘图（动画回调函数）"""
        with data_lock:
            for name, line in self.lines.items():
                if len(self.plot_data[name]) > 0 and len(self.time_data) == len(self.plot_data[name]):
                    line.set_data(self.time_data, self.plot_data[name])

            if len(self.time_data) > 0:
                self.ax.set_xlim(max(0, self.time_data[-1] - 10), self.time_data[-1] + 1)

        return list(self.lines.values())

    def print_robot_state(self):
        """打印机器人状态"""
        current_time = self.data.time
        if not hasattr(self, "last_print_time"):
            self.last_print_time = 0.0
            self.frame_count = 0
            self.start_time = current_time

        self.frame_count += 1
        elapsed_time = current_time - self.start_time
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time

        if current_time - self.last_print_time >= self.config.state_print_interval:
            print(
                f"\n===== 机器人状态（时间：{current_time:.2f}s | 帧率：{self.fps:.1f} FPS | 模式：{self.current_mode}）=====")
            for name in self.joint_names:
                ctrl_id = self.joint_ctrl_ids[name]
                qpos_idx = self.joint_qpos_indices[name]

                if ctrl_id != -1 and qpos_idx != -1 and qpos_idx < len(self.data.qpos) and ctrl_id < len(
                        self.data.ctrl):
                    print(
                        f"关节 {name}: 位置 = {self.data.qpos[qpos_idx]:.2f} rad, 控制信号 = {self.data.ctrl[ctrl_id]:.2f}")
                elif ctrl_id == -1:
                    print(f"关节 {name}: 无控制ID")
                elif qpos_idx == -1:
                    print(f"关节 {name}: 无位置索引")
                else:
                    print(f"关节 {name}: 索引超出范围")

            self.last_print_time = current_time

    def reset_robot(self):
        """重置机器人到初始状态"""
        with data_lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[0:7] = [0, 0, 1.0, 1, 0, 0, 0]

            # 重置控制信号和动作状态
            for name in self.joint_names:
                self.last_ctrl_signals[name] = 0.0
                ctrl_id = self.joint_ctrl_ids[name]
                if ctrl_id != -1 and ctrl_id < len(self.data.ctrl):
                    self.data.ctrl[ctrl_id] = 0.0

            self.walk_phase = 0.0
            self.wave_arm = "right"

            # 清空绘图数据
            self.plot_data = {name: [] for name in self.joint_names}
            self.time_data = []
            self.frame_counter = 0

        print("\n🔄 机器人已重置到初始状态！")

    def check_user_input(self):
        """检查用户输入（Windows兼容）"""
        if sys.platform == 'win32':
            try:
                import msvcrt
                if msvcrt.kbhit():
                    user_input = sys.stdin.readline().strip().lower()
                    return user_input
            except:
                return None
        return None

    def process_user_input(self, user_input):
        """处理用户输入指令"""
        if not user_input:
            return

        if user_input == 'r':
            self.reset_robot()
        elif user_input in ["walk", "wave", "sin", "random", "stop"]:
            self.current_mode = user_input
            print(f"\n🔄 运动模式已切换为：{user_input}")
            if user_input == "walk":
                print("👣 行走模式：机器人将进行自然行走动作")
            elif user_input == "wave":
                print("✋ 挥手模式：机器人将交替挥动手臂")
        elif user_input == 'q':
            global sim_running
            sim_running = False
            print("\n📤 收到退出指令，仿真将结束...")
        elif user_input == 'clear':
            with data_lock:
                self.plot_data = {name: [] for name in self.joint_names}
                self.time_data = []
            print("\n🧹 绘图数据已清空！")
        elif user_input:
            print(f"\n❓ 未知指令：{user_input}，支持的指令：")
            print("  - r：重置机器人")
            print("  - walk：行走模式（新增）")
            print("  - wave：挥手模式（新增）")
            print("  - sin：正弦运动模式")
            print("  - random：随机运动模式")
            print("  - stop：停止运动")
            print("  - clear：清空绘图数据")
            print("  - q：退出仿真")

    def run_simulation(self):
        """运行仿真主循环"""
        self.load_model()

        # 初始化绘图
        self.init_plot()

        # 启动可视化动画
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=True, cache_frame_data=False)

        # 显示绘图窗口
        plt.show(block=False)

        # 启动MuJoCo可视化
        with viewer.launch_passive(self.model, self.data) as v:
            # 设置相机参数
            pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            if pelvis_id != -1:
                v.cam.trackbodyid = pelvis_id
            v.cam.distance = self.config.cam_distance
            v.cam.azimuth = self.config.cam_azimuth
            v.cam.elevation = self.config.cam_elevation

            # 打印操作提示
            print("\n📌 仿真操作提示：")
            print("  - 输入 'r' 回车：重置机器人")
            print("  - 输入 'walk' 回车：行走模式（新增）")
            print("  - 输入 'wave' 回车：挥手模式（新增）")
            print("  - 输入 'sin' 回车：正弦运动模式")
            print("  - 输入 'random' 回车：随机运动模式")
            print("  - 输入 'stop' 回车：停止运动")
            print("  - 输入 'clear' 回车：清空绘图数据")
            print("  - 输入 'q' 回车：退出仿真")
            print("  - 按 Ctrl+C：强制退出仿真")
            print("\n🚀 仿真开始（默认模式：行走）...")

            # 仿真主循环
            last_step_time = time.perf_counter()

            while sim_running and v.is_running():
                current_time = time.perf_counter()

                # 检查并处理用户输入
                user_input = self.check_user_input()
                if user_input:
                    self.process_user_input(user_input)

                if current_time - last_step_time >= self.config.timestep:
                    # 更新关节控制
                    self.update_joint_controls()

                    # 执行仿真步
                    try:
                        mujoco.mj_step(self.model, self.data)
                    except Exception as e:
                        print(f"\n⚠️ 仿真步执行失败：{e}")
                        self.reset_robot()

                    # 更新可视化
                    v.sync()

                    # 收集绘图数据
                    self.collect_plot_data()

                    # 打印状态
                    self.print_robot_state()

                    last_step_time = current_time

                # 处理matplotlib事件
                plt.pause(0.001)

        # 清理资源
        plt.close(self.fig)
        print("\n🏁 仿真结束！")


# ====================== 程序入口 ======================
if __name__ == "__main__":
    # 设置matplotlib后端
    import matplotlib

    matplotlib.use('TkAgg')

    # Windows控制台编码修复
    if sys.platform == 'win32':
        try:
            # 设置控制台编码为UTF-8
            import subprocess

            subprocess.call('chcp 65001', shell=True)
        except:
            pass

    # 初始化配置
    config = SimConfig()

    # 创建仿真器并运行
    simulator = HumanoidSimulator(config)

    try:
        simulator.run_simulation()
    except KeyboardInterrupt:
        sim_running = False
        print("\n⚠️ 用户中断，正在退出...")
    except Exception as e:
        print(f"\n❌ 程序异常：{e}")
        import traceback

        traceback.print_exc()
    finally:
        # 确保资源正确释放
        plt.close('all')
        sys.exit(0)
