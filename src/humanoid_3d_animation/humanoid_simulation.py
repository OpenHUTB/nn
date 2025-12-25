import mujoco
import mujoco.viewer as viewer
import os
import time
import math
import threading
import signal
import sys
import select
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ====================== 配置抽离 ======================
@dataclass
class SimConfig:
    """仿真配置类：集中管理所有可配置参数"""
    # 文件路径配置
    xml_filename: str = "humanoid.xml"
    # 仿真参数
    timestep: float = 0.005
    sim_frequency: float = 2.0
    state_print_interval: float = 1.0
    # 相机参数
    cam_distance: float = 2.0
    cam_azimuth: float = 45.0
    cam_elevation: float = -20.0
    # 关节运动幅度配置
    joint_amplitudes = {
        "left_shoulder": 1.0, "right_shoulder": 1.0,
        "left_elbow": 0.5, "right_elbow": 0.5,
        "left_hip": 0.8, "right_hip": 0.8,
        "left_knee": 0.6, "right_knee": 0.6
    }
    # 控制模式
    default_mode: str = "sin"
    # 可视化配置
    plot_update_interval: int = 50  # 绘图更新间隔（帧数）
    max_plot_points: int = 200  # 图表最大显示数据点


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
        self.input_thread_running = False

        # 新增：可视化相关变量
        self.plot_data = {name: [] for name in self.joint_names}
        self.time_data = []
        self.frame_counter = 0

        # 绘图相关
        self.fig, self.ax = None, None
        self.lines = {}
        self.ani = None

    def create_xml_file(self, file_path):
        """创建人形机器人XML文件"""
        xml_content = f"""<mujoco model="simple_humanoid">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{self.config.timestep}" gravity="0 0 -9.81"/>
  <visual>
    <global azimuth="135" elevation="-30" perspective="0.01"/>
  </visual>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1" specular="0.1 0.1 0.1"/>
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.8 0.8 0.8 1"/>
    <body name="pelvis" pos="0 0 1.0">
      <joint name="root" type="free"/>
      <geom name="pelvis_geom" type="capsule" size="0.1" fromto="0 0 0 0 0 0.2" rgba="0.5 0.5 0.9 1"/>
      <body name="torso" pos="0 0 0.2">
        <geom name="torso_geom" type="capsule" size="0.1" fromto="0 0 0 0 0 0.3" rgba="0.5 0.5 0.9 1"/>
        <body name="head" pos="0 0 0.3">
          <geom name="head_geom" type="sphere" size="0.15" pos="0 0 0" rgba="0.8 0.5 0.5 1"/>
        </body>
        <!-- 左手臂 -->
        <body name="left_arm" pos="0.15 0 0.15">
          <joint name="left_shoulder" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
          <geom name="left_upper_arm" type="capsule" size="0.05" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1"/>
          <body name="left_forearm" pos="0 0 0.2">
            <joint name="left_elbow" type="hinge" axis="1 0 0" range="-1.57 0"/>
            <geom name="left_forearm_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1"/>
          </body>
        </body>
        <!-- 右手臂 -->
        <body name="right_arm" pos="-0.15 0 0.15">
          <joint name="right_shoulder" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
          <geom name="right_upper_arm" type="capsule" size="0.05" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1"/>
          <body name="right_forearm" pos="0 0 0.2">
            <joint name="right_elbow" type="hinge" axis="1 0 0" range="-1.57 0"/>
            <geom name="right_forearm_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1"/>
          </body>
        </body>
        <!-- 左腿部 -->
        <body name="left_leg" pos="0.05 0 -0.2">
          <joint name="left_hip" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
          <geom name="left_thigh" type="capsule" size="0.06" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.5 1"/>
          <body name="left_calf" pos="0 0 -0.3">
            <joint name="left_knee" type="hinge" axis="1 0 0" range="0 1.57"/>
            <geom name="left_calf_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.5 1"/>
          </body>
        </body>
        <!-- 右腿部 -->
        <body name="right_leg" pos="-0.05 0 -0.2">
          <joint name="right_hip" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
          <geom name="right_thigh" type="capsule" size="0.06" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.5 1"/>
          <body name="right_calf" pos="0 0 -0.3">
            <joint name="right_knee" type="hinge" axis="1 0 0" range="0 1.57"/>
            <geom name="right_calf_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.5 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <!-- 手臂关节 -->
    <motor name="left_shoulder_motor" joint="left_shoulder" ctrlrange="-1.57 1.57" gear="10"/>
    <damping joint="left_shoulder" damping="0.1"/>
    <motor name="right_shoulder_motor" joint="right_shoulder" ctrlrange="-1.57 1.57" gear="10"/>
    <damping joint="right_shoulder" damping="0.1"/>
    <motor name="left_elbow_motor" joint="left_elbow" ctrlrange="-1.57 0" gear="10"/>
    <damping joint="left_elbow" damping="0.1"/>
    <motor name="right_elbow_motor" joint="right_elbow" ctrlrange="-1.57 0" gear="10"/>
    <damping joint="right_elbow" damping="0.1"/>
    <!-- 腿部关节 -->
    <motor name="left_hip_motor" joint="left_hip" ctrlrange="-1.57 1.57" gear="10"/>
    <damping joint="left_hip" damping="0.1"/>
    <motor name="right_hip_motor" joint="right_hip" ctrlrange="-1.57 1.57" gear="10"/>
    <damping joint="right_hip" damping="0.1"/>
    <motor name="left_knee_motor" joint="left_knee" ctrlrange="0 1.57" gear="10"/>
    <damping joint="left_knee" damping="0.1"/>
    <motor name="right_knee_motor" joint="right_knee" ctrlrange="0 1.57" gear="10"/>
    <damping joint="right_knee" damping="0.1"/>
  </actuator>
</mujoco>"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        print(f"✅ 已在 {file_path} 创建XML文件！")

    def load_model(self):
        """加载MuJoCo模型"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, self.config.xml_filename)

        if not os.path.exists(self.model_path):
            self.create_xml_file(self.model_path)
        else:
            print(f"ℹ️ XML文件已存在（路径：{self.model_path}），无需重新创建！")

        try:
            with open(self.model_path, "r", encoding="utf-8") as f:
                xml_content = f.read()
            self.model = mujoco.MjModel.from_xml_string(xml_content)
            self.data = mujoco.MjData(self.model)
            print("✅ 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            sys.exit(1)

        # 预存关节ID
        for name in self.joint_names:
            ctrl_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor")
            if ctrl_id == -1:
                ctrl_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.joint_ctrl_ids[name] = ctrl_id

            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id != -1:
                self.joint_qpos_indices[name] = 7 + joint_id
            else:
                self.joint_qpos_indices[name] = -1

            self.last_ctrl_signals[name] = 0.0

    def get_joint_ctrl_signal(self, name, t):
        """生成关节控制信号"""
        amplitude = self.config.joint_amplitudes[name]
        freq = self.config.sim_frequency

        if self.current_mode == "sin":
            if "left" in name or "hip" in name or "knee" in name:
                if "shoulder" in name or "elbow" in name:
                    signal = math.sin(t * freq) * amplitude
                else:
                    signal = math.cos(t * freq) * amplitude
            else:
                if "shoulder" in name or "elbow" in name:
                    signal = -math.sin(t * freq) * amplitude
                else:
                    signal = -math.cos(t * freq) * amplitude
        elif self.current_mode == "random":
            signal = (math.sin(t * freq * 0.5) * 0.5 + 0.5) * amplitude * 2 - amplitude
        elif self.current_mode == "stop":
            signal = 0.0
        else:
            signal = 0.0

        # 平滑过渡
        smooth_factor = 0.1
        self.last_ctrl_signals[name] = (1 - smooth_factor) * self.last_ctrl_signals[name] + smooth_factor * signal
        return self.last_ctrl_signals[name]

    def update_joint_controls(self):
        """更新关节控制信号"""
        t = self.data.time
        for name in self.joint_names:
            ctrl_id = self.joint_ctrl_ids[name]
            if ctrl_id == -1:
                continue
            ctrl_signal = self.get_joint_ctrl_signal(name, t)
            try:
                self.data.ctrl[ctrl_id] = ctrl_signal
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
                if qpos_idx != -1 and qpos_idx < len(self.data.qpos):
                    angle = self.data.qpos[qpos_idx]
                    self.plot_data[name].append(angle)

            # 限制数据点数量，避免内存占用过大
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

        # 设置y轴范围
        self.ax.set_ylim(-2, 2)

        plt.tight_layout()
        print("📊 关节角度可视化图表已创建！")

    def update_plot(self, frame):
        """更新绘图（动画回调函数）"""
        with data_lock:
            # 更新每条线的数据
            for name, line in self.lines.items():
                if len(self.plot_data[name]) > 0 and len(self.time_data) == len(self.plot_data[name]):
                    line.set_data(self.time_data, self.plot_data[name])

            # 自动调整x轴范围
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
            print(f"\n===== 机器人状态（时间：{current_time:.2f}s | 帧率：{self.fps:.1f} FPS）=====")
            for name in self.joint_names:
                ctrl_id = self.joint_ctrl_ids[name]
                qpos_idx = self.joint_qpos_indices[name]
                if ctrl_id != -1 and qpos_idx != -1 and qpos_idx < len(self.data.qpos):
                    print(
                        f"关节 {name}: 位置 = {self.data.qpos[qpos_idx]:.2f} rad, 控制信号 = {self.data.ctrl[ctrl_id]:.2f}")
            self.last_print_time = current_time

    def reset_robot(self):
        """重置机器人到初始状态"""
        with data_lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[0:7] = [0, 0, 1.0, 1, 0, 0, 0]
            # 重置控制信号缓存
            for name in self.joint_names:
                self.last_ctrl_signals[name] = 0.0
            # 清空绘图数据
            self.plot_data = {name: [] for name in self.joint_names}
            self.time_data = []
            self.frame_counter = 0
        print("\n🔄 机器人已重置到初始状态！")

    def input_listener(self):
        """后台线程：监听控制台输入"""
        global sim_running
        self.input_thread_running = True
        timeout = 0.1

        while self.input_thread_running and sim_running:
            try:
                ready, _, _ = select.select([sys.stdin], [], [], timeout)
                if ready:
                    user_input = sys.stdin.readline().strip().lower()
                    if user_input == 'r':
                        self.reset_robot()
                    elif user_input in ["sin", "random", "stop"]:
                        self.current_mode = user_input
                        print(f"\n🔄 运动模式已切换为：{user_input}")
                    elif user_input == 'q':
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
                        print("  - sin/random/stop：切换运动模式")
                        print("  - clear：清空绘图数据")
                        print("  - q：退出仿真")
            except Exception as e:
                print(f"\n⚠️ 输入处理失败：{e}")
                break

        print("\n🔌 输入监听线程已优雅退出")

    def run_simulation(self):
        """运行仿真主循环"""
        self.load_model()

        # 初始化绘图
        self.init_plot()

        # 启动输入监听线程
        input_thread = threading.Thread(target=self.input_listener)
        input_thread.start()

        # 启动可视化动画
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=True, cache_frame_data=False)

        # 显示绘图窗口（非阻塞）
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
            print("  - 输入 'sin'/'random'/'stop' 回车：切换运动模式")
            print("  - 输入 'clear' 回车：清空绘图数据")
            print("  - 输入 'q' 回车：退出仿真")
            print("  - 按 Ctrl+C：强制退出仿真")
            print("\n🚀 仿真开始...")

            # 仿真主循环
            global sim_running
            last_step_time = time.perf_counter()

            while sim_running and v.is_running():
                current_time = time.perf_counter()
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

        # 停止输入监听线程
        self.input_thread_running = False
        input_thread.join(timeout=1.0)

        # 关闭绘图窗口
        plt.close(self.fig)

        print("\n🏁 仿真结束！")


# ====================== 程序入口 ======================
if __name__ == "__main__":
    # 设置matplotlib后端（避免显示问题）
    import matplotlib

    matplotlib.use('TkAgg')

    # 初始化配置
    config = SimConfig()

    # 创建仿真器并运行
    simulator = HumanoidSimulator(config)
    simulator.run_simulation()

    sys.exit(0)