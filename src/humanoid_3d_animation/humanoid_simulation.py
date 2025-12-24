import mujoco
import mujoco.viewer as viewer
import os
import time
import math
import threading
import signal
import sys
from dataclasses import dataclass  # 用于配置类

# ====================== 配置抽离（核心优化点）======================
@dataclass
class SimConfig:
    """仿真配置类：集中管理所有可配置参数"""
    # 文件路径配置（修改：XML文件在当前项目目录下）
    xml_filename: str = "humanoid.xml"
    # 仿真参数
    timestep: float = 0.005  # 与XML中的timestep保持一致
    sim_frequency: float = 2.0  # 关节运动频率（Hz）
    state_print_interval: float = 1.0  # 状态打印间隔（秒）
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
    # 控制模式：sin（正弦运动）、random（随机运动）、stop（静止）
    default_mode: str = "sin"

# 全局变量：用于优雅退出
sim_running = True

def signal_handler(sig, frame):
    """处理Ctrl+C中断信号，实现优雅退出"""
    global sim_running
    sim_running = False
    print("\n⚠️ 收到中断信号，正在退出仿真...")
    sys.exit(0)

# 注册信号处理
signal.signal(signal.SIGINT, signal_handler)

# ====================== 核心功能类 ======================
class HumanoidSimulator:
    def __init__(self, config: SimConfig):
        self.config = config
        self.model = None
        self.data = None
        self.joint_names = list(config.joint_amplitudes.keys())
        # 预存关节ID和控制ID（避免每次循环重复计算，性能优化）
        self.joint_ctrl_ids = {}
        self.joint_qpos_indices = {}
        # 运动模式和控制信号缓存（用于平滑控制）
        self.current_mode = config.default_mode
        self.last_ctrl_signals = {}  # 存储上一帧的控制信号

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
    <<damping joint="left_shoulder" damping="0.1"/>
    <motor name="right_shoulder_motor" joint="right_shoulder" ctrlrange="-1.57 1.57" gear="10"/>
    <<damping joint="right_shoulder" damping="0.1"/>
    <motor name="left_elbow_motor" joint="left_elbow" ctrlrange="-1.57 0" gear="10"/>
    <<damping joint="left_elbow" damping="0.1"/>
    <motor name="right_elbow_motor" joint="right_elbow" ctrlrange="-1.57 0" gear="10"/>
    <<damping joint="right_elbow" damping="0.1"/>
    <!-- 腿部关节 -->
    <motor name="left_hip_motor" joint="left_hip" ctrlrange="-1.57 1.57" gear="10"/>
    <<damping joint="left_hip" damping="0.1"/>
    <motor name="right_hip_motor" joint="right_hip" ctrlrange="-1.57 1.57" gear="10"/>
    <<damping joint="right_hip" damping="0.1"/>
    <motor name="left_knee_motor" joint="left_knee" ctrlrange="0 1.57" gear="10"/>
    <<damping joint="left_knee" damping="0.1"/>
    <motor name="right_knee_motor" joint="right_knee" ctrlrange="0 1.57" gear="10"/>
    <<damping joint="right_knee" damping="0.1"/>
  </actuator>
</mujoco>"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        print(f"✅ 已在 {file_path} 创建XML文件！")

    def load_model(self):
        """加载MuJoCo模型，预存关节ID和控制ID（性能优化）"""
        # 核心修改：获取当前项目目录（即脚本所在的文件夹路径）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, self.config.xml_filename)

        # 检查并创建文件
        if not os.path.exists(self.model_path):
            self.create_xml_file(self.model_path)
        else:
            print(f"ℹ️ XML文件已存在（路径：{self.model_path}），无需重新创建！")

        # 读取XML内容并加载模型
        try:
            with open(self.model_path, "r", encoding="utf-8") as f:
                xml_content = f.read()
            self.model = mujoco.MjModel.from_xml_string(xml_content)
            self.data = mujoco.MjData(self.model)
            print("✅ 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            sys.exit(1)

        # 预存关节控制ID和qpos索引（只计算一次，性能优化）
        for name in self.joint_names:
            # 获取控制ID
            ctrl_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor")
            if ctrl_id == -1:
                ctrl_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.joint_ctrl_ids[name] = ctrl_id

            # 获取qpos索引（根关节占前7个自由度）
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id != -1:
                self.joint_qpos_indices[name] = 7 + joint_id
            else:
                self.joint_qpos_indices[name] = -1

            # 初始化控制信号缓存
            self.last_ctrl_signals[name] = 0.0

    def get_joint_ctrl_signal(self, name, t):
        """根据运动模式生成关节控制信号（功能扩展：多模式）"""
        amplitude = self.config.joint_amplitudes[name]
        freq = self.config.sim_frequency

        if self.current_mode == "sin":
            # 正弦/余弦运动：左右关节反向
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
            # 随机运动：在幅度范围内随机变化
            signal = (math.sin(t * freq * 0.5) * 0.5 + 0.5) * amplitude * 2 - amplitude
        elif self.current_mode == "stop":
            # 静止：控制信号为0
            signal = 0.0
        else:
            signal = 0.0

        # 平滑过渡：避免控制信号突变（用户体验优化）
        smooth_factor = 0.1  # 平滑系数，越小越平滑
        self.last_ctrl_signals[name] = (1 - smooth_factor) * self.last_ctrl_signals[name] + smooth_factor * signal
        return self.last_ctrl_signals[name]

    def update_joint_controls(self):
        """更新关节控制信号（函数拆分：主循环更简洁）"""
        t = self.data.time
        for name in self.joint_names:
            ctrl_id = self.joint_ctrl_ids[name]
            if ctrl_id == -1:
                continue
            # 生成控制信号并设置
            ctrl_signal = self.get_joint_ctrl_signal(name, t)
            try:
                self.data.ctrl[ctrl_id] = ctrl_signal
            except Exception as e:
                print(f"⚠️ 关节 {name} 控制失败：{e}")

    def print_robot_state(self):
        """打印机器人状态（优化：控制打印频率，添加帧率显示）"""
        current_time = self.data.time
        if not hasattr(self, "last_print_time"):
            self.last_print_time = 0.0
            self.frame_count = 0
            self.start_time = current_time

        # 累计帧数，计算帧率
        self.frame_count += 1
        elapsed_time = current_time - self.start_time
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time

        # 按间隔打印
        if current_time - self.last_print_time >= self.config.state_print_interval:
            print(f"\n===== 机器人状态（时间：{current_time:.2f}s | 帧率：{self.fps:.1f} FPS）=====")
            for name in self.joint_names:
                ctrl_id = self.joint_ctrl_ids[name]
                qpos_idx = self.joint_qpos_indices[name]
                if ctrl_id != -1 and qpos_idx != -1 and qpos_idx < len(self.data.qpos):
                    print(f"关节 {name}: 位置 = {self.data.qpos[qpos_idx]:.2f} rad, 控制信号 = {self.data.ctrl[ctrl_id]:.2f}")
            self.last_print_time = current_time

    def reset_robot(self):
        """重置机器人到初始状态"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:7] = [0, 0, 1.0, 1, 0, 0, 0]
        # 重置控制信号缓存
        for name in self.joint_names:
            self.last_ctrl_signals[name] = 0.0
        print("\n🔄 机器人已重置到初始状态！")

    def input_listener(self):
        """后台线程：监听控制台输入，支持多指令（功能扩展）"""
        global sim_running
        while sim_running:
            try:
                user_input = input().strip().lower()
                if user_input == 'r':
                    self.reset_robot()
                elif user_input in ["sin", "random", "stop"]:
                    self.current_mode = user_input
                    print(f"\n🔄 运动模式已切换为：{user_input}")
                elif user_input == 'q':
                    sim_running = False
                    print("\n📤 收到退出指令，仿真将结束...")
                else:
                    print(f"\n❓ 未知指令：{user_input}，支持的指令：r（重置）、sin/random/stop（模式）、q（退出）")
            except EOFError:
                continue
            except Exception as e:
                print(f"\n⚠️ 输入处理失败：{e}")

    def run_simulation(self):
        """运行仿真主循环"""
        # 加载模型
        self.load_model()

        # 启动输入监听线程
        input_thread = threading.Thread(target=self.input_listener, daemon=True)
        input_thread.start()

        # 启动可视化
        with viewer.launch_passive(self.model, self.data) as v:
            # 设置相机参数（配置化）
            pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            if pelvis_id != -1:
                v.cam.trackbodyid = pelvis_id
            v.cam.distance = self.config.cam_distance
            v.cam.azimuth = self.config.cam_azimuth
            v.cam.elevation = self.config.cam_elevation

            # 打印操作提示（用户体验优化）
            print("\n📌 仿真操作提示：")
            print("  - 输入 'r' 回车：重置机器人")
            print("  - 输入 'sin'/'random'/'stop' 回车：切换运动模式")
            print("  - 输入 'q' 回车：退出仿真")
            print("  - 按 Ctrl+C：强制退出仿真")
            print("\n🚀 仿真开始...")

            # 仿真主循环（使用perf_counter优化时间控制）
            global sim_running
            last_step_time = time.perf_counter()
            while sim_running and v.is_running():
                # 控制仿真步长（更精准的时间控制）
                current_time = time.perf_counter()
                if current_time - last_step_time >= self.config.timestep:
                    # 更新关节控制
                    self.update_joint_controls()

                    # 执行仿真步（异常捕获，健壮性优化）
                    try:
                        mujoco.mj_step(self.model, self.data)
                    except Exception as e:
                        print(f"\n⚠️ 仿真步执行失败：{e}")
                        self.reset_robot()

                    # 更新可视化
                    v.sync()

                    # 打印状态
                    self.print_robot_state()

                    last_step_time = current_time

        print("\n🏁 仿真结束！")

# ====================== 程序入口 ======================
if __name__ == "__main__":
    # 初始化配置
    config = SimConfig()
    # 创建仿真器并运行
    simulator = HumanoidSimulator(config)
    simulator.run_simulation()