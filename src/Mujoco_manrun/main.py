import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mujoco import viewer
from pathlib import Path
import time

# 解决字体警告
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


class HumanoidWalker:
    def __init__(self, model_path):
        # 强制确认路径是字符串（双重保险）
        if not isinstance(model_path, str):
            raise TypeError(f"模型路径必须是字符串，当前是 {type(model_path)} 类型")

        # 加载模型
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}\n请检查：1.路径是否为字符串 2.文件是否存在 3.文件是否完整")

        # 仿真参数
        self.sim_duration = 20.0
        self.dt = self.model.opt.timestep
        self.init_wait_time = 1.0

        # PID参数
        self.kp = 30.0
        self.ki = 0.01
        self.kd = 5.0

        # 关节状态记录
        self.joint_errors = np.zeros(self.model.nu)
        self.joint_integrals = np.zeros(self.model.nu)

        # 初始化姿势
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # 存储数据
        self.times = []
        self.root_pos = []

    def get_gait_trajectory(self, t):
        """生成步态轨迹"""
        if t < 2.0:
            return np.zeros(self.model.nu)

        t_adjusted = t - 2.0
        cycle = t_adjusted % 1.5
        phase = 2 * np.pi * cycle / 1.5

        leg_amp = 0.3
        arm_amp = 0.2
        torso_amp = 0.05

        target = np.zeros(self.model.nu)
        leg_joint_offset = 5
        if len(target) > leg_joint_offset + 6:
            target[leg_joint_offset] = -leg_amp * np.sin(phase)
            target[leg_joint_offset + 1] = leg_amp * 1.5 * np.sin(phase + np.pi)
            target[leg_joint_offset + 2] = leg_amp * 0.5 * np.sin(phase)
            target[leg_joint_offset + 3] = -leg_amp * np.sin(phase + np.pi)
            target[leg_joint_offset + 4] = leg_amp * 1.5 * np.sin(phase)
            target[leg_joint_offset + 5] = leg_amp * 0.5 * np.sin(phase + np.pi)

        if len(target) > 0:
            target[0] = torso_amp * np.sin(phase + np.pi / 2)
        if len(target) > 16:
            target[16] = arm_amp * np.sin(phase + np.pi)
        if len(target) > 20:
            target[20] = arm_amp * np.sin(phase)

        return target

    def pid_controller(self, target_pos):
        """PID控制器"""
        current_pos = self.data.qpos[7:]
        if len(current_pos) != len(target_pos):
            return np.zeros_like(target_pos)

        error = target_pos - current_pos
        self.joint_integrals += error * self.dt
        self.joint_integrals = np.clip(self.joint_integrals, -2.0, 2.0)

        derivative = (error - self.joint_errors) / self.dt if self.dt != 0 else 0
        self.joint_errors = error.copy()

        torque = self.kp * error + self.ki * self.joint_integrals + self.kd * derivative
        return np.clip(torque, -5.0, 5.0)

    def simulate_with_visualization(self):
        """带稳定启动的可视化仿真"""
        with viewer.launch_passive(self.model, self.data) as v:
            print("可视化窗口已启动（前2秒保持静止，随后开始步行）")
            print("操作：鼠标拖动旋转视角，滚轮缩放，W/A/S/D平移，关闭窗口结束")

            # 初始等待
            start_time = time.time()
            while time.time() - start_time < self.init_wait_time:
                v.sync()
                time.sleep(0.01)

            # 仿真主循环
            while self.data.time < self.sim_duration:
                target = self.get_gait_trajectory(self.data.time)
                self.data.ctrl[:] = self.pid_controller(target)
                mujoco.mj_step(self.model, self.data)
                v.sync()
                time.sleep(0.001)

                self.times.append(self.data.time)
                self.root_pos.append(self.data.qpos[:3].copy())

        return {
            "time": np.array(self.times),
            "root_pos": np.array(self.root_pos)
        }

    def plot_results(self, data):
        """绘制结果"""
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.plot(data["time"], data["root_pos"][:, 0], label="Forward Distance (X)")
        plt.ylabel("Position (m)")
        plt.xlabel("Time (s)")
        plt.title("Walking Trajectory")
        plt.grid(alpha=0.3)
        plt.legend()

        plt.subplot(122)
        plt.plot(data["time"], data["root_pos"][:, 2], label="Torso Height (Z)", color='orange')
        plt.xlabel("Time (s)")
        plt.ylabel("Height (m)")
        plt.title("Torso Height During Walking")
        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 1. 手动拼接字符串路径（完全避开 Path 对象传递问题）
    # 原理：直接用字符串表示路径，不依赖 Path 对象的转换
    import os

    # 获取当前脚本所在目录的字符串路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接 XML 文件的字符串路径
    model_path = os.path.join(current_dir, "humanoid.xml")

    # 2. 打印路径信息（方便排查）
    print(f"当前脚本目录：{current_dir}")
    print(f"模型文件路径：{model_path}")
    print(f"路径类型：{type(model_path)}")  # 应显示 <class 'str'>

    # 3. 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型文件不存在！\n查找路径：{model_path}\n"
            f"请确认 humanoid.xml 放在以下目录：{current_dir}"
        )

    # 4. 运行仿真
    try:
        walker = HumanoidWalker(model_path)
        print("开始仿真...")
        results = walker.simulate_with_visualization()
        print("仿真完成！")
        walker.plot_results(results)
    except Exception as e:
        print(f"仿真出错：{e}")
