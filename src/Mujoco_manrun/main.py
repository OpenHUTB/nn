import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mujoco import viewer
from pathlib import Path
import time

# 彻底解决字体警告：使用matplotlib默认字体，避免指定中文字体（若无需中文可关闭）
# 若需要中文显示，可先运行下方注释的代码查看系统可用字体并替换
plt.rcParams["font.family"] = ["sans-serif"]  # 使用系统默认无衬线字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# 查看系统可用字体（可选：运行一次找到可用的中文字体名称替换）
# import matplotlib.font_manager
# fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
# for font in sorted(fonts):
#     print(font)

class HumanoidWalker:
    def __init__(self, model_path):
        # 加载模型（添加异常处理）
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}，请检查模型文件是否正确")

        # 仿真参数（延长初始等待时间，避免窗口闪退）
        self.sim_duration = 20.0  # 仿真时长
        self.dt = self.model.opt.timestep
        self.init_wait_time = 1.0  # 初始等待1秒，确保窗口稳定打开

        # PID参数（降低增益，避免初始力矩过大导致失衡）
        self.kp = 30.0  # 降低比例增益，减少震荡
        self.ki = 0.01
        self.kd = 5.0

        # 关节状态记录
        self.joint_errors = np.zeros(self.model.nu)
        self.joint_integrals = np.zeros(self.model.nu)

        # 初始化姿势（确保从稳定状态开始）
        mujoco.mj_resetData(self.model, self.data)  # 重置到零状态
        mujoco.mj_forward(self.model, self.data)  # 计算初始状态

        # 存储数据
        self.times = []
        self.root_pos = []

    def get_gait_trajectory(self, t):
        """生成步态轨迹（兼容不同模型关节索引的容错设计）"""
        # 延迟启动步态（前2秒保持站立，避免初始动作过大）
        if t < 2.0:
            return np.zeros(self.model.nu)

        t_adjusted = t - 2.0  # 从第2秒开始计算步态
        cycle = t_adjusted % 1.5  # 更长的周期，更稳定
        phase = 2 * np.pi * cycle / 1.5

        # 关节目标角度（使用比例系数，适应不同模型关节范围）
        leg_amp = 0.3  # 腿部摆动幅度（缩小幅度，提高稳定性）
        arm_amp = 0.2  # 手臂摆动幅度
        torso_amp = 0.05  # 躯干摆动幅度

        # 腿部轨迹（通用化设计，不依赖固定索引）
        target = np.zeros(self.model.nu)
        # 假设前6个自由度是根节点（不控制），从第7个开始是关节
        # 若模型关节索引不同，只需调整以下索引偏移
        leg_joint_offset = 5  # 腿部关节起始索引（根据模型调整）
        if len(target) > leg_joint_offset + 6:  # 确保索引不越界
            # 右腿（髋关节、膝关节、踝关节）
            target[leg_joint_offset] = -leg_amp * np.sin(phase)
            target[leg_joint_offset + 1] = leg_amp * 1.5 * np.sin(phase + np.pi)
            target[leg_joint_offset + 2] = leg_amp * 0.5 * np.sin(phase)
            # 左腿（与右腿反相）
            target[leg_joint_offset + 3] = -leg_amp * np.sin(phase + np.pi)
            target[leg_joint_offset + 4] = leg_amp * 1.5 * np.sin(phase)
            target[leg_joint_offset + 5] = leg_amp * 0.5 * np.sin(phase + np.pi)

        # 上身和手臂（容错设计）
        if len(target) > 0:
            target[0] = torso_amp * np.sin(phase + np.pi / 2)  # 躯干
        if len(target) > 16:
            target[16] = arm_amp * np.sin(phase + np.pi)  # 右臂
        if len(target) > 20:
            target[20] = arm_amp * np.sin(phase)  # 左臂

        return target

    def pid_controller(self, target_pos):
        """PID控制器（增加安全限制）"""
        current_pos = self.data.qpos[7:]  # 跳过根节点6自由度
        if len(current_pos) != len(target_pos):
            # 关节数量不匹配时，返回零力矩避免崩溃
            return np.zeros_like(target_pos)

        error = target_pos - current_pos
        self.joint_integrals += error * self.dt
        self.joint_integrals = np.clip(self.joint_integrals, -2.0, 2.0)  # 更严格的积分限制

        derivative = (error - self.joint_errors) / self.dt if self.dt != 0 else 0
        self.joint_errors = error.copy()

        torque = self.kp * error + self.ki * self.joint_integrals + self.kd * derivative
        # 力矩限制在安全范围（进一步缩小，避免失控）
        return np.clip(torque, -5.0, 5.0)

    def simulate_with_visualization(self):
        """带稳定启动的可视化仿真"""
        with viewer.launch_passive(self.model, self.data) as v:
            print("可视化窗口已启动（前2秒保持静止，随后开始步行）")
            print("操作：鼠标拖动旋转视角，滚轮缩放，W/A/S/D平移，关闭窗口结束")

            # 初始等待，确保窗口稳定打开
            start_time = time.time()
            while time.time() - start_time < self.init_wait_time:
                v.sync()
                time.sleep(0.01)

            # 仿真主循环
            while self.data.time < self.sim_duration:
                # 生成目标轨迹
                target = self.get_gait_trajectory(self.data.time)
                # 计算控制力矩
                self.data.ctrl[:] = self.pid_controller(target)
                # 单步仿真
                mujoco.mj_step(self.model, self.data)
                # 刷新窗口（增加延迟，避免窗口刷新过快）
                v.sync()
                time.sleep(0.001)

                # 记录数据
                self.times.append(self.data.time)
                self.root_pos.append(self.data.qpos[:3].copy())

        return {
            "time": np.array(self.times),
            "root_pos": np.array(self.root_pos)
        }

    def plot_results(self, data):
        """绘制结果（使用英文标签避免字体问题）"""
        plt.figure(figsize=(12, 6))
        # 前进距离（X轴）
        plt.subplot(121)
        plt.plot(data["time"], data["root_pos"][:, 0], label="Forward Distance (X)")
        plt.ylabel("Position (m)")
        plt.xlabel("Time (s)")
        plt.title("Walking Trajectory")
        plt.grid(alpha=0.3)
        plt.legend()

        # 躯干高度（Z轴）
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
    model_path = "D:\\Github\\file\\nn\\src\\Mujoco_manrun\\humanoid.xml" #应为自己的humanoid.xml的位置

    # 检查模型文件
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在：{model_path}，请检查路径")

    # 运行仿真
    try:
        walker = HumanoidWalker(model_path)
        print("开始仿真...")
        results = walker.simulate_with_visualization()
        print("仿真完成！")
        walker.plot_results(results)
    except Exception as e:
        print(f"仿真出错：{e}")