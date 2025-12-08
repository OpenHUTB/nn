# 导入必要的库
import mujoco  # 导入MuJoCo物理仿真引擎核心库
import numpy as np  # 导入NumPy库，用于高效的数值计算，特别是数组操作
from mujoco import viewer  # 从MuJoCo导入viewer模块，用于创建交互式可视化窗口
import time  # 导入time模块，用于控制仿真速度和等待时间


# 人形机器人行走控制器类
class HumanoidWalker:
    """
    一个封装了人形机器人行走控制逻辑的类。
    它负责加载模型、生成步态、计算控制力矩、运行仿真和可视化。
    """

    def __init__(self, model_path):
        """
        类的初始化方法，在创建类的实例时自动调用。

        :param model_path: 字符串类型，指向机器人模型XML文件的路径。
        """
        # 强制确认传入的路径是字符串类型，避免因路径类型错误导致后续加载失败
        if not isinstance(model_path, str):
            # 如果不是字符串，抛出类型错误异常
            raise TypeError(f"模型路径必须是字符串，当前是 {type(model_path)} 类型")

        # 尝试从指定路径加载并编译MuJoCo模型
        try:
            # 从XML文件路径加载模型，返回一个MjModel对象，包含了机器人的所有物理和几何信息
            self.model = mujoco.MjModel.from_xml_path(model_path)
            # 创建一个与模型关联的MjData对象，用于存储仿真过程中的所有动态状态（如位置、速度、力等）
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            # 如果加载过程中出现任何错误，抛出运行时错误，并给出排查建议
            raise RuntimeError(f"模型加载失败：{e}\n请检查：1.路径是否为字符串 2.文件是否存在 3.文件是否完整")

        # 设置仿真相关的参数
        self.sim_duration = 20.0  # 设定仿真的总时长，单位为秒
        self.dt = self.model.opt.timestep  # 获取模型中定义的仿真步长，单位为秒。这是物理世界更新一次的时间
        self.init_wait_time = 1.0  # 设定仿真开始前的等待时间，单位为秒，用于让可视化窗口稳定显示初始状态

        # 设置PID控制器的参数
        self.kp = 30.0  # 比例增益 (Proportional Gain)，用于快速响应当前的位置误差
        self.ki = 0.01  # 积分增益 (Integral Gain)，用于消除长期存在的静差（稳态误差）
        self.kd = 5.0  # 微分增益 (Derivative Gain)，用于抑制系统震荡，提高稳定性

        # 初始化PID控制器所需的状态变量，用于存储历史误差信息
        # self.model.nu 是机器人可驱动关节的数量
        self.joint_errors = np.zeros(self.model.nu)  # 存储上一次的关节位置误差
        self.joint_integrals = np.zeros(self.model.nu)  # 存储关节位置误差的积分值

        # 将模型的所有数据重置到其初始状态（如初始位置、速度为零等）
        mujoco.mj_resetData(self.model, self.data)
        # 执行一次前向动力学计算，根据当前状态更新所有派生量（如世界坐标系下的位置、 Jacobian 等）
        mujoco.mj_forward(self.model, self.data)

    def get_gait_trajectory(self, t):
        """
        根据当前仿真时间t，生成一个目标步态轨迹（即每个关节的期望角度）。
        这是一个基于正弦函数的简单步态生成器。

        :param t: 当前的仿真时间，单位为秒。
        :return: 一个NumPy数组，包含了每个可驱动关节的目标角度（弧度）。
        """
        # 在仿真开始的前2秒，让机器人保持初始的零角度姿势，不进行任何动作
        if t < 2.0:
            return np.zeros(self.model.nu)  # 返回一个全为零的数组作为目标

        # 从第2秒开始，调整时间基准，使得步态周期从0开始计算
        t_adjusted = t - 2.0
        cycle = t_adjusted % 1.5  # 设定步态周期为1.5秒，计算当前时间处于哪个周期内
        phase = 2 * np.pi * cycle / 1.5  # 将周期转换为相位（0到2π之间），用于正弦函数

        # 定义步态中各个关节的摆动幅度（最大角度）
        leg_amp = 0.3  # 腿部关节的摆动幅度
        arm_amp = 0.2  # 手臂关节的摆动幅度
        torso_amp = 0.05  # 躯干关节的摆动幅度

        # 初始化一个全为零的目标关节角度数组
        target = np.zeros(self.model.nu)

        # 假设模型中腿部关节从索引5开始（这取决于你的XML模型定义）
        leg_joint_offset = 5
        # 检查目标数组长度是否足够容纳腿部关节
        if len(target) > leg_joint_offset + 5:
            # 为左右腿的髋、膝、踝关节设置目标角度
            # 通过不同相位的正弦函数组合，模拟出腿部的摆动和蹬地动作
            target[leg_joint_offset] = -leg_amp * np.sin(phase)
            target[leg_joint_offset + 1] = leg_amp * 1.5 * np.sin(phase + np.pi)
            target[leg_joint_offset + 2] = leg_amp * 0.5 * np.sin(phase)
            target[leg_joint_offset + 3] = -leg_amp * np.sin(phase + np.pi)
            target[leg_joint_offset + 4] = leg_amp * 1.5 * np.sin(phase)
            target[leg_joint_offset + 5] = leg_amp * 0.5 * np.sin(phase + np.pi)

        # 为躯干和手臂关节设置目标角度，以协调行走姿态，保持平衡
        if len(target) > 0:
            target[0] = torso_amp * np.sin(phase + np.pi / 2)
        if len(target) > 16:
            target[16] = arm_amp * np.sin(phase + np.pi)
        if len(target) > 20:
            target[20] = arm_amp * np.sin(phase)

        # 返回计算好的目标关节角度数组
        return target

    def pid_controller(self, target_pos):
        """
        PID控制器的核心计算函数。根据当前关节位置和目标位置，计算出每个关节需要施加的力矩。

        :param target_pos: 目标关节位置数组（弧度）。
        :return: 计算出的关节控制力矩数组（牛顿·米）。
        """
        # 从仿真数据中获取当前关节的实际位置。
        # self.data.qpos 包含了所有自由度的位置（7个根节点自由度 + 关节自由度）
        # [7:] 切片操作，跳过前7个根节点自由度，获取所有关节的位置
        current_pos = self.data.qpos[7:]

        # 安全检查：确保目标位置数组和当前位置数组长度一致
        if len(current_pos) != len(target_pos):
            # 如果不一致，返回一个全为零的力矩数组，防止程序崩溃
            return np.zeros_like(target_pos)

        # 1. 计算比例项 (P)
        # error 是一个数组，每个元素代表对应关节的目标位置与当前位置的差值
        error = target_pos - current_pos

        # 2. 计算积分项 (I)
        # 将当前误差乘以时间步dt，累加到积分误差数组上
        self.joint_integrals += error * self.dt
        # 对积分值进行限幅（clipping），防止积分饱和导致控制器失控
        self.joint_integrals = np.clip(self.joint_integrals, -2.0, 2.0)

        # 3. 计算微分项 (D)
        # 微分项是误差的变化率，即 (当前误差 - 上次误差) / 时间步
        derivative = (error - self.joint_errors) / self.dt if self.dt != 0 else 0

        # 更新上次误差记录，为下一次计算微分项做准备
        self.joint_errors = error.copy()

        # 4. 计算PID总输出力矩
        torque = self.kp * error + self.ki * self.joint_integrals + self.kd * derivative

        # 对计算出的力矩进行限幅，防止超过电机或关节的最大承受能力
        return np.clip(torque, -5.0, 5.0)

    def simulate_with_visualization(self):
        """
        启动带可视化的仿真主循环。这是整个程序的核心执行部分。
        """
        # 使用viewer.launch_passive创建一个被动模式的可视化窗口。
        # "被动"意味着我们需要手动调用v.sync()来更新画面，这给了我们对仿真循环的完全控制。
        # 这是一个上下文管理器（with语句），确保窗口能被正确关闭。
        with viewer.launch_passive(self.model, self.data) as v:
            print("可视化窗口已启动（前2秒保持静止，随后开始步行）")
            print("操作：鼠标拖动旋转视角，滚轮缩放，W/A/S/D平移，关闭窗口结束")

            # 初始等待阶段
            start_time = time.time()  # 记录开始等待的时间
            # 循环等待，直到等待时间超过设定的self.init_wait_time
            while time.time() - start_time < self.init_wait_time:
                v.sync()  # 同步可视化窗口，更新一帧画面
                time.sleep(0.01)  # 短暂休眠，降低CPU占用

            # 仿真主循环
            # 持续运行，直到仿真时间达到设定的总时长self.sim_duration
            while self.data.time < self.sim_duration:
                # 1. 规划：根据当前仿真时间获取目标步态
                target_joint_positions = self.get_gait_trajectory(self.data.time)

                # 2. 控制：根据目标位置和当前位置，使用PID控制器计算控制力矩
                control_torques = self.pid_controller(target_joint_positions)

                # 3. 执行：将计算出的控制力矩赋值给机器人的执行器
                self.data.ctrl[:] = control_torques

                # 4. 步进：执行一步物理仿真。MuJoCo会根据当前状态和控制力矩计算下一个状态
                mujoco.mj_step(self.model, self.data)

                # 5. 可视化：同步可视化窗口，将新的仿真状态绘制出来
                v.sync()

                # 6. 速度控制：短暂休眠，以控制仿真的实时速度，使其不至于过快
                time.sleep(0.001)

        # 当仿真循环结束后（窗口关闭或时间到期），打印提示信息
        print("仿真完成！")


if __name__ == "__main__":
    # 这部分代码只有在当前脚本作为主程序直接运行时才会执行

    # 1. 构建模型文件的完整路径
    import os  # 导入os模块，用于处理文件路径

    # os.path.abspath(__file__) 获取当前脚本的绝对路径
    # os.path.dirname(...) 获取该路径的目录部分
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # os.path.join(...) 安全地拼接目录和文件名，跨平台兼容
    model_file_path = os.path.join(current_directory, "humanoid.xml")

    # 2. 打印路径信息，方便用户排查路径问题
    print(f"当前脚本所在目录：{current_directory}")
    print(f"模型文件完整路径：{model_file_path}")

    # 3. 检查模型文件是否存在
    if not os.path.exists(model_file_path):
        # 如果文件不存在，抛出文件未找到异常
        raise FileNotFoundError(
            f"模型文件不存在！\n查找路径：{model_file_path}\n"
            f"请确认 'humanoid.xml' 文件放在以下目录中：{current_directory}"
        )

    # 4. 实例化控制器并启动仿真
    try:
        # 创建HumanoidWalker类的实例，并传入模型文件路径
        walker = HumanoidWalker(model_file_path)
        print("\n开始仿真...")
        # 调用实例的simulate_with_visualization方法，启动仿真
        walker.simulate_with_visualization()
    except Exception as e:
        # 如果在整个过程中发生任何未捕获的异常，打印错误信息
        print(f"\n仿真过程中发生错误：{e}")