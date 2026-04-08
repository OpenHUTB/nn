import os
import logging
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from gymnasium import spaces

# 初始化日志配置，设置日志级别为INFO
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------- 全局依赖检测 ----------------
# 检测MuJoCo是否安装（核心物理仿真引擎）
try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    mujoco = None
    HAS_MUJOCO = False

# 检测MuJoCo Viewer是否安装（原生可视化工具）
try:
    import mujoco_viewer
    HAS_MUJOCO_VIEWER = True
except ImportError:
    mujoco_viewer = None
    HAS_MUJOCO_VIEWER = False


class Simulator:
    """
    机械臂/机械手仿真器核心类（兼容Gymnasium接口）
    核心特性：
    1. 支持MuJoCo物理引擎后端，自动生成机械臂/手MJCF模型文件
    2. 无MuJoCo时自动降级为轻量级占位实现（保证模块可导入）
    3. 提供标准RL接口：reset/step/render/close
    4. 支持Pygame可视化（2D占位渲染）和MuJoCo Viewer（3D原生渲染）
    """
    def __init__(self, simulator_folder: str, render_mode: Optional[str] = None):
        """
        仿真器初始化入口
        Args:
            simulator_folder: 仿真配置/模型文件存储路径
            render_mode: 渲染模式 - "human"（可视化窗口）/None（无渲染）
        """
        # 基础属性初始化
        self.simulator_folder = simulator_folder  # 仿真文件根目录
        self.render_mode = render_mode            # 渲染模式
        self.step_count = 0                       # 当前仿真步数
        self.terminated = False                   # 任务是否完成（自然终止）
        self.truncated = False                    # 任务是否截断（步数超限）
        self.last_reward = 0.0                    # 上一步奖励值
        
        # 核心组件初始化（先置空，后续分步加载）
        self.model = None       # MuJoCo模型对象（物理模型定义）
        self.data = None        # MuJoCo数据对象（存储仿真状态：关节位置/速度等）
        self.viewer = None      # MuJoCo Viewer对象（3D渲染）
        self.screen = None      # Pygame窗口对象（2D渲染）

        # 初始化流程（分层解耦，便于维护）
        # 1. 加载/生成配置文件（config.yaml）
        self.config = self._load_config()
        # 2. 加载/生成MuJoCo模型（MJCF文件）
        self.model, self.data = self._load_model()
        # 3. 校验并补全配置（保证关键参数存在）
        self._validate_config()

        # 初始化RL核心空间和渲染
        self._init_action_space()   # 动作空间（执行器控制信号）
        self._init_observation_space()  # 观测空间（关节状态）
        if self.render_mode:
            self._init_render()     # 初始化渲染组件

    @classmethod
    def get(cls, simulator_folder: str, **kwargs):
        """
        类工厂方法（兼容原有接口）
        Args:
            simulator_folder: 仿真文件根目录
            **kwargs: 其他初始化参数（如render_mode）
        Returns:
            Simulator: 仿真器实例
        """
        return cls(simulator_folder, **kwargs)

    def _load_config(self) -> Dict[str, Any]:
        """
        加载/生成仿真配置文件（config.yaml）
        - 若配置文件不存在，生成默认机械臂配置
        - 若存在，读取并返回配置字典
        Returns:
            Dict[str, Any]: 仿真配置字典
        """
        # 配置文件路径
        config_path = os.path.join(self.simulator_folder, "config.yaml")
        
        # 配置文件不存在时，生成默认配置
        if not os.path.exists(config_path):
            default_config = {
                "simulation": {
                    "max_steps": 1000,          # 单回合最大步数
                    "model_path": "arm_model.mjcf",  # MuJoCo模型文件路径
                    "control_frequency": 20,    # 控制频率（Hz）
                    "target_joint_pos": [0.0]   # 目标关节位置（奖励函数用）
                }
            }
            # 保存默认配置到文件
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(default_config, f)
            logger.info(f"生成默认配置文件: {config_path}")
        
        # 读取配置文件
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config

    def _load_model(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """
        加载/生成MuJoCo模型文件（MJCF）
        - 若模型文件不存在，生成简单机械臂模型
        - 加载模型并创建MjData对象（存储仿真状态）
        Returns:
            Tuple[mujoco.MjModel, mujoco.MjData]: MuJoCo模型和数据对象
        """
        # 模型文件路径（从配置读取）
        model_path = os.path.join(
            self.simulator_folder, 
            self.config["simulation"].get("model_path", "arm_model.mjcf")
        )
        
        # 模型文件不存在时，生成简单单关节机械臂MJCF
        if not os.path.exists(model_path):
            mjcf_content = """<mujoco model="simple_arm">
  <option timestep="0.01"/>  <!-- 仿真步长（10ms） -->
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>  <!-- 环境光 -->
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>  <!-- 地面 -->
    <body name="arm_base" pos="0 0 0.1">  <!-- 机械臂基座 -->
      <joint name="arm_joint" type="hinge" axis="0 1 0"/>  <!-- 铰链关节（绕Y轴旋转） -->
      <geom name="arm_link" type="capsule" fromto="0 0 0 0.5 0 0" size="0.05" rgba="0.5 0.5 0.5 1"/>  <!-- 机械臂连杆 -->
    </body>
  </worldbody>
  <actuator>
    <motor name="arm_motor" joint="arm_joint" gear="100"/>  <!-- 关节电机（传动比100） -->
  </actuator>
</mujoco>"""
            # 保存MJCF文件
            with open(model_path, "w", encoding="utf-8") as f:
                f.write(mjcf_content)
            logger.info(f"生成默认机械臂模型: {model_path}")
        
        # 加载MuJoCo模型和数据
        try:
            model = mujoco.MjModel.from_xml_path(model_path)
            data = mujoco.MjData(model)
            logger.info(f"成功加载MuJoCo模型: {model_path}")
            return model, data
        except Exception as e:
            logger.error(f"加载MuJoCo模型失败: {e}")
            raise

    def _validate_config(self):
        """
        校验并补全配置参数
        - 保证simulation字段存在
        - 根据模型自动补全目标关节位置维度
        - 设置缺失参数的默认值
        """
        # 确保simulation字段存在
        if "simulation" not in self.config:
            self.config["simulation"] = {}
        
        # 获取模型关节数（nq），无模型时默认1
        nq = self.model.nq if self.model is not None else 1
        
        # 补全目标关节位置（维度匹配模型）
        self.config["simulation"].setdefault("target_joint_pos", [0.0] * nq)
        # 补全最大步数
        self.config["simulation"].setdefault("max_steps", 1000)
        # 补全控制频率
        self.config["simulation"].setdefault("control_frequency", 20)
        
        logger.info("配置校验完成，补全缺失参数")

    def _init_action_space(self):
        """
        初始化动作空间（Gymnasium Box空间）
        - 动作维度 = MuJoCo执行器数（nu）
        - 动作范围 [-1.0, 1.0]（标准化，便于RL训练）
        Raises:
            ValueError: 模型未加载时抛出异常
        """
        # 校验模型是否加载
        if self.model is None:
            raise ValueError("模型未加载，无法初始化动作空间")
        
        # 执行器数量（每个执行器对应一个动作维度）
        n_actuators = self.model.nu
        # 定义动作空间（Box空间，float32类型）
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_actuators,),
            dtype=np.float32
        )
        logger.info(f"动作空间初始化完成: 维度={n_actuators}, 范围=[-1.0, 1.0]")

    def _init_observation_space(self):
        """
        初始化观测空间（Gymnasium Box空间）
        - 观测维度 = 关节位置数（nq） + 关节速度数（nv）
        - 观测范围 [-∞, +∞]（无界，兼容关节任意状态）
        Raises:
            ValueError: 模型未加载时抛出异常
        """
        # 校验模型是否加载
        if self.model is None:
            raise ValueError("模型未加载，无法初始化观测空间")
        
        # 关节位置数和速度数
        n_qpos = self.model.nq  # 关节位置维度
        n_qvel = self.model.nv  # 关节速度维度
        obs_dim = n_qpos + n_qvel  # 总观测维度
        
        # 定义观测空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        logger.info(f"观测空间初始化完成: 总维度={obs_dim} (位置={n_qpos}, 速度={n_qvel})")

    def _init_render(self):
        """
        初始化渲染组件
        - 仅在render_mode="human"时初始化
        - 使用Pygame创建2D可视化窗口
        """
        if self.render_mode == "human":
            try:
                import pygame
                pygame.init()
                # 创建800x600的可视化窗口
                self.screen = pygame.display.set_mode((800, 600))
                pygame.display.set_caption("MuJoCo Arm Simulation")
                logger.info("Pygame渲染窗口初始化成功")
            except Exception as e:
                logger.error(f"Pygame渲染初始化失败: {e}")
                raise

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        重置仿真环境（新回合开始）
        Args:
            seed: 随机种子（保证复现性）
        Returns:
            Tuple[np.ndarray, dict]: 初始观测 + 信息字典
        """
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"设置随机种子: {seed}")
            
        # 重置MuJoCo仿真状态（关节位置/速度恢复初始值）
        mujoco.mj_resetData(self.model, self.data)
        
        # 重置计数器和状态标记
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.last_reward = 0.0
        
        # 获取初始观测
        obs = self._get_obs()
        
        # 渲染初始状态
        if self.render_mode == "human":
            self.render()
        
        logger.debug(f"仿真环境重置完成，初始观测维度: {obs.shape}")
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一步仿真（RL核心循环单元）
        Args:
            action: 动作数组（来自RL策略，范围[-1.0, 1.0]）
        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: 
                观测 + 奖励 + 终止标记 + 截断标记 + 信息字典
        """
        # 动作预处理（确保类型和范围合法）
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 将标准化动作映射到执行器控制信号（放大10倍，匹配电机范围）
        self.data.ctrl[:] = action * 10.0
        
        # 执行MuJoCo仿真步（物理引擎核心计算）
        mujoco.mj_step(self.model, self.data)
        
        # 更新步数计数器
        self.step_count += 1
        
        # 计算奖励（关节位置与目标的误差平方和的负值）
        reward = self._compute_reward()
        self.last_reward = reward
        
        # 检查是否自然终止（如关节角度超限）
        self.terminated = self._check_terminated()
        
        # 检查是否截断（达到最大步数）
        max_steps = self.config["simulation"].get("max_steps", 1000)
        self.truncated = self.step_count >= max_steps
        
        # 获取当前观测
        obs = self._get_obs()
        
        # 渲染当前状态
        if self.render_mode == "human":
            self.render()
        
        # 日志输出（每10步打印一次）
        if self.step_count % 10 == 0:
            logger.debug(f"Step {self.step_count}: Reward={reward:.2f}, Terminated={self.terminated}, Truncated={self.truncated}")
        
        return obs, reward, self.terminated, self.truncated, {}

    def _get_obs(self) -> np.ndarray:
        """
        获取当前观测（关节位置 + 关节速度）
        Returns:
            np.ndarray: 观测数组（float32类型）
        """
        # 复制关节位置和速度（避免直接修改MuJoCo内部数据）
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # 拼接为观测数组
        obs = np.concatenate([qpos, qvel])
        return obs.astype(np.float32)

    def _compute_reward(self) -> float:
        """
        计算奖励函数（目标：让关节位置接近目标值）
        - 奖励 = -Σ(当前位置 - 目标位置)²
        - 误差越小，奖励越高（最大值0）
        Returns:
            float: 即时奖励值
        """
        # 获取目标关节位置和当前位置
        target_pos = np.array(self.config["simulation"]["target_joint_pos"])
        current_pos = self.data.qpos.copy()
        
        # 确保维度匹配（避免广播错误）
        if len(target_pos) != len(current_pos):
            target_pos = np.zeros_like(current_pos)
        
        # 计算位置误差的平方和（L2损失）
        pos_error = np.sum((current_pos - target_pos) ** 2)
        # 奖励为负的误差（鼓励误差减小）
        reward = -pos_error
        
        return float(reward)

    def _check_terminated(self) -> bool:
        """
        检查是否达到自然终止条件
        - 终止条件：任意关节角度超过π弧度（180度）
        Returns:
            bool: True=终止，False=继续
        """
        joint_pos = self.data.qpos.copy()
        
        # 检查是否有关节角度超限
        if np.any(np.abs(joint_pos) > np.pi):
            logger.info(f"关节角度超限（>π），仿真终止 | 当前关节位置: {joint_pos}")
            return True
            
        return False

    def render(self):
        """
        渲染当前仿真状态（2D Pygame可视化）
        - 绘制机械臂2D投影
        - 显示步数、奖励、关节角度等信息
        - 处理窗口事件（关闭/ESC退出）
        """
        if self.render_mode == "human" and self.screen is not None:
            import pygame
            
            # 处理Pygame窗口事件
            for event in pygame.event.get():
                # 关闭窗口事件
                if event.type == pygame.QUIT:
                    self.close()
                    return
                # ESC键退出
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.close()
                        return
            
            # 清屏（白色背景）
            self.screen.fill((255, 255, 255))
            
            # 绘制机械臂2D投影
            # 获取关节角度（单关节机械臂）
            joint_angle = self.data.qpos[0] if self.model.nq > 0 else 0
            # 机械臂长度（像素）
            arm_length = 200
            # 窗口中心坐标
            center_x, center_y = 400, 300
            # 计算机械臂末端坐标（极坐标转笛卡尔）
            end_x = center_x + arm_length * np.cos(joint_angle)
            end_y = center_y + arm_length * np.sin(joint_angle)
            
            # 绘制机械臂连杆（黑色线条，宽度5）
            pygame.draw.line(
                self.screen, (0, 0, 0), 
                (center_x, center_y), (end_x, end_y), 5
            )
            # 绘制关节（红色圆心，半径10）
            pygame.draw.circle(
                self.screen, (255, 0, 0), 
                (int(center_x), int(center_y)), 10
            )
            # 绘制末端执行器（蓝色圆心，半径8）
            pygame.draw.circle(
                self.screen, (0, 0, 255), 
                (int(end_x), int(end_y)), 8
            )
            
            # 绘制文本信息（步数、奖励、关节角度）
            font = pygame.font.Font(None, 36)
            # 步数文本
            step_text = font.render(f"Step: {self.step_count}", True, (0, 0, 0))
            # 奖励文本
            reward_text = font.render(f"Reward: {self.last_reward:.2f}", True, (0, 0, 0))
            # 关节角度文本
            angle_text = font.render(f"Joint Angle: {joint_angle:.2f} rad", True, (0, 0, 0))
            
            # 绘制文本到窗口
            self.screen.blit(step_text, (10, 10))
            self.screen.blit(reward_text, (10, 50))
            self.screen.blit(angle_text, (10, 90))
            
            # 更新窗口显示
            pygame.display.flip()
            
            # 控制渲染帧率（匹配控制频率）
            control_freq = self.config["simulation"].get("control_frequency", 20)
            pygame.time.delay(int(1000 / control_freq))

    def close(self):
        """
        关闭仿真环境（清理资源）
        - 关闭Pygame窗口
        - 释放所有渲染资源
        """
        if self.render_mode == "human":
            try:
                import pygame
                pygame.quit()
                logger.info("Pygame窗口已关闭")
            except Exception as e:
                logger.warning(f"关闭Pygame失败: {e}")


# ---------------- 测试代码（独立运行时执行）----------------
if __name__ == "__main__":
    # 初始化模拟器（存储路径：./mujoco_arm，开启可视化）
    sim = Simulator(render_mode="human", simulator_folder="./mujoco_arm")
    
    # 重置环境（设置随机种子保证复现）
    obs, info = sim.reset(seed=42)
    print(f"初始观测形状: {obs.shape}")

    # 运行100步仿真
    for _ in range(100):
        # 随机采样动作（从动作空间中）
        action = sim.action_space.sample()
        # 执行一步仿真
        obs, reward, terminated, truncated, info = sim.step(action)
        
        # 每10步打印状态
        if _ % 10 == 0:
            print(f"Step: {sim.step_count}, Reward: {reward:.3f}")
        
        # 终止/截断时退出循环
        if terminated or truncated:
            print(f"仿真在第 {sim.step_count} 步结束")
            break

    # 关闭模拟器（清理资源）
    sim.close()