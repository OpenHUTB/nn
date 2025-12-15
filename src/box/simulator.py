import os
import numpy as np
import pygame
import mujoco
import yaml
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
import mujoco.viewer

class Simulator:
    def __init__(self, simulator_folder: str, render_mode: Optional[str] = None):
        self.simulator_folder = simulator_folder
        self.render_mode = render_mode
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.last_reward = 0.0
        self.model = None
        self.data = None
        self.viewer = None
        self.screen = None

        # 1. 先加载配置
        self.config = self._load_config()
        # 2. 再加载模型
        self.model, self.data = self._load_model()
        # 3. 最后校验配置
        self._validate_config()

        # 初始化动作空间、观测空间、渲染
        self._init_action_space()
        self._init_observation_space()
        
        if self.render_mode:
            self._init_render()

    @classmethod
    def get(cls, simulator_folder: str, **kwargs):
        return cls(simulator_folder, **kwargs)

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(self.simulator_folder, "config.yaml")
        if not os.path.exists(config_path):
            default_config = {
                "simulation": {
                    "max_steps": 1000, 
                    "model_path": "arm_model.mjcf",
                    "control_frequency": 20,
                    "target_joint_pos": [0.0]
                }
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(default_config, f)
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_model(self):
        model_path = os.path.join(self.simulator_folder, self.config["simulation"].get("model_path", "arm_model.mjcf"))
        if not os.path.exists(model_path):
            with open(model_path, "w", encoding="utf-8") as f:
                f.write("""<mujoco model="simple_arm">
  <option timestep="0.01"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="arm_base" pos="0 0 0.1">
      <joint name="arm_joint" type="hinge" axis="0 1 0"/>
      <geom name="arm_link" type="capsule" fromto="0 0 0 0.5 0 0" size="0.05" rgba="0.5 0.5 0.5 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="arm_motor" joint="arm_joint" gear="100"/>
  </actuator>
</mujoco>""")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        return model, data

    def _validate_config(self):
        if "simulation" not in self.config:
            self.config["simulation"] = {}
        
        # 如果有模型，使用模型信息，否则使用默认值
        if self.model is not None:
            nq = self.model.nq
        else:
            nq = 1  # 默认值
            
        self.config["simulation"].setdefault("target_joint_pos", [0.0] * nq)
        self.config["simulation"].setdefault("max_steps", 1000)
        self.config["simulation"].setdefault("control_frequency", 20)

    def _init_action_space(self):
        """初始化动作空间"""
        # 确保模型已加载
        if self.model is None:
            raise ValueError("模型未加载，无法初始化动作空间")
        
        n_actuators = self.model.nu
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_actuators,),
            dtype=np.float32
        )
        print(f"动作空间初始化: 维度={n_actuators}")

    def _init_observation_space(self):
        """初始化观测空间"""
        # 确保模型已加载
        if self.model is None:
            raise ValueError("模型未加载，无法初始化观测空间")
        
        n_qpos = self.model.nq
        n_qvel = self.model.nv
        
        obs_dim = n_qpos + n_qvel
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        print(f"观测空间初始化: 维度={obs_dim} (位置={n_qpos}, 速度={n_qvel})")

    def _init_render(self):
        """初始化渲染"""
        if self.render_mode == "human":
            try:
                pygame.init()
                self.screen = pygame.display.set_mode((800, 600))
                pygame.display.set_caption("MuJoCo Arm Simulation")
                print("Pygame渲染初始化成功")
            except Exception as e:
                print(f"渲染初始化失败: {e}")

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """重置仿真环境"""
        if seed is not None:
            np.random.seed(seed)
            
        # 重置MuJoCo数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 重置计数器
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.last_reward = 0.0
        
        # 获取初始观测
        obs = self._get_obs()
        
        # 渲染初始状态
        if self.render_mode == "human":
            self.render()
        
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步仿真"""
        # 将动作应用到执行器
        self.data.ctrl[:] = action * 10.0
        
        # 执行一步仿真
        mujoco.mj_step(self.model, self.data)
        
        # 更新计数器
        self.step_count += 1
        
        # 计算奖励
        reward = self._compute_reward()
        self.last_reward = reward
        
        # 检查是否终止
        self.terminated = self._check_terminated()
        
        # 检查是否截断（达到最大步数）
        max_steps = self.config["simulation"].get("max_steps", 1000)
        self.truncated = self.step_count >= max_steps
        
        # 获取观测
        obs = self._get_obs()
        
        # 渲染当前状态
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, self.terminated, self.truncated, {}

    def _get_obs(self) -> np.ndarray:
        """获取当前观测"""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        obs = np.concatenate([qpos, qvel])
        return obs.astype(np.float32)

    def _compute_reward(self) -> float:
        """计算奖励函数"""
        target_pos = np.array(self.config["simulation"]["target_joint_pos"])
        current_pos = self.data.qpos.copy()
        
        # 确保数组维度匹配
        if len(target_pos) != len(current_pos):
            target_pos = np.zeros_like(current_pos)
        
        pos_error = np.sum((current_pos - target_pos) ** 2)
        reward = -pos_error
        
        return float(reward)

    def _check_terminated(self) -> bool:
        """检查是否达到终止条件"""
        joint_pos = self.data.qpos.copy()
        
        if np.any(np.abs(joint_pos) > np.pi):
            return True
            
        return False

    def render(self):
        """渲染当前仿真状态"""
        if self.render_mode == "human" and self.screen is not None:
            # 处理pygame事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.close()
                        return
            
            # 清屏
            self.screen.fill((255, 255, 255))
            
            # 绘制简单表示（2D投影）
            joint_angle = self.data.qpos[0] if self.model.nq > 0 else 0
            
            arm_length = 200
            center_x, center_y = 400, 300
            
            end_x = center_x + arm_length * np.cos(joint_angle)
            end_y = center_y + arm_length * np.sin(joint_angle)
            
            # 绘制机械臂
            pygame.draw.line(self.screen, (0, 0, 0), 
                           (center_x, center_y), (end_x, end_y), 5)
            
            # 绘制关节
            pygame.draw.circle(self.screen, (255, 0, 0), 
                             (int(center_x), int(center_y)), 10)
            pygame.draw.circle(self.screen, (0, 0, 255), 
                             (int(end_x), int(end_y)), 8)
            
            # 显示信息
            font = pygame.font.Font(None, 36)
            step_text = font.render(f"Step: {self.step_count}", True, (0, 0, 0))
            reward_text = font.render(f"Reward: {self.last_reward:.2f}", True, (0, 0, 0))
            angle_text = font.render(f"Joint Angle: {joint_angle:.2f} rad", True, (0, 0, 0))
            
            self.screen.blit(step_text, (10, 10))
            self.screen.blit(reward_text, (10, 50))
            self.screen.blit(angle_text, (10, 90))
            
            # 更新显示
            pygame.display.flip()
            
            # 控制帧率
            control_freq = self.config["simulation"].get("control_frequency", 20)
            pygame.time.delay(int(1000 / control_freq))

    def close(self):
        """关闭环境"""
        if self.render_mode == "human":
            pygame.quit()