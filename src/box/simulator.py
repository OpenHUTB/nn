import os
 my-featurn-branch
import logging
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 全局依赖检测
try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    mujoco = None
    HAS_MUJOCO = False

try:
    import mujoco_viewer
    HAS_MUJOCO_VIEWER = True
except ImportError:
    mujoco_viewer = None
    HAS_MUJOCO_VIEWER = False


class Simulator:  # 第27行：类定义后必须有缩进的代码块
    """Mechanical hand simulator with optional MuJoCo backend.
    Behavior:
    - If `mujoco` is available, generate a simple hand MJCF, load model/data,
    and expose `reset`/`step` using MuJoCo stepping routines.
    - If `mujoco` is not available, fall back to a lightweight placeholder
    implementation so the module remains importable and testable.
    """
    # --------------- 核心：类内所有方法必须缩进（通常4个空格）---------------
    def __init__(self, render_mode: str = "human", simulator_folder: str = "./mujoco_models"):
        """初始化模拟器（类的构造方法，必须缩进）"""
        self.render_mode = render_mode
import numpy as np
import pygame
import mujoco
import yaml
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
import mujoco.viewer

class Simulator:
    def __init__(self, simulator_folder: str, render_mode: Optional[str] = None):
      main
        self.simulator_folder = simulator_folder
        self.render_mode = render_mode
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.last_reward = 0.0

    # ---------------- MuJoCo后端实现（必须缩进）----------------
    def _init_mujoco(self):
        """初始化MuJoCo仿真环境"""
        self.config: Dict[str, Any] = {
            "simulation": {
                "max_steps": 1000,
                "model_path": "hand_model.mjcf",
                "target_joint_pos": []
            }
        }

        # 5根手指的基座位置和连杆长度（单位：米）
        bases = [(-0.20, 0.06), (-0.08, 0.06), (0.04, 0.06), (0.16, 0.06), (0.28, 0.06)]
        lengths = [[0.06, 0.05, 0.04] for _ in bases]
        self.finger_bases = bases
        self.link_lengths = lengths

        # 生成MJCF模型文件路径
        model_path = os.path.join(self.simulator_folder, self.config["simulation"]["model_path"])
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 构建MJCF模型内容
        mjcf_lines: List[str] = [
            '<mujoco model="mechanical_hand">',
            '  <option timestep="0.005" solver="PGS" iterations="50" tolerance="1e-6"/>',
            '  <default>',
            '    <joint damping="0.5" armature="0.01" limited="true" range="-1 1"/>',
            '    <geom density="1000" friction="1 0.5 0.5"/>',
            '  </default>',
            '  <worldbody>',
            '    <light pos="0 0 1" dir="0 0 -1"/>',
            '    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.95 0.95 0.95 1"/>',
            '    <body name="palm" pos="0 0 0.02">',
            '      <geom type="box" size="0.18 0.08 0.02" rgba="0.6 0.4 0.3 1"/>'
        ]

        # 生成每根手指的MJCF描述
        for i, base in enumerate(bases):
            bx, by = base
            mjcf_lines.append(f'      <body name="finger{i}_base" pos="{bx} {by} 0.05">')
            cum_x = 0.0
            for j, L in enumerate(lengths[i]):
                joint_name = f'f{i}j{j}'
                mjcf_lines.append(f'        <joint name="{joint_name}" type="hinge" axis="0 0 1" pos="{cum_x} 0 0"/>')
                next_x = cum_x + L
                mjcf_lines.append(f'        <geom type="capsule" fromto="{cum_x} 0 0 {next_x} 0 0" size="0.01" rgba="0.85 0.85 0.85 1"/>')
                cum_x = next_x
            mjcf_lines.append('      </body>')

        # 完成MJCF模型定义
        mjcf_lines.extend(['    </body>', '  </worldbody>', '  <actuator>'])
        for i in range(len(bases)):
            for j in range(len(lengths[i])):
                joint_name = f'f{i}j{j}'
                mjcf_lines.append(f'    <motor name="m_{joint_name}" joint="{joint_name}" gear="20"/>')
        mjcf_lines.append('  </actuator>')
        mjcf_lines.append('</mujoco>')

        # 保存MJCF文件
        with open(model_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(mjcf_lines))

        # 加载MuJoCo模型和数据
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            logger.exception(f'Failed to load MuJoCo model — falling back to placeholder: {e}')
            self._init_placeholder()
            return

        # 定义动作/观测空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(int(self.model.nu),), dtype=np.float32
        )
        obs_dim = int(self.model.nq) + int(self.model.nv)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )

        # 初始化MuJoCo Viewer
        self.viewer = None
        if self.render_mode == 'human' and HAS_MUJOCO_VIEWER:
            try:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
                self.viewer.cam.azimuth = 45
                self.viewer.cam.elevation = -20
                self.viewer.cam.distance = 0.5
            except Exception as e:
                logger.exception(f'Failed to initialize mujoco_viewer: {e}')

    # ---------------- 占位实现（必须缩进）----------------
    def _init_placeholder(self):
        """轻量级占位实现"""
        self.model = type('M', (), {'nq': 15, 'nu': 15, 'nv': 15})()
        self.data = type('D', (), {
            'qpos': np.zeros(self.model.nq),
            'qvel': np.zeros(self.model.nv)
        })()

        # 定义兼容的动作/观测空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(int(self.model.nu),), dtype=np.float32
        )
        obs_dim = int(self.model.nq) + int(self.model.nv)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )

        # 占位的手指参数
        self.finger_bases = [(-0.20, 0.06), (-0.08, 0.06), (0.04, 0.06), (0.16, 0.06), (0.28, 0.06)]
        self.link_lengths = [[0.06, 0.05, 0.04] for _ in self.finger_bases]

    # ---------------- 通用API（必须缩进）----------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """重置仿真环境"""
        if seed is not None:
            np.random.seed(int(seed))
            if HAS_MUJOCO and hasattr(mujoco, 'set_rng_seed'):
                try:
                    mujoco.set_rng_seed(int(seed))
                except Exception as e:
                    logger.warning(f'Failed to set MuJoCo seed: {e}')

        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.last_reward = 0.0

        if HAS_MUJOCO and isinstance(self.data, mujoco.MjData):
            mujoco.mj_resetData(self.model, self.data)
        else:
            self.data.qpos = np.zeros(int(self.model.nq))
            self.data.qvel = np.zeros(int(self.model.nv))

        obs = self._get_obs()
        return obs, {'step': self.step_count}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步仿真"""
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # MuJoCo步进
        if HAS_MUJOCO and isinstance(self.data, mujoco.MjData):
            self.data.ctrl[:] = action * 5.0
            try:
                mujoco.mj_step(self.model, self.data)
            except Exception:
                try:
                    mujoco.mj_step1(self.model, self.data)
                    mujoco.mj_step2(self.model, self.data)
                except Exception as e:
                    logger.exception(f'MuJoCo stepping failed: {e}')
        else:
            # 占位动力学
            gain = 0.01
            self.data.qvel = self.data.qvel + action[:int(self.model.nv)] * gain
            self.data.qpos = self.data.qpos + self.data.qvel

        self.step_count += 1
        reward = self._compute_reward()
        self.last_reward = float(reward)

        self.terminated = self._check_terminated()
        max_steps = int(self.config['simulation'].get('max_steps', 1000)) if hasattr(self, 'config') else 1000
        self.truncated = self.step_count >= max_steps

        obs = self._get_obs()
        info = {'step': self.step_count, 'reward': float(reward)}
        return obs, float(reward), bool(self.terminated), bool(self.truncated), info

    def _get_obs(self) -> np.ndarray:
        """获取当前观测"""
        if HAS_MUJOCO and isinstance(self.data, mujoco.MjData):
            qpos = self.data.qpos.copy()
            qvel = self.data.qvel.copy()
        else:
            qpos = np.asarray(self.data.qpos).copy()
            qvel = np.asarray(self.data.qvel).copy()
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def _compute_reward(self) -> float:
        """计算奖励"""
        if hasattr(self, 'config') and 'simulation' in self.config:
            target = np.array(self.config['simulation'].get('target_joint_pos', [0.0] * int(self.model.nq)))
        else:
            target = np.zeros(int(self.model.nq))
        
        current = np.asarray(self.data.qpos).copy()
        if len(target) != len(current):
            target = np.zeros_like(current)
        
        return float(-np.sum((current - target) ** 2))

    def _check_terminated(self) -> bool:
        """检查是否终止"""
        return False

    def render(self):
        """渲染仿真画面"""
        if self.render_mode != 'human':
            return

        # MuJoCo Viewer渲染
        if HAS_MUJOCO and self.viewer is not None:
            try:
                self.viewer.render()
                return
            except Exception as e:
                logger.warning(f'MuJoCo Viewer render failed: {e}')

        # 占位渲染（Pygame）
        try:
            import pygame
        except ImportError:
            logger.warning("Pygame not installed — skip placeholder rendering")
            return

        # 初始化Pygame
        if not (pygame.get_init() and pygame.display.get_init()):
            try:
                pygame.init()
                self.screen = pygame.display.set_mode((800, 600))
                pygame.display.set_caption('Mechanical Hand (Placeholder)')
            except Exception as e:
                logger.warning(f'Pygame init failed: {e}')
                return

        # 绘制背景
        screen = self.screen
        screen.fill((240, 240, 255))

        # 绘制手掌
        center_x, center_y = 400, 420
        palm_w, palm_h = 360, 80
        pygame.draw.rect(screen, (160, 120, 90), 
                        (center_x - palm_w // 2, center_y - palm_h // 2, palm_w, palm_h))

        # 绘制手指连杆
        q = np.asarray(self.data.qpos).copy()
        ptr = 0
        for i, base in enumerate(self.finger_bases):
            bx, by = base
            sx = center_x + int(bx * 1000)
            sy = center_y - int(by * 1000)
            angle = 0.0
            x0, y0 = sx, sy

            for j, L in enumerate(self.link_lengths[i]):
                if ptr < len(q):
                    angle += float(q[ptr])
                ex = x0 + int(np.cos(angle) * (L * 1000))
                ey = y0 - int(np.sin(angle) * (L * 1000))
                
                pygame.draw.line(screen, (10, 10, 10), (x0, y0), (ex, ey), max(1, 6 - j))
                pygame.draw.circle(screen, (200, 80, 80), (x0, y0), 6)
                
                x0, y0 = ex, ey
                ptr += 1

        # 绘制状态文本
        font = pygame.font.Font(None, 28)
        txt = font.render(f"Step: {self.step_count}  Reward: {self.last_reward:.3f}", True, (0, 0, 0))
        screen.blit(txt, (10, 10))

        # 更新画面
        pygame.display.flip()

        # 处理Pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        """关闭模拟器"""
        # 关闭MuJoCo Viewer
        if HAS_MUJOCO and self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass

        # 关闭Pygame
        if self.render_mode == 'human':
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass


# 测试代码（类外代码无需缩进）
if __name__ == "__main__":
    # 初始化模拟器
    sim = Simulator(render_mode="human", simulator_folder="./mujoco_hand")
    
    # 重置环境
    obs, info = sim.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")

    # 运行100步仿真
    for _ in range(100):
        # 随机动作
        action = sim.action_space.sample()
        obs, reward, terminated, truncated, info = sim.step(action)
        
        # 渲染画面
        sim.render()
        
        # 打印状态
        if _ % 10 == 0:
            print(f"Step: {info['step']}, Reward: {reward:.3f}")
        
        if terminated or truncated:
            print(f"Simulation ended at step {info['step']}")
            break

    # 关闭模拟器
    sim.close()
=======
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
