import os
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
        self.simulator_folder = simulator_folder
        self.render_mode = render_mode
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.last_reward = 0.0
        self.screen = None  # pygame屏幕对象
        
        # 创建模型保存目录
        os.makedirs(self.simulator_folder, exist_ok=True)
        
        # 根据MuJoCo是否可用选择初始化方式
        if HAS_MUJOCO:
            self._init_mujoco()
        else:
            logger.warning("MuJoCo not available — using placeholder simulator.")
            self._init_placeholder()

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