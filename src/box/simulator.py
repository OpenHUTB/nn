import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
from gymnasium import spaces

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_SIMULATION_CONFIG: Dict[str, Any] = {
    "simulation": {
        "max_steps": 1000,
        "model_path": "arm_model.mjcf",
        "control_frequency": 20,
        "target_joint_pos": [0.0],
    }
}

try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:  # pragma: no cover - runtime dependency
    mujoco = None
    HAS_MUJOCO = False

try:
    import mujoco_viewer  # noqa: F401

    HAS_MUJOCO_VIEWER = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_MUJOCO_VIEWER = False


class Simulator:
    """Simple MuJoCo arm simulator with Gymnasium-like APIs."""

    def __init__(self, simulator_folder: str, render_mode: Optional[str] = None):
        if not HAS_MUJOCO:
            raise ImportError("mujoco is required. Install dependencies with: pip install -r requirements.txt")

        self.simulator_folder = simulator_folder
        os.makedirs(self.simulator_folder, exist_ok=True)

        self.render_mode = render_mode
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.last_reward = 0.0

        self.model = None
        self.data = None
        self.viewer = None
        self.screen = None

        self.config = self._load_config()
        self.model, self.data = self._load_model()
        self._validate_config()

        self._init_action_space()
        self._init_observation_space()

        if self.render_mode:
            self._init_render()

    @classmethod
    def get(cls, simulator_folder: str, **kwargs):
        return cls(simulator_folder, **kwargs)

    def _load_config(self) -> Dict[str, Any]:
        """Load config.yaml and keep it aligned to the expected schema."""
        config_path = os.path.join(self.simulator_folder, "config.yaml")

        if not os.path.exists(config_path):
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(DEFAULT_SIMULATION_CONFIG, f, sort_keys=False)
            logger.info("Generated default config file: %s", config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}

        if not isinstance(loaded, dict):
            loaded = {}

        config: Dict[str, Any] = {"simulation": {}}
        config["simulation"].update(DEFAULT_SIMULATION_CONFIG["simulation"])
        config["simulation"].update(loaded.get("simulation", {}))

        if config.get("simulation") != loaded.get("simulation", {}):
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)
            logger.info("Normalized and updated config file: %s", config_path)

        logger.info("Loaded config file: %s", config_path)
        return config

    def _load_model(self) -> Tuple["mujoco.MjModel", "mujoco.MjData"]:
        model_path = os.path.join(self.simulator_folder, self.config["simulation"].get("model_path", "arm_model.mjcf"))

        if not os.path.exists(model_path):
            mjcf_content = """<mujoco model=\"simple_arm\">
  <option timestep=\"0.01\"/>
  <worldbody>
    <light pos=\"0 0 3\" dir=\"0 0 -1\"/>
    <geom name=\"floor\" type=\"plane\" size=\"5 5 0.1\" rgba=\"0.9 0.9 0.9 1\"/>
    <body name=\"arm_base\" pos=\"0 0 0.1\">
      <joint name=\"arm_joint\" type=\"hinge\" axis=\"0 1 0\"/>
      <geom name=\"arm_link\" type=\"capsule\" fromto=\"0 0 0 0.5 0 0\" size=\"0.05\" rgba=\"0.5 0.5 0.5 1\"/>
    </body>
  </worldbody>
  <actuator>
    <motor name=\"arm_motor\" joint=\"arm_joint\" gear=\"100\"/>
  </actuator>
</mujoco>"""
            with open(model_path, "w", encoding="utf-8") as f:
                f.write(mjcf_content)
            logger.info("Generated default arm model: %s", model_path)

        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        logger.info("Loaded MuJoCo model: %s", model_path)
        return model, data

    def _validate_config(self):
        if "simulation" not in self.config:
            self.config["simulation"] = {}

        nq = self.model.nq if self.model is not None else 1
        self.config["simulation"].setdefault("target_joint_pos", [0.0] * nq)
        self.config["simulation"].setdefault("max_steps", 1000)
        self.config["simulation"].setdefault("control_frequency", 20)

        logger.info("Configuration validation completed.")

    def _init_action_space(self):
        if self.model is None:
            raise ValueError("Model is not loaded; cannot initialize action space.")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

    def _init_observation_space(self):
        if self.model is None:
            raise ValueError("Model is not loaded; cannot initialize observation space.")

        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _init_render(self):
        if self.render_mode != "human":
            return

        import pygame

        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("MuJoCo Arm Simulation")

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.last_reward = 0.0

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Keep the original control mapping behavior.
        self.data.ctrl[:] = action * 10.0

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        reward = self._compute_reward()
        self.last_reward = reward

        self.terminated = self._check_terminated()
        self.truncated = self.step_count >= self.config["simulation"].get("max_steps", 1000)

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return obs, reward, self.terminated, self.truncated, {}

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def _compute_reward(self) -> float:
        target_pos = np.asarray(self.config["simulation"].get("target_joint_pos", []), dtype=np.float32)
        current_pos = self.data.qpos.copy()

        if len(target_pos) != len(current_pos):
            target_pos = np.zeros_like(current_pos)

        pos_error = np.sum((current_pos - target_pos) ** 2)
        return float(-pos_error)

    def _check_terminated(self) -> bool:
        return bool(np.any(np.abs(self.data.qpos.copy()) > np.pi))

    def render(self):
        if self.render_mode != "human" or self.screen is None:
            return

        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                return

        self.screen.fill((255, 255, 255))

        joint_angle = self.data.qpos[0] if self.model.nq > 0 else 0.0
        arm_length = 200
        center_x, center_y = 400, 300
        end_x = center_x + arm_length * np.cos(joint_angle)
        end_y = center_y + arm_length * np.sin(joint_angle)

        pygame.draw.line(self.screen, (0, 0, 0), (center_x, center_y), (end_x, end_y), 5)
        pygame.draw.circle(self.screen, (255, 0, 0), (int(center_x), int(center_y)), 10)
        pygame.draw.circle(self.screen, (0, 0, 255), (int(end_x), int(end_y)), 8)

        font = pygame.font.Font(None, 30)
        self.screen.blit(font.render(f"Step: {self.step_count}", True, (0, 0, 0)), (10, 10))
        self.screen.blit(font.render(f"Reward: {self.last_reward:.2f}", True, (0, 0, 0)), (10, 40))
        self.screen.blit(font.render(f"Joint Angle: {joint_angle:.2f} rad", True, (0, 0, 0)), (10, 70))

        pygame.display.flip()
        control_freq = self.config["simulation"].get("control_frequency", 20)
        pygame.time.delay(int(1000 / control_freq))

    def close(self):
        if self.render_mode == "human":
            try:
                import pygame

                pygame.quit()
            except Exception:
                pass


if __name__ == "__main__":
    sim = Simulator(render_mode="human", simulator_folder="./mujoco_arm")
    obs, _ = sim.reset(seed=42)
    print(f"initial obs shape: {obs.shape}")

    for _ in range(100):
        action = sim.action_space.sample()
        obs, reward, terminated, truncated, _ = sim.step(action)
        if _ % 10 == 0:
            print(f"Step: {sim.step_count}, Reward: {reward:.3f}")
        if terminated or truncated:
            print(f"Simulation ended at step {sim.step_count}")
            break

    sim.close()