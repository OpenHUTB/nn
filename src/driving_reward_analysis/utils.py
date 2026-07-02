"""工具函数：配置加载、数据生成、文件保存."""

import os
import yaml
import numpy as np


def load_config(config_path: str = None) -> dict:
    """加载 YAML 配置文件.

    Args:
        config_path: 配置文件路径，默认为同目录下的 config.yaml

    Returns:
        配置字典
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def generate_scenario_data(
    num_steps: int = 200,
    dt: float = 0.1,
    seed: int = 42,
) -> dict:
    """生成仿真场景数据.

    模拟车辆在高速公路上行驶的场景，包含速度、距离和加速度数据.

    Args:
        num_steps: 时间步数
        dt: 时间步长 (s)
        seed: 随机种子

    Returns:
        包含 velocity, distance, acceleration 的字典
    """
    rng = np.random.default_rng(seed)
    t = np.arange(num_steps) * dt

    # 模拟速度变化：基准速度 + 随机波动 + 周期性变化
    velocity = 25.0 + 5.0 * np.sin(0.1 * t) + rng.normal(0, 2.0, num_steps)

    # 模拟前车距离变化
    distance = 20.0 - 8.0 * np.sin(0.08 * t) + rng.normal(0, 3.0, num_steps)
    distance = np.clip(distance, 1.0, 40.0)

    # 加速度通过速度差分计算
    acceleration = np.gradient(velocity, dt)

    return {
        "time": t,
        "velocity": velocity,
        "distance": distance,
        "acceleration": acceleration,
    }


def compute_rewards(
    scenario: dict,
    config: dict,
) -> dict:
    """计算场景中所有奖励值.

    Args:
        scenario: 场景数据字典
        config: 配置字典

    Returns:
        包含各类奖励值的字典
    """
    from rewards import speed_reward, distance_reward, comfort_reward

    v = scenario["velocity"]
    d = scenario["distance"]
    a = scenario["acceleration"]

    return {
        "time": scenario["time"],
        "speed_reward": speed_reward(v, config["speed_reward"]),
        "distance_reward": distance_reward(d, config["distance_reward"]),
        "comfort_reward": comfort_reward(a, config["comfort_reward"]),
        "total_reward": (
            speed_reward(v, config["speed_reward"])
            + distance_reward(d, config["distance_reward"])
            + comfort_reward(a, config["comfort_reward"])
        ),
    }


def ensure_output_dir(output_dir: str) -> str:
    """确保输出目录存在.

    Args:
        output_dir: 输出目录路径

    Returns:
        输出目录的绝对路径
    """
    base = os.path.dirname(__file__)
    full_path = os.path.join(base, output_dir)
    os.makedirs(full_path, exist_ok=True)
    return full_path
