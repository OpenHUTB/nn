"""
min-carla-env: 极简 2D CARLA 强化学习环境
=====================================
提供类 Gym 接口的 Carla 自动驾驶仿真环境。
"""

# 配置模块始终可导入（无外部依赖）
from min_carla_env.config import (  # noqa: F401
    # 主配置
    CONFIG,
    # 子配置
    OBS_CONFIG,
    ENV_CONFIG,
    REWARD_CONFIG,
    RENDER_CONFIG,
    # 预设
    WEATHER_PRESETS,
    MAP_OPTIONS,
    # 动作空间
    ACTIONS,
)

# 环境 / 世界模块（需要安装 gym 和 carla）
try:
    from min_carla_env.env import CarlaEnv, reconnect_carla_client  # noqa: F401
    from min_carla_env.matrix_world import MatrixWorld  # noqa: F401
except ImportError as e:
    import warnings
    warnings.warn(
        f"部分模块导入失败（可能缺少 gym 或 carla 依赖）: {e}",
        ImportWarning, stacklevel=2
    )

__all__ = [
    # config
    "CONFIG", "OBS_CONFIG", "ENV_CONFIG", "REWARD_CONFIG", "RENDER_CONFIG",
    "WEATHER_PRESETS", "MAP_OPTIONS", "ACTIONS",
    # env
    "CarlaEnv", "MatrixWorld", "reconnect_carla_client",
]
