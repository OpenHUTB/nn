"""Rain-aware deep sensor fusion for autonomous driving demos."""

from .fusion.deep_fusion import FusionResult, MultiModalRainFusionNet, RainFusionEngine
from .sensor.synthetic import SensorSnapshot, make_demo_snapshot, make_sensor_snapshot

__all__ = [
    "FusionResult",
    "MultiModalRainFusionNet",
    "RainFusionEngine",
    "SensorSnapshot",
    "make_demo_snapshot",
    "make_sensor_snapshot",
]
