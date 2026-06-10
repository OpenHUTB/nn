"""Synthetic sensor utilities for rain sensor fusion."""

from .synthetic import DEMO_SCENARIOS, SensorSnapshot, generate_training_batch
from .synthetic import make_demo_snapshot, make_sensor_snapshot

__all__ = [
    "DEMO_SCENARIOS",
    "SensorSnapshot",
    "generate_training_batch",
    "make_demo_snapshot",
    "make_sensor_snapshot",
]
