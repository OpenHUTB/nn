"""Configuration for rain-aware multimodal fusion."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionConfig:
    """Small default configuration for the lightweight fusion model."""

    camera_dim: int = 5
    lidar_dim: int = 4
    radar_dim: int = 4
    weather_dim: int = 3
    hidden_dim: int = 32
    quality_prior_strength: float = 0.72
    network_risk_weight: float = 0.25
    physical_risk_weight: float = 0.65
    rain_risk_weight: float = 0.10
    slow_down_threshold: float = 0.45
    brake_threshold: float = 0.72
    max_detection_distance_m: float = 80.0

    def validate(self) -> None:
        """Validate values that influence tensor shapes and scoring."""

        if self.camera_dim <= 0 or self.lidar_dim <= 0 or self.radar_dim <= 0:
            raise ValueError("sensor feature dimensions must be positive")
        if self.weather_dim != 3:
            raise ValueError("weather_dim must be 3: rain, fog, wet road")
        if not 0.0 <= self.quality_prior_strength <= 1.0:
            raise ValueError("quality_prior_strength must be in [0, 1]")
        if self.slow_down_threshold >= self.brake_threshold:
            raise ValueError("slow_down_threshold must be below brake_threshold")
