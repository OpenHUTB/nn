"""PyTorch multimodal fusion model for rainy autonomous-driving scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import Tensor, nn

from ..config import FusionConfig
from ..detection.risk import choose_action, estimate_box3d
from ..detection.risk import estimate_object_confidence, estimate_physical_risk
from ..sensor.synthetic import SENSOR_NAMES, SensorSnapshot


class ModalityEncoder(nn.Module):
    """Small MLP encoder for one sensor stream."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


def _ensure_2d(inputs: Tensor, expected_dim: int, name: str) -> Tensor:
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(0)
    if inputs.ndim != 2 or inputs.shape[-1] != expected_dim:
        raise ValueError(f"{name} must have shape [batch, {expected_dim}]")
    return inputs.float()


def estimate_quality_prior(
    camera: Tensor,
    lidar: Tensor,
    radar: Tensor,
    weather: Tensor,
) -> Tensor:
    """Estimate interpretable camera/LiDAR/radar reliability weights."""

    rain = weather[:, 0]
    fog = weather[:, 1]

    camera_base = (
        0.35 * camera[:, 0]
        + 0.20 * camera[:, 1]
        + 0.25 * camera[:, 2]
        + 0.10 * camera[:, 3]
        + 0.10 * camera[:, 4]
    )
    lidar_base = (
        0.30 * lidar[:, 0]
        + 0.35 * lidar[:, 1]
        + 0.25 * lidar[:, 2]
        + 0.10 * (1.0 - lidar[:, 3])
    )
    radar_base = (
        0.25 * radar[:, 0]
        + 0.25 * radar[:, 1]
        + 0.35 * radar[:, 2]
        + 0.15 * radar[:, 3]
    )

    camera_quality = camera_base * (1.0 - 0.45 * rain - 0.25 * fog)
    lidar_quality = lidar_base * (1.0 - 0.20 * rain - 0.08 * fog)
    radar_quality = radar_base * (1.0 + 0.25 * rain + 0.08 * fog)

    raw = torch.stack([camera_quality, lidar_quality, radar_quality], dim=-1)
    raw = torch.clamp(raw, min=1e-5)
    return raw / raw.sum(dim=-1, keepdim=True)


class MultiModalRainFusionNet(nn.Module):
    """Deep feature fusion with rain-aware attention over three sensors."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or FusionConfig()
        self.config.validate()
        hidden_dim = self.config.hidden_dim

        self.camera_encoder = ModalityEncoder(self.config.camera_dim, hidden_dim)
        self.lidar_encoder = ModalityEncoder(self.config.lidar_dim, hidden_dim)
        self.radar_encoder = ModalityEncoder(self.config.radar_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3 + self.config.weather_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + self.config.weather_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(
        self,
        camera: Tensor,
        lidar: Tensor,
        radar: Tensor,
        weather: Tensor,
    ) -> Dict[str, Tensor]:
        camera = _ensure_2d(camera, self.config.camera_dim, "camera")
        lidar = _ensure_2d(lidar, self.config.lidar_dim, "lidar")
        radar = _ensure_2d(radar, self.config.radar_dim, "radar")
        weather = _ensure_2d(weather, self.config.weather_dim, "weather")

        encoded_camera = self.camera_encoder(camera)
        encoded_lidar = self.lidar_encoder(lidar)
        encoded_radar = self.radar_encoder(radar)
        encoded = torch.stack([encoded_camera, encoded_lidar, encoded_radar], dim=1)

        gate_input = torch.cat([encoded.flatten(start_dim=1), weather], dim=-1)
        learned_weights = torch.softmax(self.gate(gate_input), dim=-1)
        prior_weights = estimate_quality_prior(camera, lidar, radar, weather)
        alpha = self.config.quality_prior_strength
        weights = (1.0 - alpha) * learned_weights + alpha * prior_weights
        weights = weights / weights.sum(dim=-1, keepdim=True)

        fused = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        raw = self.head(torch.cat([fused, weather], dim=-1))
        return {
            "weights": weights,
            "risk": torch.sigmoid(raw[:, 0:1]),
            "objectness": torch.sigmoid(raw[:, 1:2]),
            "distance": torch.sigmoid(raw[:, 2:3]),
            "lateral": torch.tanh(raw[:, 3:4]),
            "closing_speed": torch.sigmoid(raw[:, 4:5]),
            "rain": torch.sigmoid(raw[:, 5:6]),
            "fused_embedding": fused,
        }


@dataclass(frozen=True)
class FusionResult:
    """Runtime output consumed by the demo, tests, or planners."""

    modality_weights: Dict[str, float]
    rain_severity: float
    object_confidence: float
    risk_score: float
    recommended_action: str
    box_3d: Dict[str, float]
    network_risk: float
    physical_risk: float


class RainFusionEngine:
    """High-level runtime wrapper around the deep fusion model."""

    def __init__(
        self,
        config: FusionConfig | None = None,
        model: MultiModalRainFusionNet | None = None,
        *,
        seed: int = 7,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or FusionConfig()
        self.config.validate()
        self.device = torch.device(device or "cpu")
        torch.manual_seed(seed)
        self.model = model or MultiModalRainFusionNet(self.config)
        self.model.to(self.device)
        self.model.eval()

    def _snapshot_to_tensors(self, snapshot: SensorSnapshot) -> Dict[str, Tensor]:
        arrays = snapshot.as_arrays()
        return {
            name: torch.from_numpy(value).unsqueeze(0).to(self.device)
            for name, value in arrays.items()
        }

    def analyze(self, snapshot: SensorSnapshot) -> FusionResult:
        """Run deep fusion and return planner-friendly risk information."""

        with torch.no_grad():
            outputs = self.model(**self._snapshot_to_tensors(snapshot))

        weights_array = outputs["weights"][0].detach().cpu().numpy()
        weights = {
            name: round(float(weight), 4)
            for name, weight in zip(SENSOR_NAMES, weights_array)
        }
        network_risk = float(outputs["risk"][0, 0].detach().cpu())
        physical_risk = estimate_physical_risk(snapshot)
        rain = float(snapshot.weather[0])
        object_from_network = float(outputs["objectness"][0, 0].detach().cpu())
        object_from_sensors = estimate_object_confidence(snapshot, weights_array)
        object_confidence = float(np.clip(0.35 * object_from_network + 0.65 * object_from_sensors, 0.0, 1.0))

        risk_score = float(
            np.clip(
                self.config.network_risk_weight * network_risk
                + self.config.physical_risk_weight * physical_risk
                + self.config.rain_risk_weight * rain,
                0.0,
                1.0,
            )
        )
        box = estimate_box3d(
            snapshot,
            object_confidence,
            max_distance_m=self.config.max_detection_distance_m,
        )
        return FusionResult(
            modality_weights=weights,
            rain_severity=round(rain, 4),
            object_confidence=round(object_confidence, 4),
            risk_score=round(risk_score, 4),
            recommended_action=choose_action(risk_score, self.config),
            box_3d=box.as_dict(),
            network_risk=round(network_risk, 4),
            physical_risk=round(physical_risk, 4),
        )
