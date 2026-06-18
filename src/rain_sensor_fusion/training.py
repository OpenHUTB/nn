"""Synthetic training loop for the rain-aware fusion network."""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from .fusion.deep_fusion import MultiModalRainFusionNet
from .sensor.synthetic import generate_training_batch


def train_on_synthetic_rain(
    model: MultiModalRainFusionNet,
    *,
    steps: int = 120,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 17,
) -> List[float]:
    """Train the network on generated teacher labels.

    The goal is not to replace real CARLA data. It gives the project a quick
    reproducible deep-learning path before connecting real synchronized sensor
    frames.
    """

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    history: List[float] = []

    for step in range(steps):
        features, labels = generate_training_batch(batch_size, seed=seed + step)
        camera = torch.from_numpy(features["camera"])
        lidar = torch.from_numpy(features["lidar"])
        radar = torch.from_numpy(features["radar"])
        weather = torch.from_numpy(features["weather"])
        target_weights = torch.from_numpy(labels["weights"])
        target_risk = torch.from_numpy(labels["risk"])
        target_objectness = torch.from_numpy(labels["objectness"])
        target_rain = torch.from_numpy(labels["rain"])

        outputs = model(camera=camera, lidar=lidar, radar=radar, weather=weather)
        loss = (
            mse(outputs["weights"], target_weights)
            + mse(outputs["risk"], target_risk)
            + mse(outputs["objectness"], target_objectness)
            + 0.5 * mse(outputs["rain"], target_rain)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    model.eval()
    return history
