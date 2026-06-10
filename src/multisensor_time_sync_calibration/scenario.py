"""Synthetic scenario for multi-sensor time synchronization calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorTrack:
    """A sampled 2D object track from one sensor."""

    name: str
    timestamps: np.ndarray
    positions: np.ndarray


@dataclass
class SyncScenario:
    """Ground truth motion and two asynchronously sampled sensor tracks."""

    truth_time: np.ndarray
    truth_positions: np.ndarray
    camera: SensorTrack
    lidar: SensorTrack
    true_lidar_delay: float


def true_motion(time: np.ndarray) -> np.ndarray:
    """Generate a curved vehicle trajectory with changing velocity."""
    x = 4.2 * time + 1.5 * np.sin(time / 1.8)
    y = 2.8 * np.sin(time / 2.5) + 0.18 * time**1.7
    return np.column_stack([x, y])


def sample_track(
    name: str,
    sample_rate_hz: float,
    duration_s: float,
    timestamp_delay_s: float,
    noise_std: float,
    seed: int,
) -> SensorTrack:
    """Sample the same target motion with timestamp delay and Gaussian noise."""
    rng = np.random.default_rng(seed)
    timestamps = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)
    sensed_time = np.clip(timestamps - timestamp_delay_s, 0.0, duration_s)
    positions = true_motion(sensed_time) + rng.normal(0.0, noise_std, size=(len(timestamps), 2))
    return SensorTrack(name=name, timestamps=timestamps, positions=positions)


def make_demo_scenario(seed: int = 7) -> SyncScenario:
    """Create a deterministic camera/lidar timestamp offset scenario."""
    duration = 18.0
    truth_time = np.linspace(0.0, duration, 600)
    truth_positions = true_motion(truth_time)
    camera = sample_track("camera", 12.0, duration, timestamp_delay_s=0.0, noise_std=0.22, seed=seed)
    lidar_delay = 0.42
    lidar = sample_track("lidar", 9.0, duration, timestamp_delay_s=lidar_delay, noise_std=0.16, seed=seed + 1)
    return SyncScenario(truth_time, truth_positions, camera, lidar, true_lidar_delay=lidar_delay)


def interpolate_positions(track: SensorTrack, query_time: np.ndarray) -> np.ndarray:
    """Interpolate a sensor track at requested timestamps."""
    x = np.interp(query_time, track.timestamps, track.positions[:, 0])
    y = np.interp(query_time, track.timestamps, track.positions[:, 1])
    return np.column_stack([x, y])
