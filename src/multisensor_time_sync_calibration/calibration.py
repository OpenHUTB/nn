"""Timestamp offset estimation and synchronized fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario import SensorTrack, SyncScenario, interpolate_positions


@dataclass
class CalibrationResult:
    candidate_offsets: np.ndarray
    alignment_errors: np.ndarray
    estimated_lidar_delay: float
    before_rmse: float
    after_rmse: float
    fused_time: np.ndarray
    fused_positions: np.ndarray
    unsynced_positions: np.ndarray


def common_time_grid(camera: SensorTrack, lidar: SensorTrack, margin: float = 0.5) -> np.ndarray:
    start = max(float(camera.timestamps[0]), float(lidar.timestamps[0])) + margin
    end = min(float(camera.timestamps[-1]), float(lidar.timestamps[-1])) - margin
    return np.linspace(start, end, 220)


def alignment_error(camera: SensorTrack, lidar: SensorTrack, query_time: np.ndarray, lidar_delay: float) -> float:
    """Compute mean camera/lidar distance after correcting lidar timestamps."""
    camera_pos = interpolate_positions(camera, query_time)
    lidar_pos = interpolate_positions(lidar, query_time + lidar_delay)
    diff = camera_pos - lidar_pos
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def estimate_lidar_delay(
    camera: SensorTrack,
    lidar: SensorTrack,
    min_offset: float = -0.8,
    max_offset: float = 0.8,
    step: float = 0.01,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate lidar delay by grid-searching the alignment residual."""
    query_time = common_time_grid(camera, lidar)
    offsets = np.arange(min_offset, max_offset + step / 2.0, step)
    errors = np.array([alignment_error(camera, lidar, query_time, offset) for offset in offsets])
    best_index = int(np.argmin(errors))
    return float(offsets[best_index]), offsets, errors


def fuse_tracks(scenario: SyncScenario, lidar_delay: float) -> tuple[np.ndarray, np.ndarray]:
    """Fuse camera and delay-corrected lidar positions on a common timeline."""
    query_time = common_time_grid(scenario.camera, scenario.lidar)
    camera_pos = interpolate_positions(scenario.camera, query_time)
    lidar_pos = interpolate_positions(scenario.lidar, query_time + lidar_delay)
    # Lidar has lower simulated noise, so give it slightly higher weight.
    fused = 0.42 * camera_pos + 0.58 * lidar_pos
    return query_time, fused


def evaluate_against_truth(scenario: SyncScenario, query_time: np.ndarray, positions: np.ndarray) -> float:
    truth_x = np.interp(query_time, scenario.truth_time, scenario.truth_positions[:, 0])
    truth_y = np.interp(query_time, scenario.truth_time, scenario.truth_positions[:, 1])
    truth = np.column_stack([truth_x, truth_y])
    diff = positions - truth
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def run_calibration(scenario: SyncScenario) -> CalibrationResult:
    estimated_delay, offsets, errors = estimate_lidar_delay(scenario.camera, scenario.lidar)
    fused_time, corrected_fused = fuse_tracks(scenario, estimated_delay)
    _, unsynced_fused = fuse_tracks(scenario, 0.0)
    before_rmse = evaluate_against_truth(scenario, fused_time, unsynced_fused)
    after_rmse = evaluate_against_truth(scenario, fused_time, corrected_fused)
    return CalibrationResult(
        candidate_offsets=offsets,
        alignment_errors=errors,
        estimated_lidar_delay=estimated_delay,
        before_rmse=before_rmse,
        after_rmse=after_rmse,
        fused_time=fused_time,
        fused_positions=corrected_fused,
        unsynced_positions=unsynced_fused,
    )
