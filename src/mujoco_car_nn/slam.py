"""2-D occupancy-grid mapping with correlative scan matching (SLAM).

The pipeline is the classic "scan-to-map" front end:

1. dead-reckoning prediction from speed and yaw-rate measurements;
2. correlative scan matching of the current lidar scan against the map
   built so far (coarse-to-fine grid search);
3. log-odds occupancy update with ray casting;
4. trajectory / map accuracy evaluation against the simulator ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sim import CarSim, SimConfig
from .track import ROAD_HALF_WIDTH, WALL_HEIGHT, WALL_THICKNESS, Track


@dataclass
class GridMap:
    """Log-odds occupancy grid in world coordinates."""

    resolution: float = 0.25
    x_min: float = -55.0
    y_min: float = -45.0
    width: int = 440
    height: int = 360
    log_odds: np.ndarray = field(default=None)
    hit: float = 0.85
    miss: float = -0.35
    clamp: float = 6.0

    def __post_init__(self):
        if self.log_odds is None:
            self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)

    # ------------------------------------------------------------------
    def cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cx = ((xy[:, 0] - self.x_min) / self.resolution).astype(np.int32)
        cy = ((xy[:, 1] - self.y_min) / self.resolution).astype(np.int32)
        valid = (cx >= 0) & (cx < self.width) & (cy >= 0) & (cy < self.height)
        return cx, cy, valid

    def probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def update(self, pose: tuple[float, float, float], ranges: np.ndarray, angles: np.ndarray, max_range: float = 20.0) -> None:
        x, y, yaw = pose
        valid = ranges < max_range - 1e-3
        ranges = ranges[valid]
        angles = angles[valid]
        if len(ranges) == 0:
            return

        world_angles = yaw + angles
        dirs = np.column_stack([np.cos(world_angles), np.sin(world_angles)])
        ends = np.column_stack([x, y]) + dirs * ranges[:, None]

        # free space along each ray (skip the last metre: the wall thickness)
        n_steps = 12
        ts = np.linspace(0.1, 0.9, n_steps)
        free = (np.column_stack([x, y])[None] + dirs[:, None, :] * (ranges[:, None] * ts)[:, :, None]).reshape(-1, 2)
        cx, cy, ok = self.cells(free)
        np.add.at(self.log_odds, (cy[ok], cx[ok]), self.miss)

        # occupied endpoints, dilated by one cell to account for discretisation
        cx, cy, ok = self.cells(ends)
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                ccx = np.clip(cx + ox, 0, self.width - 1)
                ccy = np.clip(cy + oy, 0, self.height - 1)
                np.add.at(self.log_odds, (ccy[ok], ccx[ok]), self.hit / 9.0)
        np.clip(self.log_odds, -self.clamp, self.clamp, out=self.log_odds)


def ray_endpoints(pose, ranges, angles):
    x, y, yaw = pose
    a = yaw + angles
    return np.column_stack([x + ranges * np.cos(a), y + ranges * np.sin(a)])


def scan_match(
    points_local: np.ndarray,
    grid: GridMap,
    prior: tuple[float, float, float],
    search_xy: float = 0.6,
    search_yaw: float = 0.10,
    coarse: int = 5,
) -> tuple[tuple[float, float, float], float]:
    """Correlative scan matching around ``prior`` against the current map."""
    prob = grid.probability()
    x0, y0, th0 = prior

    best = (prior, -1.0)
    for level, (nxy, nth) in enumerate(((coarse, 3), (3, 3))):
        dxs = np.linspace(-search_xy, search_xy, nxy) * (0.35 if level else 1.0)
        dys = np.linspace(-search_xy, search_xy, nxy) * (0.35 if level else 1.0)
        dths = np.linspace(-search_yaw, search_yaw, nth) * (0.25 if level else 1.0)
        bx, by, bth = best[0] if level else (x0, y0, th0)
        dxs = dxs + bx
        dys = dys + by
        dths = dths + bth

        for dx in dxs:
            for dy in dys:
                for dth in dths:
                    c, s = np.cos(dth), np.sin(dth)
                    wx = c * points_local[:, 0] - s * points_local[:, 1] + dx
                    wy = s * points_local[:, 0] + c * points_local[:, 1] + dy
                    cx = ((wx - grid.x_min) / grid.resolution).astype(np.int32)
                    cy = ((wy - grid.y_min) / grid.resolution).astype(np.int32)
                    ok = (cx >= 0) & (cx < grid.width) & (cy >= 0) & (cy < grid.height)
                    if ok.sum() < 8:
                        continue
                    lv = grid.log_odds[cy[ok], cx[ok]]
                    # unknown cells have p = 0.5 and therefore contribute
                    # exactly zero, so a pose is never rewarded for simply
                    # pointing at unmapped space.
                    p = 1.0 / (1.0 + np.exp(-lv))
                    score = float(p.mean()) - 0.5
                    # motion prior: stay close to the dead-reckoned pose
                    score -= 0.6 * ((dx - x0) ** 2 + (dy - y0) ** 2) / max(search_xy**2, 1e-6)
                    score -= 0.3 * ((dth - th0) ** 2) / max(search_yaw**2, 1e-6)
                    if score > best[1]:
                        best = ((float(dx), float(dy), float(dth)), score)
    return best


def run_slam(
    track: Track,
    policy=None,
    scene: str = "day",
    laps: float = 2.0,
    max_steps: int = 2000,
    seed: int = 0,
    odom_noise: float = 0.03,
    match_blend: float = 0.5,
    record_every: int = 6,
    render=None,
):
    """Drive and build a map online; returns metrics, frames and the grid."""
    sim = CarSim(track, SimConfig(scene=scene, seed=seed, randomize_start=True))
    st = sim.reset()
    rng = np.random.default_rng(seed)

    grid = GridMap()
    pose = np.array([st.x, st.y, st.yaw], dtype=np.float64)
    traj_est = [pose[:2].copy()]
    traj_true = [np.array([st.x, st.y])]
    ate_err = []

    angles = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    prev_s = st.s
    travelled = 0.0
    steps = 0
    frames = []

    from .evaluate import rollout  # noqa: F401  (avoid cycle at import time)
    from .expert import PurePursuitExpert

    expert = PurePursuitExpert(track)

    while travelled < laps * track.length and steps < max_steps:
        # ---- 1) scan at the current (estimated) pose
        ranges = sim.lidar()
        local = np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles)])

        # ---- 2) scan-to-map matching refines the dead-reckoned pose
        if steps > 2 and grid.log_odds.any():
            (mx, my, mth), _score = scan_match(local, grid, tuple(pose), search_xy=0.35, search_yaw=0.06)
            # A single scan constrains the lateral offset and the heading far
            # better than the along-track position (the walls are almost
            # translation invariant along the road), so only the lateral and
            # yaw components of the correction are applied; the along-track
            # position keeps trusting the wheel odometry.
            yaw = pose[2]
            left = np.array([-np.sin(yaw), np.cos(yaw)])
            delta = np.array([mx - pose[0], my - pose[1]])
            lateral = float(delta @ left)
            pose = np.array(
                [
                    pose[0] + match_blend * lateral * left[0],
                    pose[1] + match_blend * lateral * left[1],
                    pose[2],
                ],
                dtype=np.float64,
            )

        # ---- 3) map update with the corrected pose
        grid.update(tuple(pose), ranges, angles)

        traj_est.append(pose[:2].copy())
        traj_true.append(np.array([st.x, st.y]))
        ate_err.append(float(np.linalg.norm(pose[:2] - traj_true[-1])))

        if record_every and steps % record_every == 0 and render is not None:
            frames.append(render(grid, traj_est, pose, st))

        # ---- 4) control + physics step
        if policy is None:
            action = expert.act(st)
        else:
            action = policy(sim.camera_image(), st.speed)
        st_new = sim.step(np.clip(action, -1, 1))

        # ---- 5) dead reckoning for the next prior (noisy wheel/IMU odometry)
        dt = 0.05
        v = 0.5 * (st.speed + st_new.speed) * (1.0 + rng.normal(0, odom_noise))
        w = 0.5 * (st.yaw_rate + st_new.yaw_rate) * (1.0 + rng.normal(0, odom_noise * 2))
        yaw_mid = pose[2] + 0.5 * w * dt
        pose = np.array(
            [pose[0] + v * np.cos(yaw_mid) * dt, pose[1] + v * np.sin(yaw_mid) * dt, pose[2] + w * dt],
            dtype=np.float64,
        )

        travelled += (st_new.s - prev_s) % track.length
        prev_s = st_new.s
        steps += 1
        st = st_new

    metrics = {
        "steps": steps,
        "laps": travelled / track.length,
        "ate_rmse": float(np.sqrt(np.mean(np.square(ate_err)))) if ate_err else 0.0,
        "ate_max": float(np.max(ate_err)) if ate_err else 0.0,
        "map_cells": int((grid.log_odds > 1.0).sum()),
        "trajectory": np.asarray(traj_true),
        "estimated": np.asarray(traj_est),
        "coverage": float((grid.log_odds != 0).mean()),
    }
    sim.close()
    return metrics, frames, grid


def ground_truth_map(grid: GridMap, track: Track) -> np.ndarray:
    """Rasterise the wall rectangles into an occupancy grid (for IoU)."""
    gt = np.zeros_like(grid.log_odds)
    half = ROAD_HALF_WIDTH + WALL_THICKNESS * 0.5
    for xy, yaw, hl in track.wall_poses(spacing=2.0):
        for (x, y), a in zip(xy, yaw):
            c, s = np.cos(a), np.sin(a)
            for t in np.linspace(-hl, hl, 12):
                for o in np.linspace(-WALL_THICKNESS / 2, WALL_THICKNESS / 2, 3):
                    wx = x + c * t - s * o
                    wy = y + s * t + c * o
                    cx = int((wx - grid.x_min) / grid.resolution)
                    cy = int((wy - grid.y_min) / grid.resolution)
                    if 0 <= cx < grid.width and 0 <= cy < grid.height:
                        gt[cy, cx] = 1.0
    return gt


def map_iou(grid: GridMap, track: Track, threshold: float = 1.0) -> float:
    gt = ground_truth_map(grid, track) > 0
    pred = grid.log_odds > threshold
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return float(inter / max(union, 1))
