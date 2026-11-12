"""Closed-loop test track geometry.

The centre line is a smooth, star-shaped closed curve expressed in polar
form::

    r(theta) = R0 + A * cos(2*theta) + B * sin(3*theta)

which gives a long oval with two gentle chicanes, a minimum turning
radius of about 12 m and no self intersections.  Everything else (road
mesh, lane markings, walls, expert look-ahead, SLAM ground truth) is
derived from the sampled centre line so that geometry, rendering and
evaluation always agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Real-world scale (metres): a 4.2 m car on a 9 m wide two-lane road.
ROAD_HALF_WIDTH = 4.5
CAR_LENGTH = 4.2
CAR_WIDTH = 1.8
WALL_THICKNESS = 0.16
WALL_HEIGHT = 0.7

# Shape parameters of the closed curve.
R0, AMP2, AMP3 = 34.0, 7.0, 3.5


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class Track:
    """Sampled centre line with tangents, normals and curvature."""

    points: np.ndarray  # (N, 2) centre line
    tangents: np.ndarray  # (N, 2) unit heading
    normals: np.ndarray  # (N, 2) unit left normal
    curvature: np.ndarray  # (N,) signed curvature, 1/m
    s: np.ndarray  # (N,) arc length from the start
    length: float

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _index_at(self, s: float) -> int:
        s = float(s) % self.length
        return int(np.searchsorted(self.s, s, side="right") - 1) % len(self.s)

    def heading_at(self, s: float) -> float:
        idx = self._index_at(s)
        return float(np.arctan2(self.tangents[idx, 1], self.tangents[idx, 0]))

    def point_at(self, s: float) -> np.ndarray:
        return self.points[self._index_at(s)].copy()

    def nearest(self, xy: np.ndarray) -> tuple[int, float]:
        """Index of the nearest centre-line sample and the signed lateral error.

        The lateral error is positive when the point lies to the left of
        the centre line.
        """
        xy = np.asarray(xy, dtype=np.float64)
        d2 = np.sum((self.points - xy[:2]) ** 2, axis=1)
        idx = int(np.argmin(d2))
        delta = xy[:2] - self.points[idx]
        return idx, float(delta @ self.normals[idx])

    def lateral_error(self, xy: np.ndarray, heading: float) -> tuple[float, float]:
        """Signed lateral error (m) and heading error (rad)."""
        idx, lateral = self.nearest(xy)
        track_heading = float(np.arctan2(self.tangents[idx, 1], self.tangents[idx, 0]))
        return lateral, float(_wrap(heading - track_heading))

    def curvature_ahead(self, s: float, distance: float) -> float:
        """Largest |curvature| over the next ``distance`` metres."""
        idx = self._index_at(s)
        end = self._index_at(s + distance)
        if end <= idx:
            span = np.r_[idx : len(self.curvature), 0 : end + 1]
        else:
            span = np.arange(idx, end + 1)
        return float(np.max(np.abs(self.curvature[span])))

    def progress(self, xy: np.ndarray) -> float:
        """Arc length of the nearest centre-line sample to ``xy``."""
        return float(self.s[self.nearest(xy)[0]])

    # ------------------------------------------------------------------
    # Geometry generation
    # ------------------------------------------------------------------
    def wall_poses(self, spacing: float = 2.0):
        """(xy, yaw, half_length) poses for the boxes forming both walls."""
        step = max(1, int(round(len(self.points) / self.length * spacing)))
        idx = np.arange(0, len(self.points), step)
        half_len = spacing * 0.55
        yaw = np.arctan2(self.tangents[idx, 1], self.tangents[idx, 0])
        out = []
        for side in (+1.0, -1.0):
            offset = side * (ROAD_HALF_WIDTH + WALL_THICKNESS * 0.5)
            xy = self.points[idx] + offset * self.normals[idx]
            out.append((xy, yaw, half_len))
        return out

    def write_strip_obj(self, path: str, offset: float, half_width: float, z: float = 0.012) -> None:
        """Write a quad strip (road surface or lane marking) as Wavefront OBJ."""
        step = 2
        idx = np.arange(0, len(self.points), step)
        left = self.points[idx] + (offset + half_width) * self.normals[idx]
        right = self.points[idx] + (offset - half_width) * self.normals[idx]
        n = len(idx)
        with open(path, "w", encoding="ascii") as fh:
            for (lx, ly), (rx, ry) in zip(left, right):
                fh.write(f"v {lx:.4f} {ly:.4f} {z:.4f}\n")
                fh.write(f"v {rx:.4f} {ry:.4f} {z:.4f}\n")
            for i in range(n):
                a, b = 2 * i + 1, 2 * i + 2
                c, d = 2 * ((i + 1) % n) + 1, 2 * ((i + 1) % n) + 2
                fh.write(f"f {a} {b} {d}\n")
                fh.write(f"f {a} {d} {c}\n")

    def bounds(self) -> tuple[float, float, float, float]:
        return (
            float(self.points[:, 0].min()),
            float(self.points[:, 1].min()),
            float(self.points[:, 0].max()),
            float(self.points[:, 1].max()),
        )


def build_track(resolution: float = 0.25) -> Track:
    """Sample the analytic curve at (approximately) uniform arc length."""
    t = np.linspace(0.0, 2.0 * np.pi, 20000, endpoint=False)
    r = R0 + AMP2 * np.cos(2.0 * t) + AMP3 * np.sin(3.0 * t)
    pts = np.column_stack([r * np.cos(t), r * np.sin(t)])

    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    s_raw = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    total = float(seg.sum())
    n = int(round(total / resolution))
    s = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(s, np.r_[s_raw, total], np.r_[pts[:, 0], pts[0, 0]])
    y = np.interp(s, np.r_[s_raw, total], np.r_[pts[:, 1], pts[0, 1]])
    points = np.column_stack([x, y])

    d = np.gradient(points, axis=0)
    tangents = d / np.linalg.norm(d, axis=1, keepdims=True)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])  # left normal

    nxt = np.roll(tangents, -1, axis=0)
    cross = tangents[:, 0] * nxt[:, 1] - tangents[:, 1] * nxt[:, 0]
    dtheta = np.arctan2(cross, np.sum(tangents * nxt, axis=1))
    ds = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    curvature = dtheta / np.maximum(ds, 1e-6)

    kernel = np.ones(5) / 5.0
    curvature = np.convolve(np.r_[curvature[-4:], curvature, curvature[:4]], kernel, mode="same")[4:-4]

    return Track(points=points, tangents=tangents, normals=normals, curvature=curvature, s=s, length=total)


def start_pose(track: Track, lateral_offset: float = 0.0) -> tuple[np.ndarray, float]:
    """Initial (x, y) and yaw of the car at the start line."""
    xy = track.points[0] + lateral_offset * track.normals[0]
    yaw = float(np.arctan2(track.tangents[0, 1], track.tangents[0, 0]))
    return xy.astype(np.float64), yaw
