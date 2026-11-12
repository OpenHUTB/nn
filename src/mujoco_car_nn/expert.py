"""Pure-pursuit expert used to generate the imitation-learning dataset.

The expert follows the track centre line with a speed profile derived
from the upcoming curvature::

    v_target = sqrt(a_lat / |kappa|)

and converts the pure-pursuit curvature into a normalised steering
command::

    delta = atan(2 * L * sin(alpha) / L_d)
    steer = delta / delta_max
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import MAX_STEER, MAX_SPEED, WHEELBASE, CarState
from .track import Track, ROAD_HALF_WIDTH


@dataclass
class ExpertConfig:
    lookahead_base: float = 4.0
    lookahead_gain: float = 0.55
    max_lookahead: float = 11.0
    max_speed: float = 12.0
    min_speed: float = 4.0
    lat_accel: float = 5.2
    speed_kp: float = 0.45
    speed_ff: float = 0.30


class PurePursuitExpert:
    def __init__(self, track: Track, config: ExpertConfig | None = None):
        self.track = track
        self.cfg = config or ExpertConfig()

    def act(self, st: CarState) -> np.ndarray:
        cfg = self.cfg
        s = st.s
        speed = max(st.speed, 0.0)
        lookahead = float(np.clip(cfg.lookahead_base + cfg.lookahead_gain * speed, 3.0, cfg.max_lookahead))

        target = self.track.point_at(s + lookahead)
        dx = target[0] - st.x
        dy = target[1] - st.y
        c, sn = np.cos(st.yaw), np.sin(st.yaw)
        x_f = c * dx + sn * dy
        y_f = -sn * dx + c * dy
        alpha = float(np.arctan2(y_f, max(x_f, 1e-3)))
        curvature = 2.0 * np.sin(alpha) / max(lookahead, 1e-3)
        steer = float(np.clip(np.arctan(curvature * WHEELBASE) / MAX_STEER, -1.0, 1.0))

        kappa = self.track.curvature_ahead(s, 14.0)
        v_target = float(np.clip(np.sqrt(cfg.lat_accel / max(kappa, 1e-4)), cfg.min_speed, cfg.max_speed))
        throttle = float(np.clip(cfg.speed_ff + cfg.speed_kp * (v_target - speed), -1.0, 1.0))
        return np.array([steer, throttle], dtype=np.float32)

    def body_frame_target(self, st: CarState) -> tuple[float, float]:
        """Debug helper: target point in the car frame."""
        lookahead = float(np.clip(self.cfg.lookahead_base + self.cfg.lookahead_gain * st.speed, 3.0, self.cfg.max_lookahead))
        target = self.track.point_at(st.s + lookahead)
        dx, dy = target[0] - st.x, target[1] - st.y
        c, sn = np.cos(st.yaw), np.sin(st.yaw)
        return float(c * dx + sn * dy), float(-sn * dx + c * dy)
