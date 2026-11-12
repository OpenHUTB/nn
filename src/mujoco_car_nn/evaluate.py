"""Closed-loop evaluation of driving policies."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .expert import PurePursuitExpert
from .sim import CarSim, SimConfig


@dataclass
class EvalResult:
    policy: str = ""
    scene: str = "day"
    laps_target: float = 2.0
    laps_completed: float = 0.0
    success: bool = False
    collisions: int = 0
    lateral_rmse: float = 0.0
    lateral_max: float = 0.0
    speed_mean: float = 0.0
    speed_max: float = 0.0
    sim_seconds: float = 0.0
    lap_time: float = 0.0
    steps: int = 0
    inference_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def rollout(
    policy: Callable[[np.ndarray], np.ndarray] | None,
    track,
    scene: str = "day",
    laps: float = 2.0,
    max_steps: int = 3000,
    seed: int = 0,
    randomize_start: bool = True,
    record: bool = False,
    record_every: int = 4,
    policy_name: str = "policy",
    on_step: Callable | None = None,
) -> tuple[EvalResult, list]:
    """Run ``policy`` in closed loop and compute driving metrics.

    ``policy`` takes the 64x64 ego image and returns ``[steer, throttle]``.
    When it is ``None`` the pure-pursuit expert drives (used as a reference).
    """
    sim = CarSim(track, SimConfig(scene=scene, seed=seed, randomize_start=randomize_start))
    expert = PurePursuitExpert(track)
    st = sim.reset()

    prev_s = st.s
    travelled = 0.0
    steps = 0
    collisions = 0
    contact = False
    lat, speeds, frames = [], [], []
    inference = []
    t0 = time.time()

    while travelled < laps * track.length and steps < max_steps:
        if policy is None:
            action = expert.act(st)
        else:
            t1 = time.perf_counter()
            action = policy(sim.camera_image(), st.speed)
            inference.append((time.perf_counter() - t1) * 1000.0)

        st = sim.step(np.clip(action, -1, 1))
        travelled += (st.s - prev_s) % track.length
        prev_s = st.s
        lat.append(st.lateral)
        speeds.append(st.speed)
        steps += 1

        if sim.data.ncon > 0 and not contact:
            collisions += 1
        contact = sim.data.ncon > 0

        if record and steps % record_every == 0:
            frames.append(sim.render_chase(480, 270))
        if on_step is not None:
            on_step(sim, st)
        if st.speed < 0.05 and steps > 200:
            break

    sim_seconds = steps * 0.05
    laps_completed = travelled / track.length
    res = EvalResult(
        policy=policy_name,
        scene=scene,
        laps_target=laps,
        laps_completed=laps_completed,
        success=laps_completed >= laps - 0.02 and collisions == 0,
        collisions=collisions,
        lateral_rmse=float(np.sqrt(np.mean(np.square(lat)))) if lat else 0.0,
        lateral_max=float(np.max(np.abs(lat))) if lat else 0.0,
        speed_mean=float(np.mean(speeds)) if speeds else 0.0,
        speed_max=float(np.max(speeds)) if speeds else 0.0,
        sim_seconds=sim_seconds,
        lap_time=sim_seconds / max(laps_completed, 1e-6),
        steps=steps,
        inference_ms=float(np.median(inference)) if inference else 0.0,
    )
    sim.close()
    return res, frames
