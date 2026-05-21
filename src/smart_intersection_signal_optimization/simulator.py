"""Queue simulation engine for signal timing plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from scenario import IntersectionScenario, SignalPlan


@dataclass
class SimulationResult:
    plan: SignalPlan
    time: np.ndarray
    queues: Dict[str, np.ndarray]
    served: Dict[str, np.ndarray]
    arrivals: Dict[str, np.ndarray]
    active_phase: List[str]
    average_delay: float
    max_queue: float
    throughput: float
    stops: float
    fairness: float
    score: float


def phase_at(plan: SignalPlan, phase_order: List[str], second: int) -> str:
    elapsed = second % plan.cycle_length
    cursor = 0
    for phase in phase_order:
        cursor += plan.green_for(phase)
        if elapsed < cursor:
            return phase
    return "lost_time"


def simulate_plan(scenario: IntersectionScenario, plan: SignalPlan, seed: int = 7) -> SimulationResult:
    """Simulate queue dynamics using deterministic demand plus mild waves."""

    plan = plan.normalized(scenario.approaches)
    rng = np.random.default_rng(seed)
    horizon = scenario.horizon_seconds
    time = np.arange(0, horizon, scenario.time_step)
    queues = {approach.name: np.zeros_like(time, dtype=float) for approach in scenario.approaches}
    arrivals = {approach.name: np.zeros_like(time, dtype=float) for approach in scenario.approaches}
    served = {approach.name: np.zeros_like(time, dtype=float) for approach in scenario.approaches}
    active_phase: List[str] = []
    approach_map = scenario.approach_map()
    cumulative_delay = 0.0
    cumulative_stops = 0.0

    for idx, second in enumerate(time):
        phase = phase_at(plan, scenario.names, int(second))
        active_phase.append(phase)
        for phase_index, approach in enumerate(scenario.approaches):
            wave = 1.0 + 0.18 * np.sin((second / 95.0) + phase_index * 0.9)
            pulse = 1.0 + (0.28 if (second // 300 + phase_index) % 4 == 0 else 0.0)
            noise = rng.normal(0.0, 0.015)
            arrive = max(0.0, approach.demand_per_second * wave * pulse + noise)
            previous_queue = queues[approach.name][idx - 1] if idx else 0.0
            capacity = approach.service_per_second if phase == approach.name else 0.0
            depart = min(previous_queue + arrive, capacity)
            queue = max(0.0, previous_queue + arrive - depart)
            arrivals[approach.name][idx] = arrive
            served[approach.name][idx] = depart
            queues[approach.name][idx] = queue
            cumulative_delay += queue * approach.priority
            cumulative_stops += 1.0 if queue > 0.5 and phase != approach.name else 0.0

    total_arrivals = sum(float(values.sum()) for values in arrivals.values())
    total_served = sum(float(values.sum()) for values in served.values())
    mean_delay = cumulative_delay / max(total_arrivals, 1.0)
    max_queue = max(float(values.max()) for values in queues.values())
    throughput = total_served / max(total_arrivals, 1.0)
    queue_totals = np.array([values.mean() for values in queues.values()])
    fairness = float(1.0 - np.std(queue_totals) / max(np.mean(queue_totals) + 1e-6, 1.0))
    score = mean_delay + 0.75 * max_queue + 16.0 * (1.0 - throughput) + 1.8 * cumulative_stops / len(time)
    return SimulationResult(
        plan=plan,
        time=time,
        queues=queues,
        served=served,
        arrivals=arrivals,
        active_phase=active_phase,
        average_delay=float(mean_delay),
        max_queue=float(max_queue),
        throughput=float(throughput),
        stops=float(cumulative_stops),
        fairness=float(fairness),
        score=float(score),
    )


def summarize_result(result: SimulationResult) -> Dict[str, float]:
    return {
        "cycle_length": result.plan.cycle_length,
        "average_delay": round(result.average_delay, 3),
        "max_queue": round(result.max_queue, 3),
        "throughput": round(result.throughput, 4),
        "stops": round(result.stops, 3),
        "fairness": round(result.fairness, 4),
        "score": round(result.score, 3),
    }
