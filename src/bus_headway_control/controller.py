"""Bus corridor simulation with baseline and adaptive holding control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario import BusScenario, travel_time_between_stops


@dataclass
class BusRun:
    bus_id: int
    arrival_min: np.ndarray
    departure_min: np.ndarray
    onboard_after_stop: np.ndarray
    boarded_by_stop: np.ndarray
    holding_by_stop: np.ndarray
    wait_cost_by_stop: np.ndarray


@dataclass
class HeadwayResult:
    strategy_name: str
    runs: list[BusRun]
    headways_by_stop: np.ndarray
    mean_headway_min: float
    headway_std_min: float
    headway_cv: float
    bunching_count: int
    total_passenger_wait_min: float
    average_passenger_wait_min: float
    total_holding_min: float
    max_load: int
    service_reliability_score: float


def _holding_time(
    strategy_name: str,
    current_headway: float,
    next_gap: float,
    scenario: BusScenario,
    onboard: int,
) -> float:
    if strategy_name == "no_control":
        return 0.0
    if strategy_name != "adaptive_holding":
        raise ValueError(f"unknown strategy: {strategy_name}")

    short_gap = max(0.0, scenario.target_headway_min - current_headway)
    downstream_risk = max(0.0, scenario.target_headway_min - next_gap)
    crowding_discount = 0.35 if onboard > scenario.boarding_capacity * 0.82 else 1.0
    return float(np.clip((0.28 * short_gap + 0.10 * downstream_risk) * crowding_discount, 0.0, 1.0))


def simulate_corridor(scenario: BusScenario, strategy_name: str) -> HeadwayResult:
    """Simulate a line of buses over all corridor stops."""
    stop_count = len(scenario.stop_positions_km)
    travel_times = travel_time_between_stops(scenario)
    departures = np.zeros((scenario.bus_count, stop_count))
    arrivals = np.zeros((scenario.bus_count, stop_count))
    onboard = np.zeros((scenario.bus_count, stop_count), dtype=int)
    boarded = np.zeros((scenario.bus_count, stop_count), dtype=int)
    holding = np.zeros((scenario.bus_count, stop_count))
    wait_cost = np.zeros((scenario.bus_count, stop_count))

    initial_departures = np.arange(scenario.bus_count) * scenario.target_headway_min
    for bus in range(scenario.bus_count):
        arrivals[bus, 0] = initial_departures[bus]

    for bus in range(scenario.bus_count):
        current_load = 0
        for stop in range(stop_count):
            if stop > 0:
                arrivals[bus, stop] = departures[bus, stop - 1] + travel_times[stop - 1]
                if bus == scenario.delay_bus_index and stop == scenario.delay_stop_index:
                    arrivals[bus, stop] += scenario.delay_min

            previous_departure = departures[bus - 1, stop] if bus > 0 else arrivals[bus, stop] - scenario.target_headway_min
            headway = max(1.0, arrivals[bus, stop] - previous_departure)
            next_gap = scenario.target_headway_min
            if bus > 0 and stop > 0:
                next_gap = max(0.5, departures[bus - 1, stop - 1] + travel_times[stop - 1] - arrivals[bus, stop])

            passenger_demand = scenario.passenger_arrival_rate_per_min[stop] * headway
            available_capacity = max(0, scenario.boarding_capacity - current_load)
            boarded_count = int(min(available_capacity, round(passenger_demand)))
            dwell = scenario.dwell_base_min + scenario.dwell_per_passenger_min * boarded_count
            hold = _holding_time(strategy_name, headway, next_gap, scenario, current_load)

            alight_fraction = 0.10 + 0.04 * (stop / max(stop_count - 1, 1))
            alighting = int(round(current_load * alight_fraction)) if stop > 0 else 0
            current_load = max(0, current_load - alighting) + boarded_count

            wait_cost[bus, stop] = passenger_demand * headway / 2.0
            boarded[bus, stop] = boarded_count
            onboard[bus, stop] = current_load
            holding[bus, stop] = hold
            departures[bus, stop] = arrivals[bus, stop] + dwell + hold

    headways = np.diff(departures, axis=0)
    valid_headways = headways[:, 1:]
    runs = [
        BusRun(
            bus_id=bus,
            arrival_min=arrivals[bus],
            departure_min=departures[bus],
            onboard_after_stop=onboard[bus],
            boarded_by_stop=boarded[bus],
            holding_by_stop=holding[bus],
            wait_cost_by_stop=wait_cost[bus],
        )
        for bus in range(scenario.bus_count)
    ]
    total_boarded = max(1, int(np.sum(boarded)))
    total_wait = float(np.sum(wait_cost))
    headway_std = float(np.std(valid_headways))
    headway_mean = float(np.mean(valid_headways))
    bunching_count = int(np.sum(valid_headways < scenario.target_headway_min * 0.55))
    reliability = 100.0 - 8.0 * headway_std - 1.8 * bunching_count

    return HeadwayResult(
        strategy_name=strategy_name,
        runs=runs,
        headways_by_stop=headways,
        mean_headway_min=headway_mean,
        headway_std_min=headway_std,
        headway_cv=float(headway_std / max(headway_mean, 1e-9)),
        bunching_count=bunching_count,
        total_passenger_wait_min=total_wait,
        average_passenger_wait_min=float(total_wait / total_boarded),
        total_holding_min=float(np.sum(holding)),
        max_load=int(np.max(onboard)),
        service_reliability_score=float(max(0.0, reliability)),
    )


def compare_strategies(scenario: BusScenario) -> tuple[HeadwayResult, HeadwayResult]:
    baseline = simulate_corridor(scenario, "no_control")
    optimized = simulate_corridor(scenario, "adaptive_holding")
    return baseline, optimized
