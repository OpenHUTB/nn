"""Charging station baseline and load-aware scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario import ChargingRequest, ChargingScenario


@dataclass
class VehicleSchedule:
    vehicle_id: int
    arrival_min: int
    deadline_min: int
    start_min: int | None
    finish_min: int | None
    energy_kwh: float
    delivered_kwh: float
    waiting_min: int | None
    priority: int


@dataclass
class ScheduleResult:
    strategy_name: str
    time_min: np.ndarray
    station_load_kw: np.ndarray
    net_grid_load_kw: np.ndarray
    charger_occupancy: np.ndarray
    vehicle_schedules: list[VehicleSchedule]
    completed_count: int
    average_wait_min: float
    deadline_miss_count: int
    peak_grid_load_kw: float
    overload_minutes: int
    solar_utilization_percent: float
    load_variance: float


def _slot_count(scenario: ChargingScenario) -> int:
    return scenario.horizon_min // scenario.time_step_min


def _charge_slots(request: ChargingRequest, scenario: ChargingScenario) -> int:
    energy_per_slot = scenario.charger_power_kw * scenario.time_step_min / 60.0
    return int(np.ceil(request.energy_kwh / energy_per_slot))


def _available_run(occupancy: np.ndarray, start: int, duration: int, charger_count: int) -> bool:
    end = start + duration
    if end > len(occupancy):
        return False
    return bool(np.all(occupancy[start:end] < charger_count))


def _evaluate(
    strategy_name: str,
    scenario: ChargingScenario,
    occupancy: np.ndarray,
    station_load_kw: np.ndarray,
    schedules: list[VehicleSchedule],
) -> ScheduleResult:
    time_min = np.arange(0, scenario.horizon_min, scenario.time_step_min)
    net_grid_load = scenario.base_load_kw + station_load_kw - scenario.solar_kw
    completed = [item for item in schedules if item.finish_min is not None]
    waits = [item.waiting_min for item in completed if item.waiting_min is not None]
    misses = [
        item
        for item in completed
        if item.finish_min is not None and item.finish_min > item.deadline_min
    ]
    solar_used = np.minimum(station_load_kw, scenario.solar_kw)
    total_station_energy = np.sum(station_load_kw) * scenario.time_step_min / 60.0
    solar_energy = np.sum(solar_used) * scenario.time_step_min / 60.0

    return ScheduleResult(
        strategy_name=strategy_name,
        time_min=time_min,
        station_load_kw=station_load_kw,
        net_grid_load_kw=net_grid_load,
        charger_occupancy=occupancy,
        vehicle_schedules=schedules,
        completed_count=len(completed),
        average_wait_min=float(np.mean(waits)) if waits else 0.0,
        deadline_miss_count=len(misses),
        peak_grid_load_kw=float(np.max(net_grid_load)),
        overload_minutes=int(np.sum(net_grid_load > scenario.grid_limit_kw) * scenario.time_step_min),
        solar_utilization_percent=float(solar_energy / max(total_station_energy, 1e-9) * 100.0),
        load_variance=float(np.var(net_grid_load)),
    )


def schedule_first_come_first_served(scenario: ChargingScenario) -> ScheduleResult:
    """Schedule vehicles immediately when a charger becomes available."""
    slots = _slot_count(scenario)
    occupancy = np.zeros(slots, dtype=int)
    station_load = np.zeros(slots)
    schedules: list[VehicleSchedule] = []

    for request in sorted(scenario.requests, key=lambda item: (item.arrival_min, item.vehicle_id)):
        duration = _charge_slots(request, scenario)
        earliest = request.arrival_min // scenario.time_step_min
        start_slot = earliest
        while start_slot + duration <= slots and not _available_run(occupancy, start_slot, duration, scenario.charger_count):
            start_slot += 1

        if start_slot + duration > slots:
            schedules.append(_missed_schedule(request))
            continue

        occupancy[start_slot : start_slot + duration] += 1
        station_load[start_slot : start_slot + duration] += scenario.charger_power_kw
        start_min = start_slot * scenario.time_step_min
        finish_min = (start_slot + duration) * scenario.time_step_min
        schedules.append(_finished_schedule(request, start_min, finish_min, scenario))

    return _evaluate("first_come_first_served", scenario, occupancy, station_load, schedules)


def _missed_schedule(request: ChargingRequest) -> VehicleSchedule:
    return VehicleSchedule(
        vehicle_id=request.vehicle_id,
        arrival_min=request.arrival_min,
        deadline_min=request.deadline_min,
        start_min=None,
        finish_min=None,
        energy_kwh=request.energy_kwh,
        delivered_kwh=0.0,
        waiting_min=None,
        priority=request.priority,
    )


def _finished_schedule(request: ChargingRequest, start_min: int, finish_min: int, scenario: ChargingScenario) -> VehicleSchedule:
    return VehicleSchedule(
        vehicle_id=request.vehicle_id,
        arrival_min=request.arrival_min,
        deadline_min=request.deadline_min,
        start_min=start_min,
        finish_min=finish_min,
        energy_kwh=request.energy_kwh,
        delivered_kwh=request.energy_kwh,
        waiting_min=start_min - request.arrival_min,
        priority=request.priority,
    )


def _score_window(scenario: ChargingScenario, station_load: np.ndarray, occupancy: np.ndarray, start: int, duration: int, request: ChargingRequest) -> float:
    end = start + duration
    added_load = station_load[start:end] + scenario.charger_power_kw
    net_load = scenario.base_load_kw[start:end] + added_load - scenario.solar_kw[start:end]
    overload = np.maximum(0.0, net_load - scenario.grid_limit_kw)
    solar_match = np.minimum(scenario.charger_power_kw, scenario.solar_kw[start:end])
    waiting = start * scenario.time_step_min - request.arrival_min
    finish = end * scenario.time_step_min
    deadline_penalty = max(0, finish - request.deadline_min)
    occupancy_penalty = np.mean(occupancy[start:end]) * 1.8
    return float(
        1.25 * np.mean(net_load)
        + 6.5 * np.mean(overload)
        + 0.18 * waiting / max(request.priority, 1)
        + 4.0 * deadline_penalty
        + occupancy_penalty
        - 0.65 * np.mean(solar_match)
    )


def schedule_load_aware(scenario: ChargingScenario) -> ScheduleResult:
    """Schedule charging windows by balancing waiting, deadlines, solar and grid peaks."""
    slots = _slot_count(scenario)
    occupancy = np.zeros(slots, dtype=int)
    station_load = np.zeros(slots)
    schedules: list[VehicleSchedule] = []

    ordered = sorted(
        scenario.requests,
        key=lambda item: (
            item.deadline_min,
            -item.priority,
            item.arrival_min,
            item.energy_kwh,
        ),
    )

    for request in ordered:
        duration = _charge_slots(request, scenario)
        earliest = request.arrival_min // scenario.time_step_min
        latest = min(slots - duration, max(earliest, request.deadline_min // scenario.time_step_min - duration))
        candidates: list[tuple[float, int]] = []
        for start_slot in range(earliest, latest + 1):
            if _available_run(occupancy, start_slot, duration, scenario.charger_count):
                candidates.append(
                    (
                        _score_window(scenario, station_load, occupancy, start_slot, duration, request),
                        start_slot,
                    )
                )

        if not candidates:
            schedules.append(_missed_schedule(request))
            continue

        _, best_start = min(candidates, key=lambda item: item[0])
        occupancy[best_start : best_start + duration] += 1
        station_load[best_start : best_start + duration] += scenario.charger_power_kw
        start_min = best_start * scenario.time_step_min
        finish_min = (best_start + duration) * scenario.time_step_min
        schedules.append(_finished_schedule(request, start_min, finish_min, scenario))

    schedules.sort(key=lambda item: item.vehicle_id)
    return _evaluate("load_aware_scheduler", scenario, occupancy, station_load, schedules)


def compare_schedulers(scenario: ChargingScenario) -> tuple[ScheduleResult, ScheduleResult]:
    baseline = schedule_first_come_first_served(scenario)
    optimized = schedule_load_aware(scenario)
    return baseline, optimized
