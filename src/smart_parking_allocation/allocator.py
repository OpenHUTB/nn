"""Parking allocation baselines and congestion-aware optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario import ParkingScenario, ParkingSpot, VehicleRequest, manhattan_distance


@dataclass
class Assignment:
    vehicle_id: int
    strategy_name: str
    spot_id: int | None
    zone: str | None
    arrival_min: int
    departure_min: int | None
    walking_distance_m: float | None
    entry_drive_m: float | None
    exit_drive_m: float | None
    waiting_min: int
    needs_charger: bool
    needs_accessible: bool


@dataclass
class AllocationResult:
    strategy_name: str
    assignments: list[Assignment]
    occupancy_by_zone: dict[str, np.ndarray]
    total_occupancy: np.ndarray
    served_count: int
    rejected_count: int
    average_walking_distance_m: float
    p90_walking_distance_m: float
    average_wait_min: float
    charger_mismatch_count: int
    accessible_mismatch_count: int
    peak_zone_occupancy_ratio: float
    zone_imbalance_index: float
    traffic_distance_m: float


def _time_slots(scenario: ParkingScenario) -> int:
    return scenario.horizon_min // scenario.time_step_min


def _spot_point(spot: ParkingSpot) -> tuple[float, float]:
    return (spot.x_m, spot.y_m)


def _is_free(occupancy: dict[int, np.ndarray], spot: ParkingSpot, start_slot: int, end_slot: int) -> bool:
    return bool(np.all(occupancy[spot.spot_id][start_slot:end_slot] == 0))


def _eligible_spots(request: VehicleRequest, scenario: ParkingScenario) -> list[ParkingSpot]:
    candidates = []
    for spot in scenario.spots:
        if request.needs_accessible and not spot.is_accessible:
            continue
        if request.needs_charger and not spot.has_charger:
            continue
        candidates.append(spot)
    return candidates


def _zone_counts(spots: list[ParkingSpot]) -> dict[str, int]:
    zones = sorted({spot.zone for spot in spots})
    return {zone: sum(1 for spot in spots if spot.zone == zone) for zone in zones}


def _zone_load(occupancy_by_zone: dict[str, np.ndarray], zone_counts: dict[str, int], zone: str, start: int, end: int) -> float:
    if end <= start:
        return 0.0
    return float(np.mean(occupancy_by_zone[zone][start:end] / max(zone_counts[zone], 1)))


def _score_spot(
    strategy_name: str,
    request: VehicleRequest,
    spot: ParkingSpot,
    scenario: ParkingScenario,
    occupancy_by_zone: dict[str, np.ndarray],
    zone_counts: dict[str, int],
    start_slot: int,
    end_slot: int,
) -> float:
    entry_distance = manhattan_distance(scenario.entrance_xy, _spot_point(spot))
    exit_distance = manhattan_distance(_spot_point(spot), scenario.exit_xy)
    walking_distance = spot.distance_to_exit_m
    if strategy_name == "nearest_spot":
        return entry_distance + 0.35 * walking_distance

    zone_load = _zone_load(occupancy_by_zone, zone_counts, spot.zone, start_slot, end_slot)
    charger_bonus = -18.0 if request.needs_charger and spot.has_charger else 0.0
    accessible_bonus = -22.0 if request.needs_accessible and spot.is_accessible else 0.0
    priority_weight = 1.0 / max(request.priority, 1)
    return (
        0.42 * entry_distance
        + 0.50 * walking_distance * priority_weight
        + 0.20 * exit_distance
        + 72.0 * zone_load
        + charger_bonus
        + accessible_bonus
    )


def _evaluate(
    strategy_name: str,
    scenario: ParkingScenario,
    assignments: list[Assignment],
    occupancy_by_zone: dict[str, np.ndarray],
) -> AllocationResult:
    total_occupancy = np.sum(np.vstack(list(occupancy_by_zone.values())), axis=0)
    served = [item for item in assignments if item.spot_id is not None]
    walk_values = [item.walking_distance_m for item in served if item.walking_distance_m is not None]
    wait_values = [item.waiting_min for item in served]
    zone_counts = _zone_counts(scenario.spots)
    ratios = []
    for zone, occupancy in occupancy_by_zone.items():
        ratios.append(occupancy / max(zone_counts[zone], 1))
    ratio_matrix = np.vstack(ratios)
    imbalance = float(np.mean(np.std(ratio_matrix, axis=0)))
    charger_mismatch = sum(item.needs_charger and item.spot_id is None for item in assignments)
    accessible_mismatch = sum(item.needs_accessible and item.spot_id is None for item in assignments)
    traffic_distance = sum((item.entry_drive_m or 0.0) + (item.exit_drive_m or 0.0) for item in served)

    return AllocationResult(
        strategy_name=strategy_name,
        assignments=assignments,
        occupancy_by_zone=occupancy_by_zone,
        total_occupancy=total_occupancy,
        served_count=len(served),
        rejected_count=len(assignments) - len(served),
        average_walking_distance_m=float(np.mean(walk_values)) if walk_values else 0.0,
        p90_walking_distance_m=float(np.percentile(walk_values, 90)) if walk_values else 0.0,
        average_wait_min=float(np.mean(wait_values)) if wait_values else 0.0,
        charger_mismatch_count=int(charger_mismatch),
        accessible_mismatch_count=int(accessible_mismatch),
        peak_zone_occupancy_ratio=float(np.max(ratio_matrix)),
        zone_imbalance_index=imbalance,
        traffic_distance_m=float(traffic_distance),
    )


def allocate_parking(scenario: ParkingScenario, strategy_name: str) -> AllocationResult:
    """Allocate spaces using either nearest-space or balanced optimization."""
    if strategy_name not in {"nearest_spot", "balanced_allocation"}:
        raise ValueError(f"unknown allocation strategy: {strategy_name}")

    slots = _time_slots(scenario)
    occupancy = {spot.spot_id: np.zeros(slots, dtype=int) for spot in scenario.spots}
    zones = sorted({spot.zone for spot in scenario.spots})
    occupancy_by_zone = {zone: np.zeros(slots, dtype=int) for zone in zones}
    zone_counts = _zone_counts(scenario.spots)
    assignments: list[Assignment] = []

    for request in sorted(scenario.requests, key=lambda item: (item.arrival_min, -item.priority, item.vehicle_id)):
        duration = max(1, int(np.ceil(request.dwell_min / scenario.time_step_min)))
        arrival_slot = request.arrival_min // scenario.time_step_min
        selected: tuple[float, ParkingSpot, int] | None = None
        wait_limit_slots = 4 if request.priority >= 2 else 8
        for start_slot in range(arrival_slot, min(slots - duration, arrival_slot + wait_limit_slots) + 1):
            end_slot = start_slot + duration
            for spot in _eligible_spots(request, scenario):
                if not _is_free(occupancy, spot, start_slot, end_slot):
                    continue
                score = _score_spot(strategy_name, request, spot, scenario, occupancy_by_zone, zone_counts, start_slot, end_slot)
                if selected is None or score < selected[0]:
                    selected = (score, spot, start_slot)

        if selected is None:
            assignments.append(
                Assignment(
                    vehicle_id=request.vehicle_id,
                    strategy_name=strategy_name,
                    spot_id=None,
                    zone=None,
                    arrival_min=request.arrival_min,
                    departure_min=None,
                    walking_distance_m=None,
                    entry_drive_m=None,
                    exit_drive_m=None,
                    waiting_min=0,
                    needs_charger=request.needs_charger,
                    needs_accessible=request.needs_accessible,
                )
            )
            continue

        _, spot, start_slot = selected
        end_slot = start_slot + duration
        occupancy[spot.spot_id][start_slot:end_slot] = 1
        occupancy_by_zone[spot.zone][start_slot:end_slot] += 1
        entry_distance = manhattan_distance(scenario.entrance_xy, _spot_point(spot))
        exit_distance = manhattan_distance(_spot_point(spot), scenario.exit_xy)
        assignments.append(
            Assignment(
                vehicle_id=request.vehicle_id,
                strategy_name=strategy_name,
                spot_id=spot.spot_id,
                zone=spot.zone,
                arrival_min=request.arrival_min,
                departure_min=end_slot * scenario.time_step_min,
                walking_distance_m=spot.distance_to_exit_m,
                entry_drive_m=entry_distance,
                exit_drive_m=exit_distance,
                waiting_min=start_slot * scenario.time_step_min - request.arrival_min,
                needs_charger=request.needs_charger,
                needs_accessible=request.needs_accessible,
            )
        )

    return _evaluate(strategy_name, scenario, assignments, occupancy_by_zone)


def compare_allocators(scenario: ParkingScenario) -> tuple[AllocationResult, AllocationResult]:
    baseline = allocate_parking(scenario, "nearest_spot")
    optimized = allocate_parking(scenario, "balanced_allocation")
    return baseline, optimized
