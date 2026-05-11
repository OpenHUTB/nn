"""Elevator group dispatch strategies and metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario import ElevatorScenario, PassengerRequest


@dataclass
class ElevatorState:
    available_s: float
    floor: int
    trips: int
    distance_floors: float
    passenger_seconds: float


@dataclass
class AssignmentRecord:
    request_id: int
    strategy_name: str
    elevator_id: int
    arrival_s: int
    pickup_s: float
    dropoff_s: float
    origin_floor: int
    destination_floor: int
    passengers: int
    wait_s: float
    ride_s: float
    empty_travel_floors: float


@dataclass
class DispatchResult:
    strategy_name: str
    records: list[AssignmentRecord]
    elevator_states: list[ElevatorState]
    mean_wait_s: float
    p90_wait_s: float
    max_wait_s: float
    mean_ride_s: float
    total_distance_floors: float
    empty_distance_floors: float
    energy_index: float
    long_wait_count: int
    passenger_throughput: int


def _travel_time(floors: float, scenario: ElevatorScenario) -> float:
    return floors * scenario.seconds_per_floor


def _direction(request: PassengerRequest) -> int:
    return 1 if request.destination_floor > request.origin_floor else -1


def _request_cost(
    elevator: ElevatorState,
    request: PassengerRequest,
    scenario: ElevatorScenario,
    current_time: float,
    strategy_name: str,
) -> float:
    available = max(elevator.available_s, current_time)
    empty_distance = abs(elevator.floor - request.origin_floor)
    pickup_time = available + _travel_time(empty_distance, scenario) + scenario.door_time_s
    wait_time = pickup_time - request.arrival_s
    loaded_distance = abs(request.destination_floor - request.origin_floor)
    trip_time = _travel_time(loaded_distance, scenario) + scenario.door_time_s

    if strategy_name == "nearest_car":
        busy_penalty = 0.018 * max(0.0, elevator.available_s - request.arrival_s)
        return empty_distance + busy_penalty + 0.012 * elevator.distance_floors

    direction_bonus = 0.0
    if elevator.available_s > request.arrival_s:
        direction_bonus = 5.0
    if request.origin_floor == 1 and request.destination_floor > 1:
        direction_bonus -= 8.0
    balance_penalty = 0.07 * elevator.distance_floors
    group_penalty = 0.8 * max(0, request.passengers - 1)
    return wait_time + 0.35 * trip_time + 1.2 * empty_distance + balance_penalty + group_penalty + direction_bonus


def _assign_request(
    elevator: ElevatorState,
    elevator_id: int,
    request: PassengerRequest,
    scenario: ElevatorScenario,
    strategy_name: str,
) -> AssignmentRecord:
    start_s = max(elevator.available_s, request.arrival_s)
    empty_distance = abs(elevator.floor - request.origin_floor)
    pickup_s = start_s + _travel_time(empty_distance, scenario) + scenario.door_time_s
    loaded_distance = abs(request.destination_floor - request.origin_floor)
    dropoff_s = pickup_s + _travel_time(loaded_distance, scenario) + scenario.door_time_s

    elevator.available_s = dropoff_s
    elevator.floor = request.destination_floor
    elevator.trips += 1
    elevator.distance_floors += empty_distance + loaded_distance
    elevator.passenger_seconds += request.passengers * (dropoff_s - pickup_s)

    return AssignmentRecord(
        request_id=request.request_id,
        strategy_name=strategy_name,
        elevator_id=elevator_id,
        arrival_s=request.arrival_s,
        pickup_s=pickup_s,
        dropoff_s=dropoff_s,
        origin_floor=request.origin_floor,
        destination_floor=request.destination_floor,
        passengers=request.passengers,
        wait_s=pickup_s - request.arrival_s,
        ride_s=dropoff_s - pickup_s,
        empty_travel_floors=empty_distance,
    )


def run_dispatch(scenario: ElevatorScenario, strategy_name: str) -> DispatchResult:
    """Run an elevator dispatch strategy on all passenger requests."""
    if strategy_name not in {"nearest_car", "destination_aware"}:
        raise ValueError(f"unknown dispatch strategy: {strategy_name}")

    elevators = [
        ElevatorState(available_s=0.0, floor=1, trips=0, distance_floors=0.0, passenger_seconds=0.0)
        for _ in range(scenario.elevator_count)
    ]
    records: list[AssignmentRecord] = []

    for request in sorted(scenario.requests, key=lambda item: (item.arrival_s, item.request_id)):
        scores = [
            _request_cost(elevator, request, scenario, float(request.arrival_s), strategy_name)
            for elevator in elevators
        ]
        elevator_id = int(np.argmin(scores))
        records.append(_assign_request(elevators[elevator_id], elevator_id, request, scenario, strategy_name))

    wait_values = np.array([record.wait_s for record in records])
    ride_values = np.array([record.ride_s for record in records])
    total_distance = float(sum(elevator.distance_floors for elevator in elevators))
    empty_distance = float(sum(record.empty_travel_floors for record in records))
    passenger_count = int(sum(record.passengers for record in records))
    energy_index = total_distance * 1.45 + empty_distance * 0.65 + passenger_count * 0.08

    return DispatchResult(
        strategy_name=strategy_name,
        records=records,
        elevator_states=elevators,
        mean_wait_s=float(np.mean(wait_values)),
        p90_wait_s=float(np.percentile(wait_values, 90)),
        max_wait_s=float(np.max(wait_values)),
        mean_ride_s=float(np.mean(ride_values)),
        total_distance_floors=total_distance,
        empty_distance_floors=empty_distance,
        energy_index=float(energy_index),
        long_wait_count=int(np.sum(wait_values > 90.0)),
        passenger_throughput=passenger_count,
    )


def compare_dispatchers(scenario: ElevatorScenario) -> tuple[DispatchResult, DispatchResult]:
    baseline = run_dispatch(scenario, "nearest_car")
    optimized = run_dispatch(scenario, "destination_aware")
    return baseline, optimized
