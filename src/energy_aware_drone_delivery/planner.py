"""Energy-aware planning for drone delivery."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count

from scenario import Cell, DeliveryScenario, DeliveryTask, ChargingStation, manhattan


@dataclass
class Segment:
    label: str
    start: Cell
    goal: Cell
    path: list[Cell]
    energy: float
    payload_kg: float
    recharged: bool = False

    @property
    def steps(self) -> int:
        return max(0, len(self.path) - 1)


@dataclass
class MissionPlan:
    segments: list[Segment] = field(default_factory=list)
    task_order: list[str] = field(default_factory=list)
    charges: int = 0
    total_energy: float = 0.0
    total_steps: int = 0
    feasible: bool = True

    @property
    def route(self) -> list[Cell]:
        route: list[Cell] = []
        for segment in self.segments:
            if not route:
                route.extend(segment.path)
            else:
                route.extend(segment.path[1:])
        return route


def movement_energy(scenario: DeliveryScenario, current: Cell, nxt: Cell, payload_kg: float) -> float:
    d_row = nxt[0] - current[0]
    d_col = nxt[1] - current[1]
    wind_u = scenario.wind_u[current]
    wind_v = scenario.wind_v[current]
    headwind = -(wind_v * d_row + wind_u * d_col)
    wind_penalty = max(0.0, headwind) * scenario.drone.wind_cost
    tailwind_bonus = max(0.0, -headwind) * scenario.drone.wind_cost * 0.35
    return max(0.25, scenario.drone.base_cost + payload_kg * scenario.drone.payload_cost + wind_penalty - tailwind_bonus)


def astar_energy_path(scenario: DeliveryScenario, start: Cell, goal: Cell, payload_kg: float) -> tuple[list[Cell], float]:
    queue: list[tuple[float, int, Cell]] = []
    tie = count()
    heappush(queue, (0.0, next(tie), start))
    came_from: dict[Cell, Cell] = {}
    cost_so_far = {start: 0.0}
    visited: set[Cell] = set()

    while queue:
        _, _, current = heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            break
        for nxt in scenario.neighbors(current):
            new_cost = cost_so_far[current] + movement_energy(scenario, current, nxt, payload_kg)
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + manhattan(nxt, goal) * scenario.drone.base_cost
                came_from[nxt] = current
                heappush(queue, (priority, next(tie), nxt))

    if goal not in cost_so_far:
        return [], float("inf")
    return reconstruct(came_from, start, goal), cost_so_far[goal]


def reconstruct(came_from: dict[Cell, Cell], start: Cell, goal: Cell) -> list[Cell]:
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def score_task(scenario: DeliveryScenario, current: Cell, task: DeliveryTask) -> float:
    to_pickup, e1 = astar_energy_path(scenario, current, task.pickup, payload_kg=0.0)
    to_dropoff, e2 = astar_energy_path(scenario, task.pickup, task.dropoff, payload_kg=task.payload_kg)
    if not to_pickup or not to_dropoff:
        return float("inf")
    return e1 + e2 - task.priority * 3.5


def nearest_station(scenario: DeliveryScenario, current: Cell) -> tuple[ChargingStation, list[Cell], float]:
    candidates: list[tuple[float, ChargingStation, list[Cell]]] = []
    for station in scenario.stations:
        path, energy = astar_energy_path(scenario, current, station.cell, payload_kg=0.0)
        if path:
            candidates.append((energy, station, path))
    if not candidates:
        raise ValueError("no reachable charging station")
    energy, station, path = min(candidates, key=lambda item: item[0])
    return station, path, energy


def plan_mission(scenario: DeliveryScenario) -> MissionPlan:
    remaining = list(scenario.tasks)
    current = scenario.drone.start
    battery = scenario.drone.battery_capacity
    plan = MissionPlan()

    while remaining:
        task = min(remaining, key=lambda item: score_task(scenario, current, item))
        pickup_path, pickup_energy = astar_energy_path(scenario, current, task.pickup, payload_kg=0.0)
        dropoff_path, dropoff_energy = astar_energy_path(scenario, task.pickup, task.dropoff, payload_kg=task.payload_kg)
        if not pickup_path or not dropoff_path:
            plan.feasible = False
            break

        needed = pickup_energy + dropoff_energy
        if battery - needed < scenario.drone.battery_capacity * scenario.drone.reserve_ratio:
            station, charge_path, charge_energy = nearest_station(scenario, current)
            if battery - charge_energy < scenario.drone.battery_capacity * scenario.drone.reserve_ratio:
                plan.feasible = False
                break
            segment = Segment(f"recharge at {station.station_id}", current, station.cell, charge_path, charge_energy, 0.0, True)
            add_segment(plan, segment)
            current = station.cell
            battery = scenario.drone.battery_capacity
            plan.charges += 1
            continue

        pickup_segment = Segment(f"to {task.task_id} pickup", current, task.pickup, pickup_path, pickup_energy, 0.0)
        dropoff_segment = Segment(f"deliver {task.task_id}", task.pickup, task.dropoff, dropoff_path, dropoff_energy, task.payload_kg)
        add_segment(plan, pickup_segment)
        add_segment(plan, dropoff_segment)
        battery -= needed
        current = task.dropoff
        plan.task_order.append(task.task_id)
        remaining.remove(task)

    return plan


def add_segment(plan: MissionPlan, segment: Segment) -> None:
    plan.segments.append(segment)
    plan.total_energy += segment.energy
    plan.total_steps += segment.steps


def naive_distance_plan(scenario: DeliveryScenario) -> MissionPlan:
    """Baseline: ignore energy and wind, visit tasks by task id without charging."""
    current = scenario.drone.start
    plan = MissionPlan()
    for task in sorted(scenario.tasks, key=lambda item: item.task_id):
        pickup_path, pickup_energy = astar_energy_path(scenario, current, task.pickup, payload_kg=0.0)
        dropoff_path, dropoff_energy = astar_energy_path(scenario, task.pickup, task.dropoff, payload_kg=task.payload_kg)
        add_segment(plan, Segment(f"to {task.task_id} pickup", current, task.pickup, pickup_path, pickup_energy, 0.0))
        add_segment(plan, Segment(f"deliver {task.task_id}", task.pickup, task.dropoff, dropoff_path, dropoff_energy, task.payload_kg))
        current = task.dropoff
        plan.task_order.append(task.task_id)
    return plan


def battery_trace(scenario: DeliveryScenario, plan: MissionPlan) -> list[float]:
    battery = scenario.drone.battery_capacity
    trace = [battery]
    for segment in plan.segments:
        if segment.recharged:
            battery -= segment.energy
            trace.append(battery)
            battery = scenario.drone.battery_capacity
            trace.append(battery)
        else:
            battery -= segment.energy
            trace.append(battery)
    return trace
