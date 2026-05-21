"""Traffic corridor scenario for eco-driving speed advice."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrafficLight:
    light_id: str
    position_m: float
    cycle_s: float
    green_s: float
    offset_s: float = 0.0

    def is_green(self, time_s: float) -> bool:
        phase = (time_s - self.offset_s) % self.cycle_s
        return 0.0 <= phase < self.green_s

    def next_green_window(self, earliest_time: float) -> tuple[float, float]:
        phase = (earliest_time - self.offset_s) % self.cycle_s
        cycle_start = earliest_time - phase
        green_start = cycle_start + self.offset_s
        green_end = green_start + self.green_s
        if phase < self.green_s:
            return earliest_time, green_end
        next_start = green_start + self.cycle_s
        return next_start, next_start + self.green_s


@dataclass(frozen=True)
class VehicleSpec:
    max_speed: float = 16.0
    min_speed: float = 4.0
    max_accel: float = 2.0
    max_decel: float = 3.0
    cruise_speed: float = 12.0


@dataclass(frozen=True)
class CorridorScenario:
    length_m: float
    lights: tuple[TrafficLight, ...]
    vehicle: VehicleSpec
    dt: float = 1.0


def make_demo_scenario() -> CorridorScenario:
    """Create a deterministic arterial road with offset signal lights."""
    lights = (
        TrafficLight("S1", 280.0, cycle_s=60.0, green_s=28.0, offset_s=5.0),
        TrafficLight("S2", 610.0, cycle_s=70.0, green_s=30.0, offset_s=18.0),
        TrafficLight("S3", 940.0, cycle_s=75.0, green_s=32.0, offset_s=34.0),
        TrafficLight("S4", 1260.0, cycle_s=65.0, green_s=26.0, offset_s=10.0),
    )
    return CorridorScenario(length_m=1500.0, lights=lights, vehicle=VehicleSpec())
