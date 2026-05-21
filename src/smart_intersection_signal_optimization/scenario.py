"""Scenario definitions for adaptive intersection signal optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class Approach:
    """One incoming road direction at the intersection."""

    name: str
    demand_vph: float
    saturation_vph: float
    priority: float = 1.0

    @property
    def demand_per_second(self) -> float:
        return self.demand_vph / 3600.0

    @property
    def service_per_second(self) -> float:
        return self.saturation_vph / 3600.0


@dataclass(frozen=True)
class SignalPlan:
    """Signal plan represented by cycle length and green splits."""

    cycle_length: int
    greens: Dict[str, int]
    lost_time: int = 12

    def normalized(self, approaches: Iterable[Approach]) -> "SignalPlan":
        names = [approach.name for approach in approaches]
        minimum_green = 8
        available = max(minimum_green * len(names), self.cycle_length - self.lost_time)
        raw_total = sum(max(minimum_green, int(self.greens.get(name, minimum_green))) for name in names)
        scaled: Dict[str, int] = {}
        remaining = available
        for index, name in enumerate(names):
            if index == len(names) - 1:
                scaled[name] = max(minimum_green, remaining)
            else:
                value = round(max(minimum_green, self.greens.get(name, minimum_green)) / raw_total * available)
                value = max(minimum_green, min(value, remaining - minimum_green * (len(names) - index - 1)))
                scaled[name] = value
                remaining -= value
        return SignalPlan(cycle_length=int(self.cycle_length), greens=scaled, lost_time=self.lost_time)

    def green_for(self, name: str) -> int:
        return int(self.greens[name])


@dataclass(frozen=True)
class IntersectionScenario:
    """A deterministic four-arm intersection scenario."""

    approaches: List[Approach]
    horizon_seconds: int = 1800
    time_step: int = 1

    @property
    def names(self) -> List[str]:
        return [approach.name for approach in self.approaches]

    def approach_map(self) -> Dict[str, Approach]:
        return {approach.name: approach for approach in self.approaches}


def make_demo_scenario() -> IntersectionScenario:
    """Create a peak-hour intersection with unbalanced traffic demand."""

    return IntersectionScenario(
        approaches=[
            Approach("north_south_through", demand_vph=1120, saturation_vph=1850, priority=1.15),
            Approach("east_west_through", demand_vph=860, saturation_vph=1800, priority=1.00),
            Approach("north_south_left", demand_vph=310, saturation_vph=1450, priority=1.30),
            Approach("east_west_left", demand_vph=240, saturation_vph=1400, priority=1.10),
        ],
        horizon_seconds=1800,
        time_step=1,
    )


def webster_baseline(scenario: IntersectionScenario) -> SignalPlan:
    """Generate a Webster-style fixed-time baseline plan."""

    lost_time = 12
    flow_ratios = [approach.demand_vph / approach.saturation_vph for approach in scenario.approaches]
    total_ratio = min(0.92, sum(flow_ratios))
    cycle = int(round((1.5 * lost_time + 5) / max(0.08, 1 - total_ratio)))
    cycle = max(60, min(150, cycle))
    available = cycle - lost_time
    greens = {
        approach.name: max(8, round(available * ratio / sum(flow_ratios)))
        for approach, ratio in zip(scenario.approaches, flow_ratios)
    }
    return SignalPlan(cycle_length=cycle, greens=greens, lost_time=lost_time).normalized(scenario.approaches)
