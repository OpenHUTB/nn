"""Genetic optimization for adaptive traffic signal timing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from scenario import IntersectionScenario, SignalPlan, webster_baseline
from simulator import SimulationResult, simulate_plan


@dataclass
class OptimizationHistory:
    generation: List[int]
    best_score: List[float]
    best_delay: List[float]
    best_queue: List[float]


@dataclass
class OptimizationResult:
    baseline: SimulationResult
    optimized: SimulationResult
    history: OptimizationHistory
    population_size: int
    generations: int


def genes_to_plan(scenario: IntersectionScenario, genes: np.ndarray) -> SignalPlan:
    cycle = int(np.clip(round(genes[0]), 60, 150))
    weights = np.clip(genes[1:], 0.05, None)
    available = cycle - 12
    greens = {}
    for approach, value in zip(scenario.approaches, weights / weights.sum() * available):
        greens[approach.name] = int(round(max(8, value)))
    return SignalPlan(cycle_length=cycle, greens=greens).normalized(scenario.approaches)


def plan_to_genes(scenario: IntersectionScenario, plan: SignalPlan) -> np.ndarray:
    greens = np.array([plan.green_for(name) for name in scenario.names], dtype=float)
    return np.concatenate([[float(plan.cycle_length)], greens / greens.sum()])


def evaluate_gene(scenario: IntersectionScenario, gene: np.ndarray) -> SimulationResult:
    return simulate_plan(scenario, genes_to_plan(scenario, gene))


def crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = rng.uniform(0.25, 0.75)
    child = alpha * parent_a + (1.0 - alpha) * parent_b
    if rng.random() < 0.35:
        cut = rng.integers(1, len(parent_a))
        child[:cut] = parent_a[:cut]
        child[cut:] = parent_b[cut:]
    return child


def mutate(gene: np.ndarray, rng: np.random.Generator, generation: int, total_generations: int) -> np.ndarray:
    scale = max(0.04, 0.18 * (1.0 - generation / max(total_generations, 1)))
    mutated = gene.copy()
    mutated[0] += rng.normal(0, 10 * scale)
    mutated[1:] += rng.normal(0, scale, size=len(gene) - 1)
    mutated[0] = np.clip(mutated[0], 60, 150)
    mutated[1:] = np.clip(mutated[1:], 0.05, 2.0)
    return mutated


def optimize_signal_plan(
    scenario: IntersectionScenario,
    population_size: int = 42,
    generations: int = 36,
    seed: int = 11,
) -> OptimizationResult:
    """Optimize the signal plan with elitist genetic search."""

    rng = np.random.default_rng(seed)
    baseline_plan = webster_baseline(scenario)
    baseline = simulate_plan(scenario, baseline_plan)
    base_gene = plan_to_genes(scenario, baseline_plan)
    population = [base_gene]

    while len(population) < population_size:
        gene = base_gene.copy()
        gene[0] = rng.uniform(65, 145)
        gene[1:] = rng.dirichlet(np.ones(len(scenario.approaches)) * 1.4)
        population.append(gene)

    history = OptimizationHistory([], [], [], [])
    best_result = baseline
    best_gene = base_gene

    for generation in range(generations):
        evaluated: List[Tuple[float, np.ndarray, SimulationResult]] = []
        for gene in population:
            result = evaluate_gene(scenario, gene)
            evaluated.append((result.score, gene, result))
        evaluated.sort(key=lambda item: item[0])
        if evaluated[0][0] < best_result.score:
            best_gene = evaluated[0][1].copy()
            best_result = evaluated[0][2]

        history.generation.append(generation)
        history.best_score.append(best_result.score)
        history.best_delay.append(best_result.average_delay)
        history.best_queue.append(best_result.max_queue)

        elites = [item[1] for item in evaluated[: max(4, population_size // 6)]]
        next_population = [elite.copy() for elite in elites]
        while len(next_population) < population_size:
            parent_indices = rng.choice(len(elites), size=2, replace=True)
            child = crossover(elites[parent_indices[0]], elites[parent_indices[1]], rng)
            child = mutate(child, rng, generation, generations)
            next_population.append(child)
        if generation % 6 == 0:
            next_population[-1] = mutate(best_gene, rng, generation, generations)
        population = next_population

    optimized = simulate_plan(scenario, genes_to_plan(scenario, best_gene))
    return OptimizationResult(
        baseline=baseline,
        optimized=optimized,
        history=history,
        population_size=population_size,
        generations=generations,
    )
