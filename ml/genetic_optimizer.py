"""
Genetic Algorithm Optimizer - Strategy Parameter Evolution.

D.E. Shaw-style evolutionary optimization:
- Optimize strategy parameters
- Evolve trading rules
- Multi-objective fitness (return, Sharpe, drawdown)
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass, field
import random
import copy

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """Individual in the genetic population."""
    genes: Dict[str, float]
    fitness: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    generation: int = 0


@dataclass
class EvolutionResult:
    """Result of genetic optimization."""
    best_individual: Individual
    best_params: Dict[str, float]
    generations_run: int
    fitness_history: List[float]
    population_diversity: float


class GeneticOptimizer:
    """
    Genetic Algorithm for strategy parameter optimization.

    Features:
    - Tournament selection
    - Uniform crossover
    - Gaussian mutation
    - Elitism preservation
    - Multi-objective fitness
    """

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        tournament_size: int = 3
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size

        self.population: List[Individual] = []
        self.best_ever: Optional[Individual] = None
        self.generation = 0

    def initialize_population(
        self,
        param_bounds: Dict[str, Tuple[float, float]]
    ) -> List[Individual]:
        """Create initial random population."""
        population = []

        for _ in range(self.population_size):
            genes = {}
            for param, (low, high) in param_bounds.items():
                genes[param] = random.uniform(low, high)

            individual = Individual(genes=genes)
            population.append(individual)

        self.population = population
        return population

    def evaluate_fitness(
        self,
        individual: Individual,
        fitness_function: Callable[[Dict[str, float]], Tuple[float, float, float]]
    ):
        """Evaluate fitness of an individual."""
        returns, sharpe, max_dd = fitness_function(individual.genes)

        # Multi-objective fitness
        # Maximize returns and Sharpe, minimize drawdown
        fitness = (
            0.3 * returns +
            0.5 * sharpe +
            0.2 * (1 - max_dd)  # Lower drawdown is better
        )

        individual.fitness = fitness
        individual.sharpe = sharpe
        individual.max_drawdown = max_dd

    def tournament_selection(self) -> Individual:
        """Select individual via tournament."""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Uniform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        child1_genes = {}
        child2_genes = {}

        for param in parent1.genes:
            if random.random() < 0.5:
                child1_genes[param] = parent1.genes[param]
                child2_genes[param] = parent2.genes[param]
            else:
                child1_genes[param] = parent2.genes[param]
                child2_genes[param] = parent1.genes[param]

        return (
            Individual(genes=child1_genes, generation=self.generation),
            Individual(genes=child2_genes, generation=self.generation)
        )

    def mutate(
        self,
        individual: Individual,
        param_bounds: Dict[str, Tuple[float, float]]
    ):
        """Gaussian mutation."""
        for param in individual.genes:
            if random.random() < self.mutation_rate:
                low, high = param_bounds[param]
                std = (high - low) * 0.1
                new_value = individual.genes[param] + random.gauss(0, std)
                individual.genes[param] = max(low, min(high, new_value))

    def evolve(
        self,
        fitness_function: Callable[[Dict[str, float]], Tuple[float, float, float]],
        param_bounds: Dict[str, Tuple[float, float]],
        generations: int = 100,
        early_stop_generations: int = 20
    ) -> EvolutionResult:
        """
        Run genetic evolution.

        Args:
            fitness_function: (params) -> (returns, sharpe, max_drawdown)
            param_bounds: Dict of param -> (min, max)
            generations: Max generations
            early_stop_generations: Stop if no improvement
        """
        # Initialize
        self.initialize_population(param_bounds)
        fitness_history = []
        no_improvement_count = 0

        for gen in range(generations):
            self.generation = gen

            # Evaluate fitness
            for individual in self.population:
                self.evaluate_fitness(individual, fitness_function)

            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            best = self.population[0]
            fitness_history.append(best.fitness)

            # Update best ever
            if self.best_ever is None or best.fitness > self.best_ever.fitness:
                self.best_ever = copy.deepcopy(best)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= early_stop_generations:
                logger.info(f"Early stopping at generation {gen}")
                break

            # Create new population
            new_population = []

            # Elitism
            for i in range(self.elite_size):
                elite = copy.deepcopy(self.population[i])
                new_population.append(elite)

            # Create offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()

                child1, child2 = self.crossover(parent1, parent2)

                self.mutate(child1, param_bounds)
                self.mutate(child2, param_bounds)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            self.population = new_population

        # Calculate diversity
        diversity = self._calculate_diversity()

        return EvolutionResult(
            best_individual=self.best_ever,
            best_params=self.best_ever.genes,
            generations_run=self.generation + 1,
            fitness_history=fitness_history,
            population_diversity=diversity
        )

    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if not self.population:
            return 0.0

        params = list(self.population[0].genes.keys())
        diversities = []

        for param in params:
            values = [ind.genes[param] for ind in self.population]
            diversity = np.std(values) / (np.mean(values) + 1e-10)
            diversities.append(diversity)

        return float(np.mean(diversities))


# Global singleton
_genetic_optimizer: Optional[GeneticOptimizer] = None


def get_genetic_optimizer() -> GeneticOptimizer:
    """Get or create global genetic optimizer."""
    global _genetic_optimizer
    if _genetic_optimizer is None:
        _genetic_optimizer = GeneticOptimizer()
    return _genetic_optimizer
