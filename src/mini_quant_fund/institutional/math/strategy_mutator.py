import logging
import random
import numpy as np

logger = logging.getLogger("StrategyMutator")

class SovereignStrategyMutator:
    """
    Genetic Algorithm for Strategy Mutation.
    "Breeds" new factor weights and thresholds based on evolutionary success.
    """
    def __init__(self):
        self.generation = 0

    def mutate_and_evolve(self, current_genome, success_metrics):
        self.generation += 1
        logger.info(f"[EVOLVE] Genetic Generation {self.generation} active.")
        
        # If success is high, keep genome stable. If low, mutate heavily.
        mutation_rate = 0.05 if np.mean(success_metrics) > 0 else 0.2
        
        new_genome = current_genome.copy()
        for key in new_genome:
            if isinstance(new_genome[key], (int, float)):
                new_genome[key] += np.random.normal(0, mutation_rate)
        
        logger.info(f"[GENETIC] Strategy mutated with rate: {mutation_rate:.1%}")
        return new_genome
