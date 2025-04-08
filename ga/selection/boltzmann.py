import math
import random
from .temperature.log_decay import log_decay

def boltzmann_selection(population, generation=0, k=None, T0=1.0):
    """
    Boltzmann selection: individuals selected with probability
    proportional to exp(fitness / T(generation)).
    """
    T = log_decay(generation, T0)
    k = k or len(population)

    # Calculate exp(fitness / T) for each individual
    exp_fitnesses = [math.exp(ind.fitness / T) for ind in population]
    total = sum(exp_fitnesses)

    if total == 0:
        return [random.choice(population).clone() for _ in range(k)]

    # Normalize probabilities
    probabilities = [f / total for f in exp_fitnesses]

    # Select based on those probabilities
    chosen = random.choices(population, weights=probabilities, k=k)
    return [ind.clone() for ind in chosen]