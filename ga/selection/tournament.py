import random

def tournament_selection(population, k=3):
    """
    Deterministic tournament selection of size k.
    """
    chosen = []
    for _ in range(len(population)):
        competitors = random.sample(population, k)
        best = max(competitors, key=lambda ind: ind.fitness)
        chosen.append(best.clone())
    return chosen