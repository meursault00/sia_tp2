import random

def tournament_selection(population, k=None, tournament_size=3):
    """
    Deterministic tournament selection of size k.
    """
    chosen = []
    for _ in range(k or len(population)):
        competitors = random.sample(population, tournament_size)
        best = max(competitors, key=lambda ind: ind.fitness)
        chosen.append(best.clone())
    return chosen