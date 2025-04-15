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

def probabilistic_tournament_selection(population, k=None, tournament_size=3):
    """
    Probabilistic tournament selection of size k.
    Selects individuals with probability proportional to fitness within each tournament.
    """
    chosen = []
    k = k or len(population)
    
    for _ in range(k):
        competitors = random.sample(population, min(tournament_size, len(population)))
        fitnesses = [ind.fitness for ind in competitors]
        total_fitness = sum(fitnesses)
        if total_fitness > 0:
            probabilities = [f / total_fitness for f in fitnesses]
        else:
            probabilities = [1.0 / len(competitors)] * len(competitors)
        selected = random.choices(competitors, weights=probabilities, k=1)[0]
        chosen.append(selected.clone())
    
    return chosen