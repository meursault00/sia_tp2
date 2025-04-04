import random

def roulette_selection(population, k=None):
    """
    Roulette (fitness-proportional) selection.
    """
    total_fitness = sum(ind.fitness for ind in population)
    # Evitar divisiones por cero
    if total_fitness == 0:
        # fallback: random or pass through
        return [random.choice(population).clone() for _ in range(k or len(population))]
    
    chosen = []
    for _ in range(k or len(population)):
        pick = random.uniform(0, total_fitness)
        current = 0
        for ind in population:
            current += ind.fitness
            if current >= pick:
                chosen.append(ind.clone())
                break
    return chosen