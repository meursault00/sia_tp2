import random

def roulette_selection(population):
    """
    Roulette (fitness-proportional) selection.
    """
    total_fitness = sum(ind.fitness for ind in population)
    # Evitar divisiones por cero
    if total_fitness == 0:
        # fallback: random or pass through
        return [ind.clone() for ind in population]
    
    chosen = []
    for _ in range(len(population)):
        pick = random.uniform(0, total_fitness)
        current = 0
        for ind in population:
            current += ind.fitness
            if current >= pick:
                chosen.append(ind.clone())
                break
    return chosen