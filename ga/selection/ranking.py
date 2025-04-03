import random

def ranking_selection(population):
    """
    Ranking-based selection. Simplificada: se ordena por fitness
    y se asocia una pseudo-aptitud. Luego, se hace ruleta con esa pseudo-aptitud.
    """
    # Ordenamos la población de peor a mejor
    sorted_pop = sorted(population, key=lambda x: x.fitness)
    n = len(population)
    
    # Pseudo-fitness: i+1 para el i-ésimo en la lista
    # (eso le da fitness 1 al peor y n al mejor)
    pseudo_fitness_list = [i+1 for i in range(n)]
    total_pf = sum(pseudo_fitness_list)
    
    chosen = []
    for _ in range(n):
        pick = random.uniform(0, total_pf)
        current = 0
        for idx, ind in enumerate(sorted_pop):
            current += (idx+1)
            if current >= pick:
                chosen.append(ind.clone())
                break
    return chosen