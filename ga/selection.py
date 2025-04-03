# ga/selection.py

import random

def tournament_selection(population, k=3):
    """
    Selección por torneo determinístico de tamaño k.
    Escoge k individuos al azar y se queda con el mejor.
    """
    chosen = []
    for _ in range(len(population)):
        competitors = random.sample(population, k)
        best = max(competitors, key=lambda ind: ind.fitness)
        chosen.append(best.clone())  # clonar para no mutar individuos originales
    return chosen