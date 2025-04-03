# ga/crossover.py

import random

def one_point_crossover(parent1, parent2):
    """
    Cruza de un punto.
    Genes = lista de tuplas (triángulos).
    """
    child1 = parent1.clone()
    child2 = parent2.clone()

    # aplanar la lista de triángulos para mostrártelo
    # pero en este ejemplo, no es muy distinto hacerlo "por triángulo".
    p1_len = len(child1.genes)
    if p1_len <= 1:
        return child1, child2  # no hay nada que cruzar

    cut_point = random.randint(1, p1_len-1)
    
    # Intercambiamos los triángulos a partir del punto
    child1.genes[cut_point:], child2.genes[cut_point:] = (
        child2.genes[cut_point:],
        child1.genes[cut_point:]
    )

    child1.fitness = None
    child2.fitness = None
    return child1, child2

def two_point_crossover(parent1, parent2):
    """
    Cruza de dos puntos.
    """
    child1 = parent1.clone()
    child2 = parent2.clone()
    
    length = len(child1.genes)
    if length <= 2:
        return child1, child2

    p1 = random.randint(1, length-2)
    p2 = random.randint(p1+1, length-1)

    segment1 = child1.genes[p1:p2]
    segment2 = child2.genes[p1:p2]

    child1.genes[p1:p2] = segment2
    child2.genes[p1:p2] = segment1

    child1.fitness = None
    child2.fitness = None
    return child1, child2