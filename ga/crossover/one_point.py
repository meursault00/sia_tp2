import random

def one_point_crossover(parent1, parent2):
    child1 = parent1.clone()
    child2 = parent2.clone()

    length = len(child1.genes)
    if length <= 1:
        return child1, child2

    cut_point = random.randint(1, length-1)
    child1.genes[cut_point:], child2.genes[cut_point:] = child2.genes[cut_point:], child1.genes[cut_point:]
    
    child1.fitness = None
    child2.fitness = None
    return child1, child2