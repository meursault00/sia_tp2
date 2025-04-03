import random

def uniform_crossover(parent1, parent2, swap_prob=0.5):
    """
    For each gene, swap with probability swap_prob.
    """
    child1 = parent1.clone()
    child2 = parent2.clone()

    length = len(child1.genes)
    for i in range(length):
        if random.random() < swap_prob:
            child1.genes[i], child2.genes[i] = child2.genes[i], child1.genes[i]

    child1.fitness = None
    child2.fitness = None
    return child1, child2