import random

def two_point_crossover(parent1, parent2):
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