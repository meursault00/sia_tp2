import random

def universal_selection(population, k=None):
    """
    Universal selection: selects individuals using Stochastic Universal Sampling (SUS).
    This method reduces stochastic variance by using equally spaced pointers on the 
    cumulative fitness distribution.

    """
    if k is None:
        k = len(population)

    total_fitness = sum(ind.fitness for ind in population)
    
    # If total fitness is zero, fall back to random sampling
    if total_fitness == 0:
        return [random.choice(population).clone() for _ in range(k)]
    
    # Calculate the distance between pointers
    pointer_distance = total_fitness / k
    start = random.uniform(0, pointer_distance)
    
    # Generate pointers at equally spaced intervals
    pointers = [start + i * pointer_distance for i in range(k)]
    
    chosen = []
    cumulative_sum = 0
    pointer_index = 0
    
    # Traverse the population, accumulating fitness until pointers are reached
    for ind in population:
        cumulative_sum += ind.fitness
        while pointer_index < k and cumulative_sum >= pointers[pointer_index]:
            chosen.append(ind.clone())
            pointer_index += 1
            
    return chosen