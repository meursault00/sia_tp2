import random

def universal_selection(population):
    """
    Universal selection: selects individuals using Stochastic Universal Sampling (SUS).
    This method reduces stochastic variance by using equally spaced pointers on the 
    cumulative fitness distribution.

    """
    n = len(population)
    total_fitness = sum(ind.fitness for ind in population)
    
    # If total fitness is zero, fall back to returning a clone of every individual
    if total_fitness == 0:
        return [ind.clone() for ind in population]
    
    # Calculate the distance between pointers
    pointer_distance = total_fitness / n
    start = random.uniform(0, pointer_distance)
    
    # Generate pointers at equally spaced intervals
    pointers = [start + i * pointer_distance for i in range(n)]
    
    chosen = []
    cumulative_sum = 0
    pointer_index = 0
    
    # Traverse the population, accumulating fitness until pointers are reached
    for ind in population:
        cumulative_sum += ind.fitness
        while pointer_index < n and cumulative_sum >= pointers[pointer_index]:
            chosen.append(ind.clone())
            pointer_index += 1
            
    return chosen