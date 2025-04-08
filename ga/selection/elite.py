def elite_selection(population, elite_rate):
    """
    Elite selection: selects the top fraction of the population based on fitness.

    """
    n = len(population)
    elite_count = max(1, int(elite_rate * n))
    
    # Sort the population by fitness (assuming higher fitness is better)
    sorted_pop = sorted(population, key=lambda x: x.fitness)
    
    # Select the best elite_count individuals (from the end of the sorted list)
    chosen = [ind.clone() for ind in sorted_pop[-elite_count:]]
    return chosen