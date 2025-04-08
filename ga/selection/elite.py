def elite_selection(population, k=None):
    """
    Elite selection: selects the top fraction of the population based on fitness.

    """
    if k is None:
        k = len(population)
    
    # Sort the population by fitness (assuming higher fitness is better)
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
    
    # Select top-k
    return [ind.clone() for ind in sorted_pop[:k]]