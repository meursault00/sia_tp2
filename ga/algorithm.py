import random
from .population import Population
from .fitness import compute_fitness
from .selection import selection_strategies
from .crossover import crossover_strategies
from .mutation import mutation_strategies

def run_ga(config, target_image):
    """
    Runs the genetic algorithm on the given patch (target_image) without rendering every generation.
    Logs progress to the console and returns the best individual.
    """
    w, h = target_image.width, target_image.height
    N = config["population_size"]
    K = config.get("parents_size", N)
    separation = config.get("separation_method", "traditional")

    selection_func = selection_strategies[config["selection_method"]]
    crossover_func = crossover_strategies[config["crossover_method"]]
    mutation_func  = mutation_strategies[config["mutation_method"]]

    population = Population(config, w, h)
    population.evaluate(compute_fitness, target_image)
    best = population.get_best()

    n_gens = config["n_generations"]
    for gen in range(n_gens):
        # Selection of parents.
        parents = selection_func(population.individuals, K)
        offspring = []
        for i in range(0, len(parents), 2):
            if i+1 < len(parents):
                p1, p2 = parents[i], parents[i+1]
                if random.random() < config["crossover_rate"]:
                    c1, c2 = crossover_func(p1, p2)
                else:
                    c1, c2 = p1.clone(), p2.clone()
                offspring.append(c1)
                offspring.append(c2)
            else:
                offspring.append(parents[i].clone())

        for child in offspring:
            mutation_func(child, config["mutation_rate"], w, h)

        for child in offspring:
            child.fitness = compute_fitness(child, target_image)

        if separation == "traditional":
            combined = population.individuals + offspring
            combined.sort(key=lambda ind: ind.fitness, reverse=True)
            population.individuals = combined[:N]
        elif separation == "young_bias":
            if len(offspring) >= N:
                offspring.sort(key=lambda ind: ind.fitness, reverse=True)
                population.individuals = offspring[:N]
            else:
                remaining = N - len(offspring)
                old_sorted = sorted(population.individuals, key=lambda ind: ind.fitness, reverse=True)
                population.individuals = offspring + old_sorted[:remaining]
        else:
            raise ValueError(f"Unknown separation method: {separation}")

        population.evaluate(compute_fitness, target_image)
        current_best = population.get_best()
        if current_best.fitness > best.fitness:
            best = current_best.clone()

        # Log progress without rendering
        print(f"Generation {gen+1}/{n_gens}: Best fitness = {best.fitness:.6f}")

    return best
