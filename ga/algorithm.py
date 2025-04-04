# ga/algorithm.py

import random
from IPython.display import clear_output
import matplotlib.pyplot as plt

from .population import Population
from .fitness import compute_fitness
from .selection import selection_strategies
from .crossover import crossover_strategies
from .mutation import mutation_strategies
from utils.render import render_individual

def run_ga(config, target_image):
    """
    A variation of run_ga that displays intermediate images
    in a Jupyter notebook after each generation.
    """
    w, h = target_image.width, target_image.height
    N = config["population_size"]
    K = config["parents_size"]
    separation = config["separation_method"]

    # Retrieve functions from config
    selection_func = selection_strategies[config["selection_method"]]
    crossover_func = crossover_strategies[config["crossover_method"]]
    mutation_func  = mutation_strategies[config["mutation_method"]]

    # Create initial population
    population = Population(config, w, h)
    population.evaluate(compute_fitness, target_image)

    best = population.get_best()

    n_gens = config["n_generations"]
    for gen in range(n_gens):
        # Selection
        parents = selection_func(population.individuals, K)

        # Crossover
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
                offspring.append(parents[i].clone()) # In case there's an odd number of parents

        # Mutation
        for child in offspring:
            mutation_func(child, config["mutation_rate"], w, h)

        # #Evaluate offspring
        for child in offspring:
            child.fitness = compute_fitness(child, target_image)

        # Separation methods
        if separation == "traditional":
            # Combine entire old population + new children and select N
            combined = population.individuals + offspring
            combined.sort(key=lambda ind: ind.fitness, reverse=True) # Elitist selection of N children
            population.individuals = combined[:N]
        elif separation == "young_bias":
            # Use children, and if not enough grab from the best of the old population
            if len(offspring) >= N:
                offspring.sort(key=lambda ind: ind.fitness, reverse=True) # Elitist selection of N children
                population.individuals = offspring[:N]
            else:
                # Not enough children, fill with the best from the old population
                remaining = N - len(offspring)
                old_sorted = sorted(population.individuals, key=lambda ind: ind.fitness, reverse=True)
                population.individuals = offspring + old_sorted[:remaining]
        else:
            raise ValueError(f"Unknown separation method: {separation}")

        # Fitness evaluation of new generation
        population.evaluate(compute_fitness, target_image)

        current_best = population.get_best()
        if current_best.fitness > best.fitness:
            best = current_best.clone()

        # --- INTERACTIVE DISPLAY IN NOTEBOOK ---
        clear_output(wait=True)  # Clear previous cell output
        # Render the best so far
        best_img = render_individual(best, w, h)

        # Display using matplotlib
        plt.imshow(best_img)
        plt.axis("off")
        plt.title(f"Generation {gen+1}/{n_gens}, Fitness: {best.fitness:.6f}")
        plt.show()

    return best