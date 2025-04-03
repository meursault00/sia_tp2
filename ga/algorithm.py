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
        parents = selection_func(population.individuals)

        # Crossover
        new_generation = []
        for i in range(0, len(parents), 2):
            if i+1 < len(parents):
                p1, p2 = parents[i], parents[i+1]
                if random.random() < config["crossover_rate"]:
                    c1, c2 = crossover_func(p1, p2)
                else:
                    c1, c2 = p1.clone(), p2.clone()
                new_generation.append(c1)
                new_generation.append(c2)
            else:
                new_generation.append(parents[i].clone())

        # Mutation
        for ind in new_generation:
            mutation_func(ind, config["mutation_rate"], w, h)

        # Replacing the entire population
        population.individuals = new_generation
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