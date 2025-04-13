import random
try:
    from IPython.display import clear_output
except ImportError:
    clear_output = None
import matplotlib.pyplot as plt
from .population import Population
from .selection import selection_strategies
from .crossover import crossover_strategies
from .mutation import mutation_strategies
from utils.render import render_individual

def load_fitness_function(config):
    mode = config.get("fitness_mode", "default")
    if mode == "interpolated":
        from .fitness.interpolation import compute_fitness
    else:
        from .fitness.default import compute_fitness
    return compute_fitness

def run_ga(config, target_image, global_target=None):
    """
    Runs the genetic algorithm on a target_image patch.
    Uses global_target as the source for color sampling.
    """
    compute_fitness = load_fitness_function(config)
    w, h = target_image.width, target_image.height
    N = config["population_size"]
    K = config.get("parents_size", N)
    separation = config.get("separation_method", "traditional")
    disable_display = config.get("disable_display", False)

    # Generation snapshot frequency (if desired).
    intermediate_freq = config.get("intermediate_images", 0)
    capture_generations = []
    if intermediate_freq > 0:
        total_gens = config["n_generations"]
        interval = max(1, total_gens // (intermediate_freq - 1))
        capture_generations = list(range(0, total_gens, interval))
        if total_gens - 1 not in capture_generations:
            capture_generations.append(total_gens - 1)
    snapshots = []  # each element: (generation, best_individual)

    selection_func = selection_strategies[config["selection_method"]]
    crossover_func = crossover_strategies[config["crossover_method"]]
    mutation_func  = mutation_strategies[config["mutation_method"]]

    population = Population(config, w, h, global_target)
    population.evaluate(compute_fitness, target_image)
    best = population.get_best()

    n_gens = config["n_generations"]
    for gen in range(n_gens):
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
        if gen in capture_generations:
            snapshots.append((gen, best.clone()))
        if not disable_display and clear_output is not None:
            clear_output(wait=True)
            best_img = render_individual(best, w, h)
            plt.imshow(best_img)
            plt.axis("off")
            plt.title(f"Generation {gen+1}/{n_gens}, Fitness: {best.fitness:.6f}")
            plt.show()
    if n_gens - 1 not in [gen for gen, _ in snapshots]:
        snapshots.append((n_gens - 1, best.clone()))
    return snapshots