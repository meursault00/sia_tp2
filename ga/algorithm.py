# ga/algorithm.py

import random
from .population import Population
from .fitness import compute_fitness
# Importamos los "diccionarios" de estrategias
from .selection import selection_strategies
from .crossover import crossover_strategies
from .mutation import mutation_strategies

def run_ga(config, target_image):
    """
    config: dict con las claves:
      population_size, n_generations, mutation_rate, crossover_rate, n_triangles,
      selection_method, crossover_method, mutation_method, ...
    """
    w, h = target_image.width, target_image.height

    # Tomamos las funciones según lo que dice el JSON
    selection_func = selection_strategies[config["selection_method"]]
    crossover_func = crossover_strategies[config["crossover_method"]]
    mutation_func  = mutation_strategies[config["mutation_method"]]

    # 1. Crear poblacion
    population = Population(config, w, h)
    population.evaluate(compute_fitness, target_image)

    best = population.get_best()

    for gen in range(config["n_generations"]):
        # 2. Selección
        parents = selection_func(population.individuals)  
        # O si tu selección necesita parámetros, e.g. tournament_func(pop, k=3)

        # 3. Cruza
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

        # 4. Mutación
        for ind in new_generation:
            mutation_func(ind, config["mutation_rate"], w, h)

        # 5. Reemplazo
        population.individuals = new_generation
        population.evaluate(compute_fitness, target_image)
        
        current_best = population.get_best()
        if current_best.fitness > best.fitness:
            best = current_best.clone()

        print(f"Generación {gen+1}, Mejor Fitness: {best.fitness:.6f}")

    return best