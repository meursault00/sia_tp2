# ga/algorithm.py

import random
from .population import Population
from .selection import tournament_selection
from .crossover import one_point_crossover
from .mutation import mutate
from .fitness import compute_fitness

def run_ga(config, target_image):
    """
    Bucle principal del AG.
    config es un dict con claves:
      population_size, n_generations, mutation_rate, crossover_rate, n_triangles...
    """
    w, h = target_image.width, target_image.height
    
    population = Population(config, w, h)  # Pasamos config al constructor
    population.evaluate(compute_fitness, target_image)

    best = population.get_best()

    for gen in range(config["n_generations"]):
        # Selección
        parents = tournament_selection(population.individuals, k=3)

        # Cruza
        new_generation = []
        for i in range(0, len(parents), 2):
            if i+1 < len(parents):
                p1, p2 = parents[i], parents[i+1]
                if random.random() < config["crossover_rate"]:
                    c1, c2 = one_point_crossover(p1, p2)
                else:
                    c1, c2 = p1.clone(), p2.clone()
                new_generation.append(c1)
                new_generation.append(c2)
            else:
                new_generation.append(parents[i].clone())

        # Mutación
        for ind in new_generation:
            mutate(
                ind, 
                config["mutation_rate"],
                w,
                h
            )

        # Reemplazo
        population.individuals = new_generation
        population.evaluate(compute_fitness, target_image)
        
        current_best = population.get_best()
        if current_best.fitness > best.fitness:
            best = current_best.clone()

        print(f"Generación {gen+1}, Mejor Fitness: {best.fitness:.6f}")

    return best