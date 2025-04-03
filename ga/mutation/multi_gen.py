import random

def multi_gen_mutation(individual, mutation_rate, image_width, image_height):
    """
    Example: each gene has its own chance to mutate independently,
    possibly bigger changes, etc.
    """
    for i, tri in enumerate(individual.genes):
        tri_list = list(tri)
        # Para cada componente en tri_list, chance de mutar
        for comp_idx in range(len(tri_list)):
            if random.random() < mutation_rate:
                if comp_idx < 6:
                    # Son coordenadas
                    if comp_idx % 2 == 0:
                        # x
                        tri_list[comp_idx] = random.randint(0, image_width - 1)
                    else:
                        # y
                        tri_list[comp_idx] = random.randint(0, image_height - 1)
                else:
                    # color
                    if comp_idx < 9:
                        # R, G, B
                        tri_list[comp_idx] = random.randint(0, 255)
                    else:
                        # A
                        tri_list[comp_idx] = random.random()

        individual.genes[i] = tuple(tri_list)

    individual.fitness = None