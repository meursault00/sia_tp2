import random

def basic_mutation(individual, mutation_rate, image_width, image_height):
    """
    Simple mutation: with probability mutation_rate,
    pick one triangle param and randomize it a bit.
    """
    for i, tri in enumerate(individual.genes):
        if random.random() < mutation_rate:
            tri_list = list(tri)
            # mutar un vértice al azar
            idx_vertex = random.randint(0, 2)  # 0->(x1,y1), 1->(x2,y2), 2->(x3,y3)
            x_index = idx_vertex*2
            y_index = idx_vertex*2 + 1

            # Pequeño desplazamiento (± 10 px)
            tri_list[x_index] = max(0, min(image_width - 1, tri_list[x_index] + random.randint(-10, 10)))
            tri_list[y_index] = max(0, min(image_height - 1, tri_list[y_index] + random.randint(-10, 10)))

            # O mutar color
            if random.random() < 0.5:
                tri_list[6] = random.randint(0, 255)  # R
                tri_list[7] = random.randint(0, 255)  # G
                tri_list[8] = random.randint(0, 255)  # B
                tri_list[9] = random.random()         # A in [0,1]

            individual.genes[i] = tuple(tri_list)

    individual.fitness = None