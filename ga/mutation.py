# ga/mutation.py

import random

def mutate(individual, mutation_rate, image_width, image_height):
    """
    Mutación por genes: con probabilidad mutation_rate,
    se modifica alguna parte del triángulo (posiciones, color).
    """
    for i, tri in enumerate(individual.genes):
        if random.random() < mutation_rate:
            # tri es una tupla (x1,y1,x2,y2,x3,y3,R,G,B,A)
            tri_list = list(tri)
            
            # Ejemplo: mutar un vértice al azar
            idx_vertex = random.randint(0,2)  # 0->(x1,y1), 1->(x2,y2), 2->(x3,y3)
            x_index = idx_vertex*2
            y_index = idx_vertex*2 + 1

            # Pequeño desplazamiento (puede ser +/− 10 px)
            tri_list[x_index] = max(0, min(image_width-1, tri_list[x_index] + random.randint(-10, 10)))
            tri_list[y_index] = max(0, min(image_height-1, tri_list[y_index] + random.randint(-10, 10)))

            # También podríamos mutar el color
            if random.random() < 0.5:  # 50% de chance
                tri_list[6] = random.randint(0,255)  # R
                tri_list[7] = random.randint(0,255)  # G
                tri_list[8] = random.randint(0,255)  # B
                tri_list[9] = random.random()        # A en [0,1]
            
            individual.genes[i] = tuple(tri_list)
    
    # Resetear el fitness porque cambió el individuo
    individual.fitness = None