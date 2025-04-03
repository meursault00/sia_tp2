# ga/individual.py

import random

class Individual:
    """
    Representa un individuo como un listado de triángulos.
    """
    def __init__(self, n_triangles, image_width, image_height):
        self.n_triangles = n_triangles
        self.genes = self._initialize_genes(image_width, image_height)
        self.fitness = None

    def _initialize_genes(self, w, h):
        genes = []
        for _ in range(self.n_triangles):
            x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
            x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
            x3, y3 = random.randint(0, w-1), random.randint(0, h-1)
            R, G, B = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            A = random.random()  # alpha en [0,1]
            triangle_tuple = (x1, y1, x2, y2, x3, y3, R, G, B, A)
            genes.append(triangle_tuple)
        return genes

    def clone(self):
        clone_ind = Individual(self.n_triangles, 1, 1)  # w,h no importan aquí
        clone_ind.genes = [tuple(tri) for tri in self.genes]
        clone_ind.fitness = self.fitness
        return clone_ind