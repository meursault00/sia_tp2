import random
from .individual import Individual

class Population:
    def __init__(self, config, image_width, image_height):
        """
        Creates an initial population based on the provided configuration.
        Here, config["n_triangles"] is expected to be the number of triangles per patch.
        """
        self.individuals = []
        for _ in range(config["population_size"]):
            ind = Individual(config["n_triangles"], image_width, image_height)
            self.individuals.append(ind)

    def evaluate(self, fitness_func, target_image):
        for ind in self.individuals:
            ind.fitness = fitness_func(ind, target_image)
    
    def get_best(self):
        return max(self.individuals, key=lambda ind: ind.fitness)