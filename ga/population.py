import random
from .individual import Individual

class Population:
    def __init__(self, config, image_width, image_height, target_image=None):
        """
        Creates an initial population based on the provided configuration.
        Passes target_image to Individual for color initialization.
        """
        self.individuals = []
        for _ in range(config["population_size"]):
            ind = Individual(config["n_triangles"], image_width, image_height, target_image)
            self.individuals.append(ind)

    def evaluate(self, fitness_func, target_image):
        for ind in self.individuals:
            ind.fitness = fitness_func(ind, target_image)
    
    def get_best(self):
        return max(self.individuals, key=lambda ind: ind.fitness)