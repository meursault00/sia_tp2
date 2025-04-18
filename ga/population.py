import random
from .individual import Individual

class Population:
    def __init__(self, config, image_width, image_height, global_target=None):
        """
        Create an initial population.
        The parameter global_target (the full image) is used for color sampling.
        Image sampling can be enabled/disabled via config.
        """
        self.individuals = []
        
        # Get image sampling configuration with defaults
        use_image_sampling = config.get("use_image_sampling", True)
        sampling_rate = config.get("sampling_rate", 0.8)
        
        for _ in range(config["population_size"]):
            ind = Individual(
                config["n_triangles"], 
                image_width, 
                image_height, 
                global_target,
                use_image_sampling,
                sampling_rate
            )
            self.individuals.append(ind)

    def evaluate(self, fitness_func, target_image):
        for ind in self.individuals:
            ind.fitness = fitness_func(ind, target_image)
    
    def get_best(self):
        return max(self.individuals, key=lambda ind: ind.fitness)