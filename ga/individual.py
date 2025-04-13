import random
from PIL import Image

class Individual:
    """
    Represents an individual as a list of triangles.
    Colors are initialized by sampling from the full (global) target image.
    """
    def __init__(self, n_triangles, image_width, image_height, global_target=None):
        self.n_triangles = n_triangles
        self.global_target = global_target  # Full target image for color sampling.
        self.genes = self._initialize_genes(image_width, image_height, global_target)
        self.fitness = None

    def _initialize_genes(self, w, h, global_target):
        genes = []
        for _ in range(self.n_triangles):
            # Generate random triangle vertices in the local coordinate system.
            x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
            x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
            x3, y3 = random.randint(0, w-1), random.randint(0, h-1)
            
            # For color sampling, use the global target if provided.
            if global_target:
                # Get the full image dimensions.
                global_w, global_h = global_target.width, global_target.height
                sample_x = random.randint(0, global_w - 1)
                sample_y = random.randint(0, global_h - 1)
                pixel = global_target.getpixel((sample_x, sample_y))
                R, G, B = pixel[:3]
            else:
                R, G, B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            
            A = random.random()  # Alpha in [0,1]
            triangle_tuple = (x1, y1, x2, y2, x3, y3, R, G, B, A)
            genes.append(triangle_tuple)
        return genes

    def clone(self):
        clone_ind = Individual(self.n_triangles, 1, 1, self.global_target)
        clone_ind.genes = [tuple(tri) for tri in self.genes]
        clone_ind.fitness = self.fitness
        return clone_ind