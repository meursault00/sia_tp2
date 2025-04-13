import random
from PIL import Image

class Individual:
    """
    Represents an individual as a list of triangles, with colors initialized from the input image.
    """
    def __init__(self, n_triangles, image_width, image_height, target_image=None):
        self.n_triangles = n_triangles
        self.genes = self._initialize_genes(image_width, image_height, target_image)
        self.fitness = None

    def _initialize_genes(self, w, h, target_image):
        genes = []
        for _ in range(self.n_triangles):
            # Random triangle vertices
            x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
            x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
            x3, y3 = random.randint(0, w-1), random.randint(0, h-1)
            
            # Sample color from target image if provided, else random
            if target_image:
                # Sample at a random vertex (e.g., (x1, y1))
                sample_x, sample_y = x1, y1
                # Ensure coordinates are within bounds
                sample_x = min(max(0, sample_x), w-1)
                sample_y = min(max(0, sample_y), h-1)
                # Get RGB from image (target_image is PIL Image)
                pixel = target_image.getpixel((sample_x, sample_y))
                # Handle RGB or RGBA images
                R, G, B = pixel[:3]
            else:
                R, G, B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            
            A = random.random()  # Alpha in [0, 1]
            triangle_tuple = (x1, y1, x2, y2, x3, y3, R, G, B, A)
            genes.append(triangle_tuple)
        return genes

    def clone(self):
        clone_ind = Individual(self.n_triangles, 1, 1)  # w, h don’t matter here
        clone_ind.genes = [tuple(tri) for tri in self.genes]
        clone_ind.fitness = self.fitness
        return clone_ind