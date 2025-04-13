import random
from PIL import Image

class Individual:
    """
    Represents an individual as a list of triangles.
    Colors can be initialized randomly or by sampling from the target image.
    """
    def __init__(self, n_triangles, image_width, image_height, global_target=None, 
                use_image_sampling=True, sampling_rate=0.8):
        self.n_triangles = n_triangles
        self.global_target = global_target  # Full target image for color sampling.
        self.use_image_sampling = use_image_sampling  # Toggle for image-based sampling
        self.sampling_rate = sampling_rate  # Percentage of triangles to sample from image
        self.genes = self._initialize_genes(image_width, image_height, global_target)
        self.fitness = None

    def _initialize_genes(self, w, h, global_target):
        genes = []
        for _ in range(self.n_triangles):
            # Generate random triangle vertices in the local coordinate system.
            x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
            x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
            x3, y3 = random.randint(0, w-1), random.randint(0, h-1)
            
            # Determine color initialization strategy
            if global_target and self.use_image_sampling and random.random() < self.sampling_rate:
                # Sample color from the target image
                global_w, global_h = global_target.width, global_target.height
                
                # Calculate center of the triangle for better sampling
                cx = int((x1 + x2 + x3) / 3)
                cy = int((y1 + y2 + y3) / 3)
                
                # Keep coordinates within image bounds
                sample_x = min(max(0, cx), global_w - 1)
                sample_y = min(max(0, cy), global_h - 1)
                
                # Get pixel color
                pixel = global_target.getpixel((sample_x, sample_y))
                R, G, B = pixel[:3]
                
                # Use higher alpha for sampled colors
                A = random.uniform(0.7, 1.0)
            else:
                # Completely random color
                R, G, B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                A = random.uniform(0.3, 0.9)  # Random alpha
            
            triangle_tuple = (x1, y1, x2, y2, x3, y3, R, G, B, A)
            genes.append(triangle_tuple)
        return genes

    def clone(self):
        clone_ind = Individual(
            self.n_triangles, 1, 1, self.global_target, 
            self.use_image_sampling, self.sampling_rate
        )
        clone_ind.genes = [tuple(tri) for tri in self.genes]
        clone_ind.fitness = self.fitness
        return clone_ind