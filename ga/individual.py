# ga/individual.py

import random
import numpy as np
from PIL import Image

class Individual:
    """
    Representa un individuo como un listado de triángulos.
    """
    def __init__(self, n_triangles, image_width, image_height, target_image=None):
        self.n_triangles = n_triangles
        self.genes = self._initialize_genes(image_width, image_height, target_image)
        self.fitness = None

    def _initialize_genes(self, w, h, target_image=None):
        genes = []
        # Convert target image to numpy array if provided
        if target_image is not None:
            img_array = np.array(target_image)
            
        for _ in range(self.n_triangles):
            # Choose a random position for the triangle
            if target_image is not None and random.random() < 0.7:  # 70% chance of image-guided initialization
                # Sample a region of the image for placing a triangle
                region_x = random.randint(0, w-1)
                region_y = random.randint(0, h-1)
                
                # Sample color from that position
                x_sample = min(region_x, img_array.shape[1]-1)
                y_sample = min(region_y, img_array.shape[0]-1)
                
                # Get color from sampled position
                if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                    R, G, B = img_array[y_sample, x_sample, 0:3]
                    # Create points around the sampled position with some randomness
                    variation = int(min(w, h) * 0.1)  # 10% of image size variation
                    x1 = max(0, min(w-1, region_x + random.randint(-variation, variation)))
                    y1 = max(0, min(h-1, region_y + random.randint(-variation, variation)))
                    x2 = max(0, min(w-1, region_x + random.randint(-variation, variation)))
                    y2 = max(0, min(h-1, region_y + random.randint(-variation, variation)))
                    x3 = max(0, min(w-1, region_x + random.randint(-variation, variation)))
                    y3 = max(0, min(h-1, region_y + random.randint(-variation, variation)))
                    
                    # Use alpha from the original image if available, otherwise random
                    A = img_array[y_sample, x_sample, 3]/255.0 if img_array.shape[2] > 3 else random.uniform(0.2, 0.8)
                else:
                    # Fallback to random
                    x1, y1, x2, y2, x3, y3 = self._generate_random_triangle_points(w, h)
                    R, G, B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    A = random.random()
            else:
                # Traditional random initialization
                x1, y1, x2, y2, x3, y3 = self._generate_random_triangle_points(w, h)
                R, G, B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                A = random.random()

            triangle_tuple = (x1, y1, x2, y2, x3, y3, R, G, B, A)
            genes.append(triangle_tuple)
        return genes

    def _generate_random_triangle_points(self, w, h):
        """Helper method to generate random triangle points"""
        x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
        x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
        x3, y3 = random.randint(0, w-1), random.randint(0, h-1)
        return x1, y1, x2, y2, x3, y3

    def clone(self):
        clone_ind = Individual(self.n_triangles, 1, 1)  # w,h no importan aquí
        clone_ind.genes = [tuple(tri) for tri in self.genes]
        clone_ind.fitness = self.fitness
        return clone_ind