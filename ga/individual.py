import random
import numpy as np

def sample_image_color(image, x, y):
    """
    Sample the color at position (x,y) from the image.
    Returns (R,G,B,A) tuple.
    """
    if x < 0 or y < 0 or x >= image.width or y >= image.height:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.random())
    
    # Get pixel color
    pixel = image.getpixel((x, y))
    
    # Handle different image modes
    if len(pixel) == 4:  # RGBA
        R, G, B, A = pixel
        return (R, G, B, A/255.0)  # Convert alpha to [0,1] range
    elif len(pixel) == 3:  # RGB
        R, G, B = pixel
        return (R, G, B, random.uniform(0.5, 1.0))
    else:  # Grayscale or other
        return (pixel, pixel, pixel, random.uniform(0.5, 1.0))

class Individual:
    """
    Represents an individual as a list of triangles.
    """
    def __init__(self, n_triangles, image_width, image_height, target_image=None):
        self.n_triangles = n_triangles
        self.image_width = image_width
        self.image_height = image_height
        self.target_image = target_image  # Used to seed initial colors and positions
        self.genes = self._initialize_genes(image_width, image_height)
        self.fitness = None

    def _initialize_genes(self, w, h):
        genes = []
        for _ in range(self.n_triangles):
            # Image-based initialization if target_image is available
            if self.target_image is not None and random.random() < 0.8:  # 80% chance to use image data
                # Sample a position from the image (with slight bias toward edges)
                cx = random.randint(0, w - 1)
                cy = random.randint(0, h - 1)
                
                # Sample the color from this position
                R, G, B, A = sample_image_color(self.target_image, cx, cy)
                
                # Create a triangle near this sampled position
                max_offset = int(min(w, h) * 0.1)  # 10% of smaller dimension
                
                # Create triangle vertices around the sampled point
                x1 = max(0, min(w - 1, cx + random.randint(-max_offset, max_offset)))
                y1 = max(0, min(h - 1, cy + random.randint(-max_offset, max_offset)))
                x2 = max(0, min(w - 1, cx + random.randint(-max_offset, max_offset)))
                y2 = max(0, min(h - 1, cy + random.randint(-max_offset, max_offset)))
                x3 = max(0, min(w - 1, cx + random.randint(-max_offset, max_offset)))
                y3 = max(0, min(h - 1, cy + random.randint(-max_offset, max_offset)))
                
            else:
                # Fall back to original random initialization methods
                if random.random() < 0.5:
                    # Global: vertices picked uniformly across image
                    x1 = random.randint(0, w - 1)
                    y1 = random.randint(0, h - 1)
                    x2 = random.randint(0, w - 1)
                    y2 = random.randint(0, h - 1)
                    x3 = random.randint(0, w - 1)
                    y3 = random.randint(0, h - 1)
                else:
                    # Local: vertices close to a random center
                    cx = random.randint(0, w - 1)
                    cy = random.randint(0, h - 1)
                    max_offset = int(min(w, h) * 0.1)
                    x1 = max(0, min(w - 1, cx + random.randint(-max_offset, max_offset)))
                    y1 = max(0, min(h - 1, cy + random.randint(-max_offset, max_offset)))
                    x2 = max(0, min(w - 1, cx + random.randint(-max_offset, max_offset)))
                    y2 = max(0, min(h - 1, cy + random.randint(-max_offset, max_offset)))
                    x3 = max(0, min(w - 1, cx + random.randint(-max_offset, max_offset)))
                    y3 = max(0, min(h - 1, cy + random.randint(-max_offset, max_offset)))
                
                # Calculate center for color sampling
                cx_tri = int((x1 + x2 + x3) / 3)
                cy_tri = int((y1 + y2 + y3) / 3)
                
                if self.target_image is not None:
                    # Sample color from the triangle's center position
                    R, G, B, A = sample_image_color(self.target_image, cx_tri, cy_tri)
                else:
                    # Random color if no image
                    R = random.randint(0, 255)
                    G = random.randint(0, 255)
                    B = random.randint(0, 255)
                    A = random.random()
            
            triangle_tuple = (x1, y1, x2, y2, x3, y3, R, G, B, A)
            genes.append(triangle_tuple)
        return genes

    def clone(self):
        clone_ind = Individual(self.n_triangles, self.image_width, self.image_height, self.target_image)
        clone_ind.genes = [tuple(tri) for tri in self.genes]
        clone_ind.fitness = self.fitness
        return clone_ind