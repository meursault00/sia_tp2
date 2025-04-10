import random

def preset_color(cx, w):
    """
    Define a preset color based on the horizontal coordinate.
    For example, for a simple Canadian flag:
      - Red for the left and right sides.
      - White for the center.
    Adjust the thresholds as needed.
    """
    # Example thresholds: left 30% and right 30% are red; center is white.
    if cx < 0.3 * w or cx > 0.7 * w:
        return (255, 0, 0, 1.0)  # Red, fully opaque
    else:
        return (255, 255, 255, 1.0)  # White, fully opaque

class Individual:
    """
    Represents an individual as a list of triangles.
    """
    def __init__(self, n_triangles, image_width, image_height, target_image=None):
        self.n_triangles = n_triangles
        self.image_width = image_width
        self.image_height = image_height
        self.target_image = target_image  # Optional: used to seed initial colors.
        self.genes = self._initialize_genes(image_width, image_height)
        self.fitness = None

    def _initialize_genes(self, w, h):
        genes = []
        for _ in range(self.n_triangles):
            # Decide randomly which initialization strategy to use.
            if random.random() < 0.5:
                # Global: each vertex is picked uniformly across the whole image.
                x1 = random.randint(0, w - 1)
                y1 = random.randint(0, h - 1)
                x2 = random.randint(0, w - 1)
                y2 = random.randint(0, h - 1)
                x3 = random.randint(0, w - 1)
                y3 = random.randint(0, h - 1)
            else:
                # Local: choose a random center and small offsets.
                cx = random.randint(0, w - 1)
                cy = random.randint(0, h - 1)
                # A relatively small offset (e.g. 10% of the smaller dimension)
                max_offset = int(min(w, h) * 0.1)
                x1 = max(0, min(w - 1, cx + random.randint(-max_offset, max_offset)))
                y1 = max(0, min(h - 1, cy + random.randint(-max_offset, max_offset)))
                x2 = max(0, min(w - 1, cx + random.randint(-max_offset, max_offset)))
                y2 = max(0, min(h - 1, cy + random.randint(-max_offset, max_offset)))
                x3 = max(0, min(w - 1, cx + random.randint(-max_offset, max_offset)))
                y3 = max(0, min(h - 1, cy + random.randint(-max_offset, max_offset)))
            
            # Compute the center (average) of the triangle's vertices.
            cx_tri = int((x1 + x2 + x3) / 3)
            
            # If a target image is provided, use the preset color function.
            if self.target_image is not None:
                # Use the x-coordinate of the triangle center (you can also incorporate y if needed)
                R, G, B, A = preset_color(cx_tri, w)
            else:
                # Otherwise use random color.
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
