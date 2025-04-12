from PIL import Image, ImageDraw

def render_individual(individual, width, height):
    """
    Renders an individual's triangles onto a white canvas (RGBA),
    ensuring proper alpha blending even in overlapping regions.
    Returns a PIL.Image.
    """
    # Create a base image that is white and fully opaque.
    base_img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    
    # Create an overlay that will accumulate triangles.
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    
    for triangle in individual.genes:
        # Unpack gene: (x1, y1, x2, y2, x3, y3, R, G, B, A)
        x1, y1, x2, y2, x3, y3, r, g, b, a = triangle
        alpha_255 = int(a * 255)
        polygon_coords = [(x1, y1), (x2, y2), (x3, y3)]
        
        # Create a temporary image for this triangle (transparent background)
        tri_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        tri_draw = ImageDraw.Draw(tri_img, "RGBA")
        tri_draw.polygon(polygon_coords, fill=(r, g, b, alpha_255))
        
        # Alpha composite the triangle image onto the overlay.
        overlay = Image.alpha_composite(overlay, tri_img)
    
    # Finally, composite the overlay onto the white base image.
    result_img = Image.alpha_composite(base_img, overlay)
    return result_img
