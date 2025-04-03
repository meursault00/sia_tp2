# utils/render.py
from PIL import Image, ImageDraw

def render_individual(individual, width, height):
    """
    Dibuja los triángulos del individuo en un canvas blanco (RGBA).
    Retorna un objeto PIL.Image.
    """
    # 1) Crear un canvas en blanco
    # "RGBA" => (0,0,0,0) para el fondo => hacemos fill blanco manual
    img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    
    # 2) Recorrer todos los triángulos del individuo
    for triangle in individual.genes:
        # triangle = (x1, y1, x2, y2, x3, y3, R, G, B, A)
        x1, y1, x2, y2, x3, y3, r, g, b, a = triangle
        
        # Pillow admite un fill con (R, G, B, A) si "RGBA"
        # A va de 0..255 si es un canal de 8 bits, pero aquí a ∈ [0..1].
        # => Convertir a ∈ [0..255] si se desea
        alpha_255 = int(a * 255)
        
        polygon_coords = [(x1,y1), (x2,y2), (x3,y3)]
        draw.polygon(polygon_coords, fill=(r, g, b, alpha_255))

    # 3) Devolver la imagen resultante
    return img