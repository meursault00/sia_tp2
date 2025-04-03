# utils/image_utils.py

from PIL import Image  # Ejemplo: usar Pillow

def load_image(path):
    """
    Carga la imagen con Pillow o similar y retorna un objeto con .width y .height
    """
    img = Image.open(path).convert("RGBA")
    return img