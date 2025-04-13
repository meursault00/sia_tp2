from PIL import Image

def load_image(path, max_size=(256, 256)):
    img = Image.open(path)
    img = img.convert("RGBA")
    # Use LANCZOS for high-quality downsampling.
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img