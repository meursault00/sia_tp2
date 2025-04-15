import re
from PIL import Image, ImageDraw, ImageFont

def load_image(path, max_size=(256, 256)):
    img = Image.open(path)
    img = img.convert("RGBA")
    # Use LANCZOS for high-quality downsampling.
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img

def create_montage(run_label, image_paths, output_path, 
                   big_title_height=50, 
                   tile_label_height=30,
                   tile_gap=10):
    """
    Creates a horizontal montage with:
      - A top title bar (big_title_height) for the run_label.
      - For each image (original + snapshots), a small label bar (tile_label_height) on top,
        the image itself below, and no explicit edges/bounding rectangle.
      - Each tile is placed side by side with tile_gap pixels of spacing.

    Parameters:
      run_label           (str): Overall run name, e.g. "guyana_default"
      image_paths   (list[str]): List of images [original, gen0, gen25, ...]
      output_path         (str): Where to save the final montage
      big_title_height    (int): Height in px for the top bar containing run_label
      tile_label_height   (int): Height in px for the label above each sub‐image
      tile_gap            (int): Horizontal gap (in px) between consecutive tiles
    """

    if not image_paths:
        print("[create_montage] No image paths provided; skipping montage.")
        return

    # Pick a reference image to define the width/height for every tile
    # Here we assume the second entry is "Gen 0" and has the typical size
    if len(image_paths) >= 2:
        ref_img = Image.open(image_paths[1]).convert("RGBA")
    else:
        ref_img = Image.open(image_paths[0]).convert("RGBA")

    ref_w, ref_h = ref_img.size

    # Build individual "tiles" for each image
    tile_images = []
    for i, path in enumerate(image_paths):
        pil_img = Image.open(path).convert("RGBA")

        # Identify label: "Original" for the first, else parse generation
        if i == 0:
            tile_label_text = "Original"
        else:
            match = re.search(r"_gen(\d+)", path)
            if match:
                tile_label_text = f"Gen {match.group(1)}"
            else:
                tile_label_text = f"Snapshot {i}"

        # Resize to match reference
        pil_img = pil_img.resize((ref_w, ref_h), Image.Resampling.LANCZOS)

        # Create a tile with a fully or partially transparent background
        # This means no explicit rectangle edges.
        tile_bg = (255, 255, 255, 0)  # RGBA white with 0 alpha
        tile = Image.new("RGBA", (ref_w, tile_label_height + ref_h), tile_bg)

        draw_tile = ImageDraw.Draw(tile)
        # Label text
        font = None
        text_color = (0, 0, 0, 255)
        bbox = draw_tile.textbbox((0, 0), tile_label_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (ref_w - text_w) // 2
        text_y = (tile_label_height - text_h) // 2
        # Draw label in black
        draw_tile.text((text_x, text_y), tile_label_text, fill=text_color, font=font)

        # Paste the main image below the label area
        tile.paste(pil_img, (0, tile_label_height), pil_img)  # use pil_img as mask

        tile_images.append(tile)

    # Now we place these tiles side by side with tile_gap pixels in between
    num_tiles = len(tile_images)
    if num_tiles == 0:
        print("[create_montage] No tiles created; skipping montage.")
        return

    tile_width = tile_images[0].width
    tile_height = tile_images[0].height
    # total montage width: sum of tile widths + spacing
    collage_width = (tile_width * num_tiles) + (tile_gap * (num_tiles - 1))
    collage_height = big_title_height + tile_height

    # Create the final montage background (white)
    montage_bg = (255, 255, 255, 255)  # opaque white
    montage = Image.new("RGBA", (collage_width, collage_height), montage_bg)
    draw_montage = ImageDraw.Draw(montage)

    # Draw the big run_label in the top bar
    font = None
    text_color = (0, 0, 0, 255)
    bbox = draw_montage.textbbox((0, 0), run_label, font=font)
    big_text_w = bbox[2] - bbox[0]
    big_text_h = bbox[3] - bbox[1]
    big_text_x = (collage_width - big_text_w) // 2
    big_text_y = (big_title_height - big_text_h) // 2
    draw_montage.text((big_text_x, big_text_y), run_label, fill=text_color, font=font)

    # Paste each tile below the top bar with tile_gap between them
    x_offset = 0
    for tile in tile_images:
        montage.paste(tile, (x_offset, big_title_height), tile)
        x_offset += tile_width + tile_gap

    montage.save(output_path)
    print(f"[create_montage] Montage saved to {output_path}")