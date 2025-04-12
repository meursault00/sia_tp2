import sys
import os
import json
import math
from PIL import Image
from utils.image_utils import load_image
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_patch_ga(patch_box, nominal_rect, local_config, patch):
    """
    Runs GA on one patch.
    Sets a flag to disable interactive display.
    Returns (patch_box, nominal_rect, best_individual).
    """
    local_config["disable_display"] = True  # disable display for parallel workers
    from ga.algorithm import run_ga  # import locally for pickling compatibility
    best_individual = run_ga(local_config, patch)
    return patch_box, nominal_rect, best_individual

def create_blend_mask(patch_w, patch_h, left_margin, right_margin, top_margin, bottom_margin):
    """
    Creates a blending weight mask for a patch of dimensions (patch_w, patch_h).
    In the horizontal dimension, weights rise linearly from 0 at the left edge to 1 at left_margin,
    remain 1 in the central region, and then drop linearly to 0 at the right edge.
    Similarly for vertical weights.
    The final mask is the outer product of the x and y 1D masks.
    """
    # Create 1D weight for x direction.
    weights_x = np.ones(patch_w, dtype=np.float32)
    for i in range(patch_w):
        if i < left_margin:
            weights_x[i] = i / left_margin
        elif i >= patch_w - right_margin:
            weights_x[i] = (patch_w - i - 1) / right_margin
    weights_x = np.clip(weights_x, 0, 1)
    
    # Similarly for y direction.
    weights_y = np.ones(patch_h, dtype=np.float32)
    for j in range(patch_h):
        if j < top_margin:
            weights_y[j] = j / top_margin
        elif j >= patch_h - bottom_margin:
            weights_y[j] = (patch_h - j - 1) / bottom_margin
    weights_y = np.clip(weights_y, 0, 1)
    
    # Outer product to get full 2D mask.
    mask = np.outer(weights_y, weights_x)
    # Expand dims to be compatible with RGBA channels.
    mask = np.repeat(mask[:, :, np.newaxis], 4, axis=2)
    return mask

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <config_path.json> <image_path>")
        sys.exit(1)

    # Load global configuration and full target image.
    config_path = sys.argv[1]
    image_path = sys.argv[2]
    with open(config_path, "r") as f:
        global_config = json.load(f)
    target_image = load_image(image_path)
    full_w, full_h = target_image.width, target_image.height

    # Calculate grid dimensions from a global triangle budget.
    n_triangles_global = global_config.get("n_triangles_global", 100)
    tri_per_patch_desired = global_config.get("tri_per_patch_desired", 10)
    total_patches = max(1, round(n_triangles_global / tri_per_patch_desired))
    grid_size = max(1, round(math.sqrt(total_patches)))
    grid_rows = grid_cols = grid_size
    n_patches = grid_rows * grid_cols
    triangles_per_patch = max(1, n_triangles_global // n_patches)

    print(f"Dividing image into a {grid_rows}x{grid_cols} grid (total patches: {n_patches}).")
    print(f"Global triangles: {n_triangles_global} → {triangles_per_patch} triangles per patch.")

    # Nominal (non-overlapping) patch dimensions.
    nom_patch_w = full_w // grid_cols
    nom_patch_h = full_h // grid_rows

    # Overlap fraction (e.g., 10%).
    overlap_frac = global_config.get("overlap_frac", 0.4)

    # Prepare patch tasks.
    tasks = []
    # Also record nominal rectangle for each patch.
    for row in range(grid_rows):
        for col in range(grid_cols):
            left_nom = col * nom_patch_w
            upper_nom = row * nom_patch_h
            right_nom = left_nom + nom_patch_w if col < grid_cols - 1 else full_w
            lower_nom = upper_nom + nom_patch_h if row < grid_rows - 1 else full_h
            nominal_rect = (left_nom, upper_nom, right_nom, lower_nom)
            
            # Compute overlap margins.
            x_margin = int(nom_patch_w * overlap_frac)
            y_margin = int(nom_patch_h * overlap_frac)
            
            left = max(0, left_nom - x_margin)
            upper = max(0, upper_nom - y_margin)
            right = min(full_w, right_nom + x_margin)
            lower = min(full_h, lower_nom + y_margin)
            patch_box = (left, upper, right, lower)
            
            patch = target_image.crop(patch_box)
            local_config = global_config.copy()
            local_config["n_triangles"] = triangles_per_patch
            tasks.append((patch_box, nominal_rect, local_config, patch))
    
    # Use parallel processing for all patches.
    num_workers = min(len(tasks), os.cpu_count() or 1)
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_patch_ga, tb, nr, conf, p) for tb, nr, conf, p in tasks]
        for fut in as_completed(futures):
            try:
                result = fut.result()  # result = (patch_box, nominal_rect, best_individual)
                results.append(result)
            except Exception as e:
                print("Error processing a patch:", e)
    
    # Composite the final image using weighted blending.
    composite_array = np.zeros((full_h, full_w, 4), dtype=np.float32)
    weight_array = np.zeros((full_h, full_w), dtype=np.float32)
    
    # For each patch result:
    from utils.render import render_individual
    for patch_box, nominal_rect, best_individual in results:
        # Render the best individual's patch over the entire overlapping region.
        left, upper, right, lower = patch_box
        patch_w_run = right - left
        patch_h_run = lower - upper
        patch_result_img = render_individual(best_individual, patch_w_run, patch_h_run)
        patch_result_arr = np.array(patch_result_img, dtype=np.float32) / 255.0

        # Create a blending weight mask for this patch.
        # Compute margins for the patch relative to its nominal region.
        nom_left, nom_upper, nom_right, nom_lower = nominal_rect
        left_margin = nom_left - left
        right_margin = right - nom_right
        top_margin = nom_upper - upper
        bottom_margin = lower - nom_lower
        # Ensure margins are at least 1 to avoid division by zero.
        left_margin = max(1, left_margin)
        right_margin = max(1, right_margin)
        top_margin = max(1, top_margin)
        bottom_margin = max(1, bottom_margin)
        
        mask = create_blend_mask(patch_w_run, patch_h_run, left_margin, right_margin, top_margin, bottom_margin)
        
        # Determine where in the composite image this patch goes.
        # We use the full patch_box region.
        comp_region = (left, upper, right, lower)
        comp_arr = np.array(composite_array, copy=False)
        weight_region = weight_array  # We'll update weight_array accordingly.
        
        # Add weighted contribution.
        # Determine composite array slice:
        comp_slice = (slice(upper, lower), slice(left, right), slice(None))
        composite_array[comp_slice] += patch_result_arr * mask
        # Also add mask weights to weight array (only one channel needed).
        weight_array[upper:lower, left:right] += mask[:, :, 0]
    
    # Normalize composite image by total weight, taking care of zeros.
    weight_array[weight_array == 0] = 1.0
    normalized = composite_array / weight_array[:, :, np.newaxis]
    normalized = (np.clip(normalized, 0, 1) * 255).astype(np.uint8)
    final_composite = Image.fromarray(normalized, mode="RGBA")

    # Save final composite image.
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    output_image_name = global_config.get("output_image_name", "composite_result.png")
    output_file_path = os.path.join(results_folder, output_image_name)
    final_composite.save(output_file_path)
    print(f"Final composite image saved as {output_file_path}")

def create_blend_mask(patch_w, patch_h, left_margin, right_margin, top_margin, bottom_margin):
    """
    Create a blending mask with dimensions (patch_w, patch_h) such that:
      - The weight is 1 over the nominal (central) region,
      - And linearly tapers to 0 over the left, right, top, and bottom margins.
    """
    # For x coordinates.
    weights_x = np.ones(patch_w, dtype=np.float32)
    for i in range(patch_w):
        if i < left_margin:
            weights_x[i] = i / left_margin
        elif i >= patch_w - right_margin:
            weights_x[i] = (patch_w - i - 1) / right_margin
    weights_x = np.clip(weights_x, 0, 1)
    
    # For y coordinates.
    weights_y = np.ones(patch_h, dtype=np.float32)
    for j in range(patch_h):
        if j < top_margin:
            weights_y[j] = j / top_margin
        elif j >= patch_h - bottom_margin:
            weights_y[j] = (patch_h - j - 1) / bottom_margin
    weights_y = np.clip(weights_y, 0, 1)
    
    mask = np.outer(weights_y, weights_x)
    mask = np.repeat(mask[:, :, np.newaxis], 4, axis=2)
    return mask

if __name__ == "__main__":
    main()
