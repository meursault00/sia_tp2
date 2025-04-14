import sys
import os
import json
import math
from PIL import Image
from utils.image_utils import load_image
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

def run_patch_ga(patch_box, nominal_rect, local_config, patch, global_target):
    """
    Runs the GA on one patch.
    Disables interactive display in parallel workers.
    Returns a tuple (patch_box, nominal_rect, best_individual).
    """
    local_config["disable_display"] = True  # disable display for parallel workers
    from ga.algorithm import run_ga  # import locally for pickling compatibility
    snapshots = run_ga(local_config, patch, global_target)
    print(f"[Worker] Finished GA for patch with overlapping bounds: {patch_box}", flush=True)
    return patch_box, nominal_rect, snapshots

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <config_path.json> <image_path>")
        sys.exit(1)

    # Load global configuration and full target image.
    config_path = sys.argv[1]
    image_path = sys.argv[2]
    with open(config_path, "r") as f:
        global_config = json.load(f)
    max_dim = global_config.get("max_dim", 256)
    full_target = load_image(image_path, max_size=(max_dim, max_dim))
    full_w, full_h = full_target.width, full_target.height
    print(f"[Main] Loaded full target image: {full_w}x{full_h}")

    # Calculate grid dimensions automatically from the global triangle budget.
    n_triangles_global = global_config.get("n_triangles_global", 100)
    tri_per_patch_desired = global_config.get("tri_per_patch_desired", 10)
    total_patches = max(1, round(n_triangles_global / tri_per_patch_desired))
    grid_size = max(1, round(math.sqrt(total_patches)))
    grid_rows = grid_cols = grid_size
    n_patches = grid_rows * grid_cols
    triangles_per_patch = max(1, n_triangles_global // n_patches)
    print(f"[Main] Dividing image into a {grid_rows}x{grid_cols} grid (total patches: {n_patches}).")
    print(f"[Main] Global triangles: {n_triangles_global} → {triangles_per_patch} triangles per patch.")

    # Nominal (non-overlapping) patch dimensions.
    nom_patch_w = full_w // grid_cols
    nom_patch_h = full_h // grid_rows

    # Overlap fraction (set to desired value; here increased to 0.3).
    overlap_frac = global_config.get("overlap_frac", 0.3)
    print(f"[Main] Using overlap fraction: {overlap_frac}")

    # Build tasks – for each grid cell, calculate nominal and overlapping boundaries.
    tasks = []
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
            
            print(f"[Main] Queuing patch: Nominal {nominal_rect}, Overlap {patch_box}")
            patch = full_target.crop(patch_box)
            local_config = global_config.copy()
            local_config["n_triangles"] = triangles_per_patch
            tasks.append((patch_box, nominal_rect, local_config, patch, full_target))

    num_tasks = len(tasks)
    print(f"[Main] Submitting {num_tasks} patch GA tasks in parallel...")

    # Run patch GA tasks in parallel.
    num_workers = min(len(tasks), os.cpu_count() or 1)
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_patch_ga, tb, nr, conf, p, full_target)
                   for tb, nr, conf, p, _ in tasks]
        future_counter = 0
        for future in as_completed(futures):
            try:
                result = future.result()  # result = (patch_box, nominal_rect, snapshots)
                results.append(result)
                print(f"[Main] Finished patch [{future_counter}] of [{num_tasks}] with overlapping bounds: {result[0]}", flush=True)
                future_counter += 1
            except Exception as e:
                print("Error processing a patch:", e)
    
    # Composite the snapshots into full images using weighted blending (make sure it's at least 1)
    num_snapshots = global_config.get("intermediate_images", 1)
    num_snapshots = 1 if num_snapshots <= 0 else num_snapshots

    snapshot_composites = [np.zeros((full_h, full_w, 4), dtype=np.float32) for _ in range(num_snapshots)]
    snapshot_weights   = [np.zeros((full_h, full_w), dtype=np.float32) for _ in range(num_snapshots)]

    from utils.blend import create_blend_mask  # assume this function is in utils/blend.py
    from utils.render import render_individual
    # For each patch result:
    for patch_box, nominal_rect, snapshots in results:
        left, upper, right, lower = patch_box
        patch_w_run = right - left
        patch_h_run = lower - upper

        # Compute blending mask once per patch
        nom_left, nom_upper, nom_right, nom_lower = nominal_rect
        left_margin   = max(1, nom_left - left)
        right_margin  = max(1, right - nom_right)
        top_margin    = max(1, nom_upper - upper)
        bottom_margin = max(1, lower - nom_lower)
        mask = create_blend_mask(patch_w_run, patch_h_run, left_margin, right_margin, top_margin, bottom_margin)

        # Add this patch's contribution to each snapshot
        for i, (_, individual) in enumerate(snapshots):
            patch_img = render_individual(individual, patch_w_run, patch_h_run)
            patch_arr = np.array(patch_img, dtype=np.float32) / 255.0

            snapshot_composites[i][upper:lower, left:right] += patch_arr * mask
            snapshot_weights[i][upper:lower, left:right] += mask[:, :, 0]

    # Normalize and save each snapshot

    # With:
    results_folder = global_config.get("results_folder", "results")
    os.makedirs(results_folder, exist_ok=True)

    output_name = global_config.get("output_image_name", "composite_result.png")
    base, ext = os.path.splitext(output_name)

    for i in range(num_snapshots):
        # Use generation number from the first patch (they all match across patches)
        gen_number, _ = results[0][2][i]

        # Aggregate fitness from all patches at this generation
        fitnesses = [snapshots[i][1].fitness for _, _, snapshots in results]
        avg_fitness = sum(fitnesses) / len(fitnesses)

        weight = snapshot_weights[i]
        weight[weight == 0] = 1.0
        normalized = snapshot_composites[i] / weight[:, :, np.newaxis]
        normalized = (np.clip(normalized, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(normalized, mode="RGBA")

        output_path = os.path.join(results_folder, f"{base}_gen{gen_number}{ext}")
        img.save(output_path)
        print(f"[Main] Saved generation {gen_number} to {output_path}, avg fitness = {avg_fitness:.6f}")
        fitness_var = np.var(fitnesses)
        print(f"[Main] Fitness variance across patches for generation {gen_number}: {fitness_var:.6f}")


if __name__ == "__main__":
    main()