import sys
import os
import json
import math
import time
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.image_utils import load_image
from utils.render import render_individual
from utils.blend import create_blend_mask
from ga.algorithm import run_ga

def run_patch_ga(patch_box, nominal_rect, local_config, patch, global_target):
    """
    Runs the GA on one patch in a process.
    Returns (patch_box, nominal_rect, best_individual).
    """
    local_config["disable_display"] = True
    try:
        snapshots = run_ga(local_config, patch, global_target)
        if not snapshots:
            raise ValueError("No snapshots returned from run_ga")
        best_individual = snapshots[-1][1]  # Last tuple: (gen, Individual)
        return patch_box, nominal_rect, best_individual
    except Exception:
        # Silently handle errors to reduce output
        raise

def run_mutation_method(mutation_method, mutation_params, base_config, image_path, output_dir):
    """
    Runs the GA for one mutation method across all patches, saving the composite image.
    Returns results dictionary.
    """
    print(f"Starting mutation method: {mutation_method}", flush=True)
    
    # Update config with mutation method and params
    config = base_config.copy()
    config["mutation_method"] = mutation_method
    config.update(mutation_params)
    config["output_image_name"] = f"composite_{mutation_method}.png"
    
    # Load full target image
    full_target = load_image(image_path)
    full_w, full_h = full_target.width, full_target.height

    # Grid setup
    n_triangles_global = config.get("n_triangles_global", 1000)
    tri_per_patch_desired = config.get("tri_per_patch_desired", 20)
    total_patches = max(1, round(n_triangles_global / tri_per_patch_desired))
    grid_size = max(1, round(math.sqrt(total_patches)))
    grid_rows = grid_cols = grid_size
    n_patches = grid_rows * grid_cols
    triangles_per_patch = max(1, n_triangles_global // n_patches)

    # Patch dimensions
    nom_patch_w = full_w // grid_cols
    nom_patch_h = full_h // grid_rows
    overlap_frac = config.get("overlap_frac", 0.2)

    # Build tasks
    tasks = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            left_nom = col * nom_patch_w
            upper_nom = row * nom_patch_h
            right_nom = left_nom + nom_patch_w if col < grid_cols - 1 else full_w
            lower_nom = upper_nom + nom_patch_h if row < grid_rows - 1 else full_h
            nominal_rect = (left_nom, upper_nom, right_nom, lower_nom)
            
            x_margin = int(nom_patch_w * overlap_frac)
            y_margin = int(nom_patch_h * overlap_frac)
            left = max(0, left_nom - x_margin)
            upper = max(0, upper_nom - y_margin)
            right = min(full_w, right_nom + x_margin)
            lower = min(full_h, lower_nom + y_margin)
            patch_box = (left, upper, right, lower)
            
            patch = full_target.crop(patch_box)
            local_config = config.copy()
            local_config["n_triangles"] = triangles_per_patch
            tasks.append((patch_box, nominal_rect, local_config, patch, full_target))
    
    # Run tasks in parallel with processes
    start_time = time.time()
    num_workers = min(len(tasks), os.cpu_count() or 1)
    results = []
    completed = set()
    progress_interval = 0.1  # Update every 10%
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_patch_ga, tb, nr, conf, p, gt)
                   for tb, nr, conf, p, gt in tasks]
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                # Calculate and print progress
                progress = len(results) / len(tasks)
                progress_pct = int(progress * 100)
                if progress_pct / 100 >= len(completed) * progress_interval:
                    completed.add(progress_pct // 10)
                    print(f"Progress: {progress_pct}% done", flush=True)
            except Exception:
                pass  # Silently skip failed patches

    # Composite image
    composite_array = np.zeros((full_h, full_w, 4), dtype=np.float32)
    weight_array = np.zeros((full_h, full_w), dtype=np.float32)
    
    for patch_box, nominal_rect, best_individual in results:
        left, upper, right, lower = patch_box
        patch_w_run = right - left
        patch_h_run = lower - upper
        try:
            if not hasattr(best_individual, 'genes'):
                continue
            patch_result_img = render_individual(best_individual, patch_w_run, patch_h_run)
            patch_result_arr = np.array(patch_result_img, dtype=np.float32) / 255.0
        except Exception:
            continue
        
        left_nom, upper_nom, right_nom, lower_nom = nominal_rect
        left_margin = max(1, left_nom - left)
        right_margin = max(1, right - right_nom)
        top_margin = max(1, upper_nom - upper)
        bottom_margin = max(1, lower_nom - lower)
        mask = create_blend_mask(patch_w_run, patch_h_run, left_margin, right_margin, 
                                 top_margin, bottom_margin)
        
        composite_array[upper:lower, left:right, :] += patch_result_arr * mask
        weight_array[upper:lower, left:right] += mask[:, :, 0]
    
    weight_array[weight_array == 0] = 1.0
    normalized = composite_array / weight_array[:, :, None]
    normalized = (np.clip(normalized, 0, 1) * 255).astype(np.uint8)
    final_composite = Image.fromarray(normalized, mode="RGBA")

    # Save result
    output_path = os.path.join(output_dir, config["output_image_name"])
    final_composite.save(output_path)
    runtime = time.time() - start_time

    # Compute average fitness (for JSON, not display)
    fitnesses = [ind.fitness for _, _, ind in results if hasattr(ind, 'fitness') and ind.fitness is not None]
    avg_fitness = np.mean(fitnesses) if fitnesses else 0.0

    print(f"Finished mutation method: {mutation_method}", flush=True)
    return {
        "mutation_method": mutation_method,
        "avg_fitness": float(avg_fitness),  # Ensure JSON serializable
        "runtime": runtime,
        "output_path": output_path,
        "n_patches": len(results)
    }

def plot_mutation_comparisons(results, output_dir):
    """
    Generates bar plots comparing average fitness and runtime across mutation methods.
    Saves plots to output_dir without printing.
    """
    valid_results = [r for r in results if r["avg_fitness"] is not None and not np.isnan(r["avg_fitness"])]
    if not valid_results:
        return

    methods = [r["mutation_method"] for r in valid_results]
    fitnesses = [r["avg_fitness"] for r in valid_results]
    runtimes = [r["runtime"] for r in valid_results]

    # Fitness plot
    plt.figure(figsize=(8, 5))
    plt.bar(methods, fitnesses, color='skyblue')
    plt.xlabel("Mutation Methods")
    plt.ylabel("Average Fitness")
    plt.title("Comparison of Average Fitness for Mutation Methods")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fitness_comparison.png"))
    plt.close()

    # Runtime plot
    plt.figure(figsize=(8, 5))
    plt.bar(methods, runtimes, color='lightcoral')
    plt.xlabel("Mutation Methods")
    plt.ylabel("Runtime (seconds)")
    plt.title("Comparison of Runtime for Mutation Methods")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_comparison.png"))
    plt.close()

def main_analysis(config_path, image_path):
    """
    Analyzes all mutation methods using a config with shared and mutation-specific settings.
    Saves results and comparison plots in analysis_results/mutation_methods/.
    """
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    base_config = config.get("shared", {})
    mutation_configs = config.get("mutation_configs", {})

    # Output directory
    output_dir = "analysis_results/mutation_methods"
    os.makedirs(output_dir, exist_ok=True)

    # Get mutation methods
    from ga.mutation import mutation_strategies
    available_methods = set(mutation_strategies.keys())
    configured_methods = set(mutation_configs.keys())
    methods_to_run = available_methods.intersection(configured_methods)
    if not methods_to_run:
        print("No valid mutation methods found in both mutation_strategies and config.")
        sys.exit(1)

    # Run each method
    results = []
    for method in methods_to_run:
        try:
            result = run_mutation_method(
                method, 
                mutation_configs[method], 
                base_config, 
                image_path, 
                output_dir
            )
            results.append(result)
        except Exception:
            results.append({
                "mutation_method": method,
                "avg_fitness": None,
                "runtime": None,
                "output_path": None,
                "n_patches": 0,
                "error": "Failed"
            })

    # Save summary
    summary_path = os.path.join(output_dir, "mutation_methods_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print minimal summary
    completed_methods = [r["mutation_method"] for r in results if r["avg_fitness"] is not None]
    print(f"Analysis complete: {completed_methods}", flush=True)

    # Generate comparison plots silently
    plot_mutation_comparisons(results, output_dir)

if __name__ == "__main__":
    # Handle command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python3 analysis/mutation_methods.py <config_path> <image_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    image_path = sys.argv[2]

    # Verify files
    for path in [config_path, image_path]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)

    main_analysis(config_path, image_path)