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

def run_sampling_condition(sampling_rate, mutation_method, base_config, image_path, output_dir):
    """
    Runs the GA for one sampling rate and mutation method across all patches, saving the composite image.
    Returns results dictionary with max and min fitness.
    """
    condition_name = f"sampling_{sampling_rate}_mutation_{mutation_method}"
    print(f"Starting condition: {condition_name}", flush=True)
    
    # Update config
    config = base_config.copy()
    config["sampling_rate"] = sampling_rate
    config["mutation_method"] = mutation_method
    config["output_image_name"] = f"composite_{condition_name}.png"
    
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

    # Compute fitness statistics
    fitnesses = [ind.fitness for _, _, ind in results if hasattr(ind, 'fitness') and ind.fitness is not None]
    avg_fitness = np.mean(fitnesses) if fitnesses else 0.0
    max_fitness = np.max(fitnesses) if fitnesses else 0.0
    min_fitness = np.min(fitnesses) if fitnesses else 0.0

    print(f"Finished condition: {condition_name}", flush=True)
    return {
        "sampling_rate": sampling_rate,
        "mutation_method": mutation_method,
        "avg_fitness": float(avg_fitness),  # Ensure JSON serializable
        "max_fitness": float(max_fitness),
        "min_fitness": float(min_fitness),
        "runtime": runtime,
        "output_path": output_path,
        "n_patches": len(results)
    }

def plot_sampling_comparisons(results, output_dir):
    """
    Generates bar plots comparing average fitness and runtime for exactly three sampling rates.
    Fitness plot includes error lines for max and min fitness.
    Saves plots to output_dir without printing.
    """
    valid_results = [r for r in results if r["avg_fitness"] is not None and not np.isnan(r["avg_fitness"])]
    if not valid_results:
        return

    # Fixed sampling rates for exactly three pairs
    sampling_rates = [0.0, 0.5, 1.0]
    # Get mutation methods (expecting two, e.g., basic, multi_gen)
    mutation_methods = sorted(set(r["mutation_method"] for r in valid_results))
    
    # Prepare data for three pairs of bars
    fitness_data = {m: [0.0] * 3 for m in mutation_methods}
    max_fitness_data = {m: [0.0] * 3 for m in mutation_methods}
    min_fitness_data = {m: [0.0] * 3 for m in mutation_methods}
    runtime_data = {m: [0.0] * 3 for m in mutation_methods}
    for i, rate in enumerate(sampling_rates):
        for method in mutation_methods:
            # Find matching result
            result = next((r for r in valid_results 
                          if r["sampling_rate"] == rate and r["mutation_method"] == method), None)
            if result:
                fitness_data[method][i] = result["avg_fitness"]
                max_fitness_data[method][i] = result["max_fitness"]
                min_fitness_data[method][i] = result["min_fitness"]
                runtime_data[method][i] = result["runtime"]

    # Plot settings
    bar_width = 0.35  # Fixed for two bars per group
    x = np.arange(len(sampling_rates))  # Positions for three groups
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(mutation_methods), 2)))  # At least two colors

    # Fitness plot with error lines
    plt.figure(figsize=(8, 6))
    for i, method in enumerate(mutation_methods):
        offset = i * bar_width - bar_width / 2 * (len(mutation_methods) - 1)
        # Plot bars
        bars = plt.bar(x + offset, fitness_data[method], bar_width, label=method, color=colors[i])
        # Compute error bounds
        yerr_lower = [fitness_data[method][j] - min_fitness_data[method][j] for j in range(len(sampling_rates))]
        yerr_upper = [max_fitness_data[method][j] - fitness_data[method][j] for j in range(len(sampling_rates))]
        # Add error bars
        plt.errorbar(x + offset, fitness_data[method], yerr=[yerr_lower, yerr_upper], fmt='none', 
                     ecolor='black', capsize=3, capthick=1, elinewidth=1)
    plt.xlabel("Sampling Rate")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness by Sampling Rate and Mutation Method")
    plt.xticks(x, [str(rate) for rate in sampling_rates])
    plt.legend(title="Mutation Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fitness_comparison.png"))
    plt.close()

    # Runtime plot (unchanged)
    plt.figure(figsize=(8, 6))
    for i, method in enumerate(mutation_methods):
        offset = i * bar_width - bar_width / 2 * (len(mutation_methods) - 1)
        plt.bar(x + offset, runtime_data[method], bar_width, label=method, color=colors[i])
    plt.xlabel("Sampling Rate")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime by Sampling Rate and Mutation Method")
    plt.xticks(x, [str(rate) for rate in sampling_rates])
    plt.legend(title="Mutation Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_comparison.png"))
    plt.close()

def plot_image_comparison(results, output_dir):
    """
    Generates a grid plot displaying composite images for each condition in landscape mode.
    Rows are mutation methods, columns are sampling rates.
    Saves plot to output_dir without printing.
    """
    valid_results = [r for r in results if r["avg_fitness"] is not None and not np.isnan(r["avg_fitness"])]
    if not valid_results:
        return

    # Fixed sampling rates and expected mutation methods
    sampling_rates = [0.0, 0.5, 1.0]
    mutation_methods = sorted(set(r["mutation_method"] for r in valid_results))
    
    # Set up figure: len(mutation_methods) rows, 3 columns (rates), landscape
    fig, axes = plt.subplots(len(mutation_methods), len(sampling_rates), 
                            figsize=(4 * len(sampling_rates), 4 * len(mutation_methods)))
    
    # Handle single method case
    if len(mutation_methods) == 1:
        axes = np.array([axes])  # Ensure 2D array for iteration
    
    # Plot images
    for i, method in enumerate(mutation_methods):
        for j, rate in enumerate(sampling_rates):
            # Find matching result
            result = next((r for r in valid_results 
                          if r["sampling_rate"] == rate and r["mutation_method"] == method), None)
            ax = axes[i, j] if len(mutation_methods) > 1 else axes[0, j]
            if result and os.path.exists(result["output_path"]):
                img = Image.open(result["output_path"]).convert("RGBA")
                # Resize to fit (maintain aspect ratio)
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                ax.imshow(img)
            else:
                # Blank placeholder for missing image
                ax.imshow(np.ones((100, 100, 4)))
            ax.set_title(f"sampling_{rate}_{method}", fontsize=8)
            ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "image_comparison.png"))
    plt.close()

def main_analysis(config_path, image_path):
    """
    Analyzes sampling rates and mutation methods using a config with shared settings.
    Saves results and comparison plots in analysis_results/sampling_analysis/.
    """
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    base_config = config.get("shared", {})
    sampling_configs = config.get("sampling_configs", [])

    # Output directory
    output_dir = "analysis_results/sampling_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Define conditions to run
    methods_to_run = [(cfg["rate"], cfg["mutation"]) for cfg in sampling_configs]
    if not methods_to_run:
        print("No valid sampling configurations found in config.")
        sys.exit(1)

    # Run each condition
    results = []
    for sampling_rate, mutation_method in methods_to_run:
        try:
            result = run_sampling_condition(
                sampling_rate, 
                mutation_method, 
                base_config, 
                image_path, 
                output_dir
            )
            results.append(result)
        except Exception:
            results.append({
                "sampling_rate": sampling_rate,
                "mutation_method": mutation_method,
                "avg_fitness": None,
                "max_fitness": None,
                "min_fitness": None,
                "runtime": None,
                "output_path": None,
                "n_patches": 0,
                "error": "Failed"
            })

    # Save summary
    summary_path = os.path.join(output_dir, "sampling_analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print minimal summary
    completed_conditions = [f"sampling_{r['sampling_rate']}_mutation_{r['mutation_method']}" 
                           for r in results if r["avg_fitness"] is not None]
    print(f"Analysis complete: {completed_conditions}", flush=True)

    # Generate comparison plots silently
    plot_sampling_comparisons(results, output_dir)
    plot_image_comparison(results, output_dir)

if __name__ == "__main__":
    # Handle command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python3 analysis/sampling_analysis.py <config_path> <image_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    image_path = sys.argv[2]

    # Verify files
    for path in [config_path, image_path]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)

    main_analysis(config_path, image_path)