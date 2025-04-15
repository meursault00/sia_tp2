#!/usr/bin/env python3
import os
import json
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from PIL import Image
import traceback

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Config setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sampling_experiment")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define sampling rates to test
SAMPLING_RATES = [0.0, 0.5, 1.0]

# Define test images - CHECK THESE PATHS!
TEST_IMAGES = [
    {"name": "photo", "path": os.path.join(IMAGES_DIR, "saints.jpeg")},
    {"name": "painting", "path": os.path.join(IMAGES_DIR, "bosch.jpg")}, 
    {"name": "flag", "path": os.path.join(IMAGES_DIR, "banderas", "argentina.png")}
]

# Verify image paths exist
for img in TEST_IMAGES[:]:
    if not os.path.exists(img["path"]):
        print(f"WARNING: Image path doesn't exist: {img['path']}")
        TEST_IMAGES.remove(img)

if not TEST_IMAGES:
    print("ERROR: No valid image paths found. Check the TEST_IMAGES paths.")
    sys.exit(1)

# Add debug mode for quicker testing
DEBUG_MODE = False  # Set to True for smaller test runs

if DEBUG_MODE:
    # Override settings for faster debugging
    SAMPLING_RATES = [0.0]  # Just one rate
    TEST_IMAGES = TEST_IMAGES[:1]  # Just one image
    
    # Add a debug function that automatically modifies create_config_files
    def debug_config():
        """Creates a minimal config for debugging"""
        return {
            "n_triangles_global": 100,
            "tri_per_patch_desired": 5,
            "population_size": 10,
            "n_generations": 10,
            "parents_size": 5,
            "crossover_rate": 0.9,
            "mutation_rate": 0.05,
            "intermediate_images": 2,  # Just 2 snapshots
            "selection_method": "tournament",
            "crossover_method": "one_point",
            "mutation_method": "basic", 
            "separation_method": "traditional",
            "fitness_mode": "interpolation",
            "overlap_frac": 0.2,
            "max_dim": 150,
        }

def create_config_files():
    """Creates config files for each sampling rate"""
    print("Creating config files...")
    base_config = {
        "n_triangles_global": 1000,
        "tri_per_patch_desired": 20,
        "population_size": 50,
        "n_generations": 500,
        "parents_size": 30,
        "crossover_rate": 0.9,
        "mutation_rate": 0.05,
        "intermediate_images": 5,  # This comment will be removed in JSON
        "selection_method": "tournament",
        "crossover_method": "one_point",
        "mutation_method": "basic", 
        "separation_method": "traditional",
        "fitness_mode": "interpolation",
        "overlap_frac": 0.2,
        "max_dim": 300,  # Use smaller images for faster testing
    }
    
    if DEBUG_MODE:
        base_config = debug_config()
    
    # Create config files
    for rate in SAMPLING_RATES:
        config = base_config.copy()
        config["use_image_sampling"] = True if rate > 0 else False
        config["sampling_rate"] = rate
        config["output_image_name"] = f"sampling_{rate}_results.png"
        
        config_path = os.path.join(CONFIG_DIR, f"sampling_{rate}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        
        print(f"Created config file: {config_path}")
    
    return [os.path.join(CONFIG_DIR, f"sampling_{rate}.json") for rate in SAMPLING_RATES]

def run_experiments(config_files):
    """Run experiments for all combinations of config files and test images"""
    print("\nRunning experiments...")
    results = {}
    run_times = {}
    
    for img in TEST_IMAGES:
        print(f"\nTesting image: {img['name']} ({img['path']})")
        results[img['name']] = {}
        run_times[img['name']] = {}
        
        for i, rate in enumerate(SAMPLING_RATES):
            # Create a unique experiment directory for each image and rate
            exp_dir = os.path.join(OUTPUT_DIR, f"{img['name']}_{rate}")
            os.makedirs(exp_dir, exist_ok=True)
            
            config_path = config_files[i]
            # Use separate output name for each image/rate combination
            modified_config_path = os.path.join(CONFIG_DIR, f"temp_{img['name']}_{rate}.json")
            
            # Read the original config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Modify the output name to include image name
            config["output_image_name"] = f"result_{img['name']}_{rate}.png"
            
            # Save modified config
            with open(modified_config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Set environment variable to record fitness history
            os.environ["RECORD_FITNESS"] = "1"
            os.environ["FITNESS_OUTPUT"] = os.path.join(exp_dir, "fitness.csv")
            
            print(f"Running test with sampling_rate={rate} on image {img['name']}...")
            
            # Record start time
            start_time = time.time()
            
            # Run the experiment with modified config
            cmd = [
                "python", 
                os.path.join(BASE_DIR, "main.py"), 
                modified_config_path, 
                img["path"]
            ]
            
            try:
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                # Record end time
                end_time = time.time()
                run_time = end_time - start_time
                run_times[img['name']][rate] = run_time
                
                if process.returncode != 0:
                    print(f"Error running experiment:")
                    print(process.stderr)
                    continue
                
                print(f"Completed in {run_time:.2f} seconds")
                
                # Look for all result files in results directory
                results_dir = os.path.join(BASE_DIR, "results")
                if os.path.exists(results_dir):
                    result_files = []
                    
                    # Look for the main output file
                    expected_output = os.path.join(results_dir, f"result_{img['name']}_{rate}.png")
                    if os.path.exists(expected_output):
                        result_files.append(expected_output)
                    
                    # Look for generation snapshots
                    for file in os.listdir(results_dir):
                        if file.startswith(f"result_{img['name']}_{rate}_gen") and file.endswith(".png"):
                            result_files.append(os.path.join(results_dir, file))
                    
                    if result_files:
                        # Copy all files to experiment directory
                        import shutil
                        for file_path in result_files:
                            dest_path = os.path.join(exp_dir, os.path.basename(file_path))
                            shutil.copy(file_path, dest_path)
                            print(f"Copied {file_path} to {dest_path}")
                        
                        # Use the final result for comparison
                        final_result = os.path.join(exp_dir, f"result_{img['name']}_{rate}.png")
                        
                        # Store results
                        results[img['name']][rate] = {
                            "output_path": final_result if os.path.exists(final_result) else result_files[0],
                            "fitness_path": os.environ["FITNESS_OUTPUT"],
                            "all_outputs": [os.path.join(exp_dir, os.path.basename(f)) for f in result_files]
                        }
                    else:
                        print(f"No result files found in {results_dir}")
                else:
                    print(f"Results directory not found: {results_dir}")
                    
            except Exception as e:
                print(f"Exception running experiment: {e}")
                traceback.print_exc()
                run_times[img['name']][rate] = 0
    
    return results, run_times

def extract_fitness_data(results):
    """Extract fitness data from CSV files"""
    fitness_data = {}
    
    for img_name, img_results in results.items():
        fitness_data[img_name] = {}
        
        for rate, paths in img_results.items():
            try:
                fitness_path = paths["fitness_path"]
                if os.path.exists(fitness_path):
                    data = np.loadtxt(fitness_path, delimiter=',', skiprows=1)  # Skip header
                    fitness_data[img_name][rate] = data[:, 1]  # Use best fitness column
                else:
                    print(f"Warning: Fitness data file not found: {fitness_path}")
                    fitness_data[img_name][rate] = None
            except Exception as e:
                print(f"Error loading fitness data for {img_name} at rate {rate}: {e}")
                fitness_data[img_name][rate] = None
    
    return fitness_data

def plot_fitness_curves(fitness_data, run_times):
    """Plot fitness curves for all experiments"""
    print("\nGenerating fitness plots...")
    
    plt.figure(figsize=(18, 12))
    
    for i, (img_name, img_fitness) in enumerate(fitness_data.items()):
        plt.subplot(1, len(fitness_data), i+1)
        
        has_data = False
        for rate, data in img_fitness.items():
            if data is not None and len(data) > 0:
                has_data = True
                plt.plot(data, label=f"Rate {rate} ({run_times[img_name].get(rate, 0):.1f}s)")
        
        plt.title(f"Fitness: {img_name.capitalize()}")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        if has_data:
            plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "fitness_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved fitness plot to {plot_path}")
    
    # Create a plot of final fitness values
    plt.figure(figsize=(12, 6))
    
    # Get the final fitness for each experiment
    final_fitness = {}
    for img_name, img_fitness in fitness_data.items():
        final_fitness[img_name] = {}
        for rate, data in img_fitness.items():
            if data is not None and len(data) > 0:
                final_fitness[img_name][rate] = data[-1]
    
    # Check if we have any data to plot
    has_data = any(len(img) > 0 for img in final_fitness.values())
    
    if has_data:
        # Plot as bar chart
        img_names = list(final_fitness.keys())
        rates = SAMPLING_RATES
        x = np.arange(len(img_names))
        width = 0.2
        
        for i, rate in enumerate(rates):
            values = [final_fitness[img].get(rate, 0) if img in final_fitness else 0 for img in img_names]
            plt.bar(x + (i-1)*width, values, width, label=f"Rate {rate}")
        
        plt.xlabel("Image Type")
        plt.ylabel("Final Fitness")
        plt.title("Final Fitness Comparison")
        plt.xticks(x, img_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        bar_path = os.path.join(OUTPUT_DIR, "final_fitness_comparison.png")
        plt.savefig(bar_path, dpi=300)
        print(f"Saved final fitness comparison to {bar_path}")
    else:
        print("No fitness data available for final comparison")

def create_visual_grid(results):
    """Create a grid showing visual results of all experiments"""
    print("\nGenerating visual comparison grid...")
    
    # Check if we have any valid results
    has_results = False
    for img_name in results:
        for rate in results[img_name]:
            output_path = results[img_name][rate].get("output_path")
            if output_path and os.path.exists(output_path):
                has_results = True
                break
        if has_results:
            break
    
    if not has_results:
        print("No valid result images found for visual comparison")
        return
    
    # Part 1: Basic grid comparing final results
    n_rows = len(SAMPLING_RATES) + 1  # +1 for original images
    n_cols = len(TEST_IMAGES)
    
    plt.figure(figsize=(n_cols*5, n_rows*5))
    
    # First row: original images
    for i, img in enumerate(TEST_IMAGES):
        plt.subplot(n_rows, n_cols, i+1)
        try:
            original_img = Image.open(img["path"])
            plt.imshow(np.array(original_img))
            plt.title(f"Original: {img['name']}")
            plt.axis('off')
        except Exception as e:
            print(f"Error loading original image {img['path']}: {e}")
    
    # Remaining rows: results for each sampling rate
    for i, rate in enumerate(SAMPLING_RATES):
        for j, img in enumerate(TEST_IMAGES):
            img_name = img["name"]
            if img_name in results and rate in results[img_name]:
                output_path = results[img_name][rate].get("output_path")
                if output_path and os.path.exists(output_path):
                    plt.subplot(n_rows, n_cols, (i+1)*n_cols + j+1)
                    try:
                        result_img = Image.open(output_path)
                        plt.imshow(np.array(result_img))
                        plt.title(f"{img_name}, Rate={rate}")
                        plt.axis('off')
                    except Exception as e:
                        print(f"Error loading result image {output_path}: {e}")
    
    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, "visual_comparison.png")
    plt.savefig(grid_path, dpi=300)
    print(f"Saved final results comparison to {grid_path}")
    
    # Part 2: Evolution grids for each image type and sampling rate
    for img_name in results:
        for rate in results[img_name]:
            # Check if we have multiple outputs for this experiment
            if "all_outputs" in results[img_name][rate]:
                all_outputs = results[img_name][rate]["all_outputs"]
                
                # Filter to just snapshot generations
                snapshots = [f for f in all_outputs if "gen" in os.path.basename(f)]
                
                if snapshots:
                    # Sort by generation number
                    snapshots = sorted(snapshots, key=lambda x: int(os.path.basename(x).split("gen")[1].split(".")[0]))
                    
                    # Create a grid for this experiment
                    n_images = len(snapshots) + 1  # +1 for original
                    rows = 1
                    cols = n_images
                    
                    plt.figure(figsize=(cols*4, rows*4))
                    
                    # Original image
                    plt.subplot(rows, cols, 1)
                    try:
                        for img in TEST_IMAGES:
                            if img["name"] == img_name:
                                original_img = Image.open(img["path"])
                                plt.imshow(np.array(original_img))
                                plt.title(f"Original")
                                plt.axis('off')
                                break
                    except Exception as e:
                        print(f"Error loading original image: {e}")
                    
                    # Snapshots
                    for i, snapshot in enumerate(snapshots):
                        plt.subplot(rows, cols, i+2)
                        try:
                            snap_img = Image.open(snapshot)
                            plt.imshow(np.array(snap_img))
                            gen_num = os.path.basename(snapshot).split("gen")[1].split(".")[0]
                            plt.title(f"Gen {gen_num}")
                            plt.axis('off')
                        except Exception as e:
                            print(f"Error loading snapshot {snapshot}: {e}")
                    
                    plt.tight_layout()
                    evolution_path = os.path.join(OUTPUT_DIR, f"evolution_{img_name}_rate{rate}.png")
                    plt.savefig(evolution_path, dpi=300)
                    print(f"Saved evolution grid for {img_name} (rate={rate}) to {evolution_path}")

def main():
    """Main function to run the entire experiment"""
    print("Starting Sampling Rate vs Convergence Experiment")
    
    # Create config files
    config_files = create_config_files()
    
    # Run experiments
    results, run_times = run_experiments(config_files)
    
    # Extract fitness data
    fitness_data = extract_fitness_data(results)
    
    # Plot fitness curves
    plot_fitness_curves(fitness_data, run_times)
    
    # Create visual comparison
    create_visual_grid(results)
    
    print("\nExperiment completed! Results are in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()