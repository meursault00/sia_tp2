import sys
import os
import json
from multiprocessing import Pool
import matplotlib.pyplot as plt
from ga.algorithm import run_ga
from utils.image_utils import load_image
from utils.render import render_individual
from ga.selection import selection_strategies
from ga.crossover import crossover_strategies
from ga.mutation import mutation_strategies

# Available methods
SELECTION_METHODS = list(selection_strategies.keys())  # ["tournament", "roulette", "ranking", "boltzmann"]
CROSSOVER_METHODS = list(crossover_strategies.keys())  # ["one_point", "two_point", "uniform"]
MUTATION_METHODS = list(mutation_strategies.keys())   # ["basic", "multi_gen"]
METHOD_MAP = {
    "selection": SELECTION_METHODS,
    "crossover": CROSSOVER_METHODS,
    "mutation": MUTATION_METHODS
}

def run_ga_with_method(args):
    """Runs GA with a specific method, returning method name, best individual, and fitness."""
    config, target_image, method_type, method = args
    config = config.copy()
    config[f"{method_type}_method"] = method
    if method_type == "selection" and method == "tournament":
        config["tournament_size"] = config.get("tournament_size", 20)
    best_individual = run_ga(config, target_image)
    return method, best_individual, best_individual.fitness

def analyze_methods(base_config_path, test_config_path, image_path, output_dir="analysis_results"):
    """
    Runs GA in parallel for the method type specified in test_config, using base_config as defaults.
    """
    # Load configs
    with open(base_config_path, "r") as f:
        base_config = json.load(f)
    with open(test_config_path, "r") as f:
        test_config = json.load(f)
    
    # Get the variable to test
    method_type = test_config.get("test_variable")
    if method_type not in METHOD_MAP:
        raise ValueError(f"Invalid test_variable in {test_config_path}: {method_type}. Must be 'selection', 'crossover', or 'mutation'")
    methods = METHOD_MAP[method_type]

    # Load image
    target_image = load_image(image_path)
    w, h = target_image.width, target_image.height

    # Prepare tasks
    tasks = [(base_config, target_image, method_type, method) for method in methods]

    # Run in parallel
    num_processes = min(os.cpu_count(), len(methods))
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_ga_with_method, tasks)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process results
    fitness_values = {}
    for method, best_individual, fitness in results:
        best_img = render_individual(best_individual, w, h)
        filename = f"best_{method_type}_{method}.png"
        output_path = os.path.join(output_dir, filename)
        best_img.save(output_path)
        print(f"{method_type}/{method}: Fitness = {fitness:.4f}, Saved as {output_path}")
        fitness_values[method] = fitness

    # Plot
    plt.figure(figsize=(8, 5))
    for method, fitness in fitness_values.items():
        plt.bar(method, fitness, label=f"{method} (Fitness: {fitness:.4f})")
    plt.xlabel(f"{method_type.capitalize()} Method")
    plt.ylabel("Fitness")
    plt.title(f"Fitness Comparison for {method_type.capitalize()} Methods")
    plt.legend()
    plt.grid(True, axis="y")
    plot_path = os.path.join(output_dir, f"{method_type}_fitness_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"{method_type.capitalize()} plot saved as {plot_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python analysis.py <base_config.json> <test_config.json> <image_path>")
        sys.exit(1)
    
    base_config_path = sys.argv[1]
    test_config_path = sys.argv[2]
    image_path = sys.argv[3]
    
    analyze_methods(base_config_path, test_config_path, image_path)