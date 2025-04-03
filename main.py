import sys
import os
import json
from ga.algorithm import run_ga
from utils.image_utils import load_image

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <config_path.json> <image_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    image_path = sys.argv[2]

    # 1. Load configuration from JSON
    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Load target image
    target_image = load_image(image_path)

    # 3. Run GA
    best_individual = run_ga(config, target_image)

    print("\n=== GA Finished ===")
    print("Best individual's fitness:", best_individual.fitness)

    # 4. Render and save the best individual's image
    from utils.render import render_individual
    w, h = target_image.width, target_image.height
    best_image = render_individual(best_individual, w, h)

    # Get the output image name from config (default if not found)
    output_image_name = config.get("output_image_name", "best_result.png")

    # Ensure the "results" folder exists
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Construct the full path and save
    output_file_path = os.path.join(results_folder, output_image_name)
    best_image.save(output_file_path)

    print(f"Best individual's image saved as {output_file_path}")

if __name__ == "__main__":
    main()