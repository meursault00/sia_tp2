import sys
import json
from ga.algorithm import run_ga
from utils.image_utils import load_image

def main():
    # Check correct usage
    if len(sys.argv) != 3:
        print("Usage: python main.py <config_path.json> <image_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    image_path  = sys.argv[2]

    # 1. Load configuration from JSON
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 2. Load target image
    target_image = load_image(image_path)

    # 3. Run GA
    best_individual = run_ga(config, target_image)
    
    print("\n=== GA Finished ===")
    print("Best individual's fitness:", best_individual.fitness)
    
    # 4. Save the best individual's image using output_image_name from config
    from utils.render import render_individual
    w, h = target_image.width, target_image.height
    best_image = render_individual(best_individual, w, h)

    # if config doesn't have "output_image_name", fallback to a default
    output_image_name = config.get("output_image_name", "best_result.png")
    best_image.save(output_image_name)
    
    print(f"Best individual's image saved as {output_image_name}")

if __name__ == "__main__":
    main()