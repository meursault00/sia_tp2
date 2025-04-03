import json
from ga.algorithm import run_ga
from utils.image_utils import load_image

def main():
    # 1. Cargar configuración desde JSON
    with open("configs/ga_config.json", "r") as f:
        config = json.load(f)
    
    # 2. Cargar imagen objetivo
    target_image = load_image("images/bosch.jpg")  # Ajusta a tu ruta
    
    # 3. Ejecutar GA
    best_individual = run_ga(config, target_image)
    
    print("\n=== GA Finished ===")
    print("Best individual's fitness:", best_individual.fitness)
    
    # 4. Opcional: guardar la imagen del mejor individuo
    from utils.render import render_individual
    w, h = target_image.width, target_image.height
    best_image = render_individual(best_individual, w, h)
    best_image.save("best_result.png")
    print("Best individual's image saved as best_result.png")

if __name__ == "__main__":
    main()