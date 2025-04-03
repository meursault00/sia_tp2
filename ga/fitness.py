# ga/fitness.py
import numpy as np
from utils.render import render_individual

def compute_fitness(individual, target_image):
    # 1) Renderizar el individuo en una imagen
    w, h = target_image.width, target_image.height
    generated_img = render_individual(individual, w, h)
    
    # 2) Convertir ambas imágenes a arrays NumPy (RGBA o RGB)
    #    target_image y generated_img deben tener el mismo tamaño
    arr_target = np.array(target_image, dtype=np.float32)
    arr_generated = np.array(generated_img, dtype=np.float32)
    
    # 3) Calcular el MSE (error cuadrático medio).
    #    Asumiendo RGBA => shape = (h, w, 4)
    diff = arr_target - arr_generated  # (h, w, 4)
    mse = np.mean(diff ** 2)
    
    # 4) Transformar el error en fitness.
    #    Cuanto menor sea el error, mayor el fitness.
    fitness = 1.0 / (1.0 + mse)
    return fitness