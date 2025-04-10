import numpy as np
from utils.render import render_individual
from skimage.metrics import structural_similarity
from PIL import Image

def compute_fitness(individual, target_array, w, h):
    """
    Computes fitness by combining Mean Squared Error (MSE) and Structural Similarity (SSIM)
    on downsampled images, with a penalty for triangle complexity, to optimize for perceptual
    quality and efficiency.
    
    Parameters:
      individual: Candidate solution with 'genes' (list of triangles).
      target_array: Precomputed NumPy array of target image (RGBA, float32).
      w, h: Original dimensions of the image.
    
    Returns:
      float: Fitness value in [0, 1], higher is better.
    """
    # Downsample to max 200x200 for speed
    max_dim = 200
    scale = min(max_dim / w, max_dim / h) if w > max_dim or h > max_dim else 1
    new_w, new_h = max(7, int(w * scale)), max(7, int(h * scale))  # Ensure min 7x7 for SSIM
    
    # Render and convert generated image
    generated_img = render_individual(individual, new_w, new_h)
    arr_generated = np.array(generated_img.convert("RGBA"), dtype=np.float32)
    
    # Resize target array to match
    target_pil = Image.fromarray(target_array.astype(np.uint8), mode="RGBA")
    target_resized = target_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    arr_target = np.array(target_resized, dtype=np.float32)
    
    # Component 1: Mean Squared Error (normalized)
    mse = np.mean((arr_target - arr_generated) ** 2)
    mse_max = 255 ** 2 * 4  # Max MSE for RGBA (255^2 per channel)
    mse_score = 1 - (mse / mse_max)  # [0, 1], 1 is perfect
    
    # Component 2: Structural Similarity (SSIM)
    win_size = 7  # Set explicitly to 7; 7 is odd and 7 <= new_w, new_h (which are at least 7)
    ssim_score = structural_similarity(arr_target, arr_generated, channel_axis=-1, 
                                       data_range=255, win_size=win_size)
    
    # Component 3: Complexity Penalty
    visible_triangles = sum(1 for tri in individual.genes 
                           if abs((tri[0]*(tri[3]-tri[5]) + tri[2]*(tri[5]-tri[1]) + 
                                   tri[4]*(tri[1]-tri[3])) / 2) > 1)  # Count visible (area > 1)
    complexity_penalty = 0.001 * (len(individual.genes) - visible_triangles)  # Penalize degenerate
    complexity_factor = max(0.9, 1 - complexity_penalty)  # Cap at 0.9 to avoid over-penalizing
    
    # Combine: 40% MSE, 40% SSIM, 20% complexity-adjusted
    fitness = (0.4 * mse_score + 0.4 * ssim_score) * complexity_factor
    
    print(f"MSE: {mse:.2f}, SSIM: {ssim_score:.4f}, Visible: {visible_triangles}/{len(individual.genes)}, Fitness: {fitness:.6f}", flush=True)
    return max(0.001, fitness)  # Ensure non-zero fitness
