import numpy as np
from utils.render import render_individual
from skimage import color
from skimage.metrics import structural_similarity
from scipy import ndimage
from PIL import Image

def compute_fitness(individual, target_array, w, h, generation=0, max_generations=100):
    """
    Computes fitness using multiple perceptual metrics optimized for transparent triangles.
    
    Parameters:
      individual: Candidate solution with 'genes' (list of triangles).
      target_array: Precomputed NumPy array of target image (RGBA, float32).
      w, h: Original dimensions of the image.
      generation: Current generation number (for adaptive weighting)
      max_generations: Total number of generations (for adaptive weighting)
    
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
    
    # Create importance map (edges are more important)
    target_gray = np.mean(arr_target[:,:,:3], axis=2)
    importance = ndimage.sobel(target_gray)
    importance = importance / np.max(importance) if np.max(importance) > 0 else importance
    importance = importance.reshape(*importance.shape, 1) + 0.5  # Add baseline importance
    
    # COMPONENT 1: LAB COLOR SPACE COMPARISON
    # Extract RGB channels
    target_rgb = arr_target[:, :, :3] / 255.0  # Normalize to [0,1]
    generated_rgb = arr_generated[:, :, :3] / 255.0
    
    # Convert to LAB color space
    try:
        target_lab = color.rgb2lab(target_rgb)
        generated_lab = color.rgb2lab(generated_rgb)
        
        # Split LAB channels and weight differently (L=luminance, A/B=chrominance)
        l_diff = target_lab[:,:,0] - generated_lab[:,:,0]
        a_diff = target_lab[:,:,1] - generated_lab[:,:,1]
        b_diff = target_lab[:,:,2] - generated_lab[:,:,2]
        
        # Higher weight for luminance (L) channel
        l_mse = np.mean((l_diff * importance[:,:,0]) ** 2)
        a_mse = np.mean((a_diff * importance[:,:,0]) ** 2)
        b_mse = np.mean((b_diff * importance[:,:,0]) ** 2)
        
        # Combined weighted LAB MSE
        lab_mse = (0.6 * l_mse + 0.2 * a_mse + 0.2 * b_mse)
        lab_score = 1.0 / (1.0 + lab_mse)
    except:
        # Fallback to regular MSE if LAB conversion fails
        diff = arr_target - arr_generated
        weighted_diff = diff * importance
        mse = np.mean(weighted_diff ** 2)
        mse_max = 255 ** 2 * 4
        lab_score = 1 - (mse / mse_max)
    
    # COMPONENT 2: Standard MSE as backup
    diff = arr_target - arr_generated
    weighted_diff = diff * importance
    mse = np.mean(weighted_diff ** 2)
    mse_max = 255 ** 2 * 4
    mse_score = 1 - (mse / mse_max)
    
    # COMPONENT 3: Structural Similarity (SSIM)
    win_size = 7  # Set explicitly to 7; 7 is odd and 7 <= new_w, new_h (which are at least 7)
    ssim_score = structural_similarity(arr_target, arr_generated, channel_axis=-1, 
                                      data_range=255, win_size=win_size)
    
    # COMPONENT 4: Edge Detection Comparison
    # Calculate edge detection score
    target_edges = ndimage.sobel(target_gray)
    generated_gray = np.mean(arr_generated[:,:,:3], axis=2)
    generated_edges = ndimage.sobel(generated_gray)
    
    # Multi-scale edge detection
    edge_scores = []
    scales = [1, 2]  # Different scales
    
    for scale in scales:
        if scale > 1:
            target_blurred = ndimage.gaussian_filter(target_gray, sigma=scale)
            generated_blurred = ndimage.gaussian_filter(generated_gray, sigma=scale)
            target_edges_scale = ndimage.sobel(target_blurred)
            generated_edges_scale = ndimage.sobel(generated_blurred)
        else:
            target_edges_scale = target_edges
            generated_edges_scale = generated_edges
            
        edge_diff = target_edges_scale - generated_edges_scale
        edge_mse = np.mean(edge_diff ** 2)
        edge_scores.append(1.0 / (1.0 + 0.01 * edge_mse))
    
    edge_score = 0.6 * edge_scores[0] + 0.4 * edge_scores[1]
    
    # COMPONENT 5: Improved Complexity Penalty
    # Consider both area AND alpha when determining if a triangle is "visible"
    visible_triangles = sum(1 for tri in individual.genes 
                           if (abs((tri[0]*(tri[3]-tri[5]) + tri[2]*(tri[5]-tri[1]) + 
                                    tri[4]*(tri[1]-tri[3])) / 2) > 1) and (tri[9] > 0.1))
    
    total_triangles = len(individual.genes)
    complexity_penalty = 0.002 * (total_triangles - visible_triangles)
    complexity_factor = max(0.8, 1 - complexity_penalty)  # Stronger penalty
    
    # COMPONENT 6: Transparency penalty
    avg_alpha = sum(tri[9] for tri in individual.genes) / len(individual.genes)
    transparency_penalty = 0.2 * (1 - avg_alpha) if avg_alpha < 0.4 else 0
    
    # ADAPTIVE WEIGHTING based on generation progress
    progress = min(1.0, generation / max_generations) if max_generations > 0 else 0.5
    
    # Early generations: focus on color matching
    # Later generations: focus more on structure and edges
    color_weight = 0.5 - (0.2 * progress)  # Decreases from 0.5 to 0.3
    struct_weight = 0.3                    # Stays constant
    edge_weight = 0.2 + (0.2 * progress)   # Increases from 0.2 to 0.4
    
    # Combined score with adaptive weights
    visual_score = (color_weight * lab_score + 
                  struct_weight * ssim_score + 
                  edge_weight * edge_score)
    
    # Apply complexity and transparency penalties
    fitness = (visual_score * complexity_factor) - transparency_penalty
    
    # Ensure fitness is in valid range
    fitness = max(0.001, min(1.0, fitness))
    
    # Log key metrics
    print(f"MSE: {mse:.2f}, SSIM: {ssim_score:.4f}, Edge: {edge_score:.4f}, "
          f"Visible: {visible_triangles}/{total_triangles}, Alpha: {avg_alpha:.2f}, "
          f"Fitness: {fitness:.6f}", flush=True)
    
    return fitness