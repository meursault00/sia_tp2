import numpy as np
from utils.render import render_individual
from skimage import color
from skimage.metrics import structural_similarity
from scipy import ndimage

def compute_fitness(individual, target_image):
    """
    Computes fitness using multiple perceptual metrics:
    - LAB color space comparison
    - Structural similarity (SSIM)
    - Edge detection comparison
    - Triangle count penalty
    """
    # 1. Compute the dynamic bounding box from triangles
    all_x = []
    all_y = []
    for tri in individual.genes:
        x1, y1, x2, y2, x3, y3 = tri[0], tri[1], tri[2], tri[3], tri[4], tri[5]
        all_x.extend([x1, x2, x3])
        all_y.extend([y1, y2, y3])
    min_x = int(min(all_x))
    max_x = int(max(all_x))
    min_y = int(min(all_y))
    max_y = int(max(all_y))
    
    # Determine patch dimensions
    patch_width = max_x - min_x
    patch_height = max_y - min_y
    
    # Make sure the patch has a positive area
    if patch_width <= 0 or patch_height <= 0:
        return 0.0
    
    # 2. Crop the target image to the bounding box
    patch_box = (min_x, min_y, max_x, max_y)
    target_patch = target_image.crop(patch_box)
    
    # 3. Render the individual
    generated_patch = render_individual(individual, patch_width, patch_height)
    
    # 4. Convert images to arrays and normalize
    arr_target = np.array(target_patch, dtype=np.float32) / 255.0
    arr_generated = np.array(generated_patch, dtype=np.float32) / 255.0
    
    # COMPONENT 1: LAB COLOR SPACE COMPARISON
    lab_score = 0
    if arr_target.shape[-1] >= 3:
        # Extract RGB channels
        target_rgb = arr_target[:, :, :3]
        generated_rgb = arr_generated[:, :, :3]
        
        # Convert to LAB color space
        target_lab = color.rgb2lab(target_rgb)
        generated_lab = color.rgb2lab(generated_rgb)
        
        # Calculate MSE in LAB space
        diff_lab = target_lab - generated_lab
        lab_mse = np.mean(diff_lab ** 2)
        lab_score = 1.0 / (1.0 + lab_mse)
    else:
        # Fallback to regular MSE for non-RGB images
        diff = arr_target - arr_generated
        mse = np.mean(diff ** 2)
        lab_score = 1.0 / (1.0 + mse)
        
    # COMPONENT 2: STRUCTURAL SIMILARITY
    ssim_score = 0
    if arr_target.shape[-1] >= 3:
        # Calculate SSIM for each RGB channel and average
        for i in range(3):
            ssim_score += structural_similarity(
                arr_target[:,:,i], 
                arr_generated[:,:,i], 
                data_range=1.0  # We normalized to [0,1]
            )
        ssim_score /= 3
    else:
        ssim_score = structural_similarity(
            arr_target.squeeze(), 
            arr_generated.squeeze(),
            data_range=1.0
        )
    
    # COMPONENT 3: EDGE DETECTION COMPARISON
    edge_score = 0
    if arr_target.shape[-1] >= 3:
        # Convert to grayscale for edge detection
        target_gray = np.mean(arr_target[:,:,:3], axis=2)
        generated_gray = np.mean(arr_generated[:,:,:3], axis=2)
        
        # Apply Sobel filter to detect edges
        target_edges = ndimage.sobel(target_gray)
        generated_edges = ndimage.sobel(generated_gray)
        
        # Compare edge maps
        edge_diff = target_edges - generated_edges
        edge_mse = np.mean(edge_diff ** 2)
        edge_score = 1.0 / (1.0 + edge_mse)
    
    # COMPONENT 4: COMPLEXITY PENALTY
    # Encourage using fewer triangles
    triangle_count = len(individual.genes)
    complexity_penalty = 0.002 * triangle_count
    complexity_factor = max(0.5, 1.0 - complexity_penalty)  # Limit penalty
    
    # COMBINE ALL COMPONENTS
    # Weighted combination of all metrics
    visual_score = (0.4 * lab_score) + (0.4 * ssim_score) + (0.2 * edge_score)
    
    # Apply complexity penalty
    final_fitness = visual_score * complexity_factor
    
    return max(0.001, final_fitness)  # Ensure positive fitness