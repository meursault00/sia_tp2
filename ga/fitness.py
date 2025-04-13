import numpy as np
from utils.render import render_individual
from skimage import color
from skimage.metrics import structural_similarity
from skimage.color import deltaE_ciede2000
from scipy import ndimage

def compute_fitness(individual, target_image):
    """
    Computes fitness on a patch by dynamically computing the bounding box from the triangles.
    It compares the rendered patch (obtained via the bounding box) with the corresponding area of the target image.
    
    A coverage factor is computed as the ratio between the area covered by the individual’s triangles 
    (i.e. the bounding box area) and the full area of the patch (target_image). This factor penalizes blank space.
    
    In this modified version, we improve the color accuracy by:
      - Computing a LAB color difference using a standard MSE in LAB space (lab_score).
      - Computing the Delta E (CIEDE2000) between the target and generated LAB images and converting this into a delta_e_score.
      - Averaging these into a combined_color_score, which is given a heavier weight in the final visual score.
    """
    # 1. Compute the dynamic bounding box from triangles.
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
    
    patch_width = max_x - min_x
    patch_height = max_y - min_y
    if patch_width <= 0 or patch_height <= 0:
        return 0.0
    
    # 2. Crop the target image to the bounding box.
    bbox = (min_x, min_y, max_x, max_y)
    target_patch = target_image.crop(bbox)
    
    # 3. Render the individual to cover the same bounding box.
    generated_patch = render_individual(individual, patch_width, patch_height)
    
    # 4. Convert images to arrays and normalize to [0,1].
    arr_target = np.array(target_patch, dtype=np.float32) / 255.0
    arr_generated = np.array(generated_patch, dtype=np.float32) / 255.0
    
    # COMPONENT 1: LAB COLOR SPACE COMPARISON (standard approach)
    lab_score = 0.0
    if arr_target.shape[-1] >= 3:
        target_rgb = arr_target[:, :, :3]
        generated_rgb = arr_generated[:, :, :3]
        target_lab = color.rgb2lab(target_rgb)
        generated_lab = color.rgb2lab(generated_rgb)
        diff_lab = target_lab - generated_lab
        lab_mse = np.mean(diff_lab ** 2)
        lab_score = 1.0 / (1.0 + lab_mse)
    else:
        diff = arr_target - arr_generated
        mse = np.mean(diff ** 2)
        lab_score = 1.0 / (1.0 + mse)
    
    # NEW: Compute Delta E (CIEDE2000) and derive a color difference score.
    delta_e_score = 0.0
    if arr_target.shape[-1] >= 3:
        # We reuse the LAB representations.
        delta_e_map = deltaE_ciede2000(target_lab, generated_lab)
        delta_e_mean = np.mean(delta_e_map)
        delta_e_score = 1.0 / (1.0 + delta_e_mean)
    else:
        delta_e_score = lab_score  # fallback
    
    # Combine LAB and Delta E scores to emphasize color fidelity.
    combined_color_score = 0.5 * lab_score + 0.5 * delta_e_score

    # COMPONENT 2: STRUCTURAL SIMILARITY (SSIM)
    ssim_score = 0.0
    if arr_target.shape[-1] >= 3:
        ssim_score = (
            structural_similarity(arr_target[:,:,0], arr_generated[:,:,0], data_range=1.0) +
            structural_similarity(arr_target[:,:,1], arr_generated[:,:,1], data_range=1.0) +
            structural_similarity(arr_target[:,:,2], arr_generated[:,:,2], data_range=1.0)
        ) / 3.0
    else:
        ssim_score = structural_similarity(arr_target.squeeze(), arr_generated.squeeze(), data_range=1.0)
    
    # COMPONENT 3: EDGE DETECTION COMPARISON
    edge_score = 0.0
    if arr_target.shape[-1] >= 3:
        target_gray = np.mean(arr_target[:,:,:3], axis=2)
        generated_gray = np.mean(arr_generated[:,:,:3], axis=2)
        target_edges = ndimage.sobel(target_gray)
        generated_edges = ndimage.sobel(generated_gray)
        edge_diff = target_edges - generated_edges
        edge_mse = np.mean(edge_diff ** 2)
        edge_score = 1.0 / (1.0 + edge_mse)
    
    # COMPONENT 4: COMPLEXITY PENALTY (fewer triangles is generally better)
    triangle_count = len(individual.genes)
    complexity_penalty = 0.002 * triangle_count
    complexity_factor = max(0.5, 1.0 - complexity_penalty)
    
    # Combine visual metrics.
    # Increase the weight on color fidelity by emphasizing combined_color_score.
    visual_score = (0.6 * combined_color_score) + (0.3 * ssim_score) + (0.1 * edge_score)
    base_fitness = visual_score * complexity_factor

    # COMPONENT 5: COVERAGE PENALTY
    full_patch_area = target_image.width * target_image.height
    bbox_area = patch_width * patch_height
    coverage_ratio = bbox_area / full_patch_area  # This ratio is <= 1
    final_fitness = base_fitness * coverage_ratio
    
    return max(0.001, final_fitness)
