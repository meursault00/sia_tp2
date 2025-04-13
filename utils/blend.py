import numpy as np

def create_blend_mask(patch_w, patch_h, left_margin, right_margin, top_margin, bottom_margin):
    """
    Creates a blending mask for a patch of dimensions (patch_w, patch_h). The mask is calculated so that:
      - In the horizontal dimension, the weight ramps linearly from 0 at the left edge up to 1 at left_margin,
        remains 1 in the central region, and then declines linearly back to 0 at the right edge.
      - The same is done in the vertical dimension.
    The final mask is the outer product of the two 1D weight arrays and is repeated over 4 channels (for RGBA).
    
    Parameters:
      patch_w (int): Width of the patch.
      patch_h (int): Height of the patch.
      left_margin (int): Number of pixels along the left border where weights ramp up.
      right_margin (int): Number of pixels along the right border where weights ramp down.
      top_margin (int): Number of pixels along the top border where weights ramp up.
      bottom_margin (int): Number of pixels along the bottom border where weights ramp down.
      
    Returns:
      mask (np.ndarray): A (patch_h, patch_w, 4) array with weights in the range [0, 1].
    """
    # Create horizontal weight array
    weights_x = np.ones(patch_w, dtype=np.float32)
    for i in range(patch_w):
        if i < left_margin:
            weights_x[i] = i / left_margin
        elif i >= patch_w - right_margin:
            weights_x[i] = (patch_w - i - 1) / right_margin
    weights_x = np.clip(weights_x, 0, 1)
    
    # Create vertical weight array
    weights_y = np.ones(patch_h, dtype=np.float32)
    for j in range(patch_h):
        if j < top_margin:
            weights_y[j] = j / top_margin
        elif j >= patch_h - bottom_margin:
            weights_y[j] = (patch_h - j - 1) / bottom_margin
    weights_y = np.clip(weights_y, 0, 1)
    
    # Outer product to generate 2D mask
    mask = np.outer(weights_y, weights_x)
    # Expand mask to 4 channels (RGBA)
    mask = np.repeat(mask[:, :, np.newaxis], 4, axis=2)
    
    return mask