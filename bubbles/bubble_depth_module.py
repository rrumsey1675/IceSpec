# Not in use, idea/concept for future work

import numpy as np

def estimate_bubble_depth(reflected_img, transmitted_low, transmitted_high, masks):
    """
    Estimate the depth of bubbles based on reflected and transmitted light scans.

    Args:
    - reflected_img (np.ndarray): Reflected light image (2D array).
    - transmitted_low (np.ndarray): Low-exposure transmitted light image (2D array).
    - transmitted_high (np.ndarray): High-exposure transmitted light image (2D array).
    - masks (list of np.ndarray): List of bubble masks (one per band).

    Returns:
    - depth_map (np.ndarray): 2D array where each pixel represents estimated bubble depth.
    """
    depth_map = np.zeros_like(reflected_img, dtype=np.float32)
    
    for mask in masks:
        # Extract intensities for the current mask
        reflected_intensity = np.mean(reflected_img[mask])
        transmitted_low_intensity = np.mean(transmitted_low[mask])
        transmitted_high_intensity = np.mean(transmitted_high[mask])
        
        # Calculate depth score
        depth = calculate_depth(reflected_intensity, transmitted_low_intensity, transmitted_high_intensity)
        
        # Assign depth to mask region
        depth_map[mask] = depth
    
    return depth_map

def calculate_depth(reflected_intensity, transmitted_low, transmitted_high):

    depth_score = (transmitted_high / (reflected_intensity + 1e-5)) * \
                  (transmitted_high / (transmitted_low + 1e-5))
    return depth_score
