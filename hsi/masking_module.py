# masking_module.py

import cv2
import numpy as np

def inpaint_masked_pixels(data_band, mask_band, inpaint_radius=3):
    """
    Inpaint masked pixels of a single band image using OpenCV's inpaint method.
    
    Args:
    - data_band (np.ndarray): 2D image data of a single spectral band.
    - mask_band (np.ndarray): 2D boolean mask where True indicates pixels to inpaint.
    - inpaint_radius (int): Radius of a circular neighborhood of each point inpainted.
    
    Returns:
        -inpainted_band (np.ndarray): Inpainted image data for the single spectral band.
    """
    inpaint_mask = (mask_band.astype(np.uint8)) * 255  # 255 for masked pixels
    inpainted_band = cv2.inpaint(data_band.astype(np.float32), inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return inpainted_band