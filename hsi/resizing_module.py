# resizing_module.py

import cv2
import numpy as np

def resize_image(image, target_shape):
    """
    Resizes an image to a specified target shape.

    Args:
    - image (np.ndarray): The input image to be resized.
    - target_shape (tuple of int): A tuple defining the target (height, width).

    Returns:
        - resized_image (np.ndarray): The image after being resized to the target shape.
    """
    return cv2.resize(image, target_shape, interpolation=cv2.INTER_LINEAR)