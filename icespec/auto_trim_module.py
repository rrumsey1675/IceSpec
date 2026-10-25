import numpy as np
import matplotlib.pyplot as plt
import spectral
spectral.settings.envi_support_nonlowercase_params = True

def trim(data_nparr: np.ndarray, w_ref: np.ndarray, d_ref: np.ndarray, dataSize: list, w_stdSize: list = [530, 800, 200, 900], Debug: bool = False) -> list:
    """
    Crops hyperspectral images to a specified width and height.
    
    Args:
        data_nparr (np.ndarray): Hyperspectral image array.
        w_ref (np.ndarray): White reference image.
        d_ref (np.ndarray): Dark reference image.
        dataSize (list): Desired dimensions for data array.
        w_stdSize (list): Standard dimensions for white reference.
        Debug (bool): Enables additional previews if True.

    Returns:
    - List of trimmed arrays: [main_trimmed, w_trimmed, d_trimmed].
    """

    # Determine left and right trim values
    if dataSize[2] > w_stdSize[2]:
        img_left = dataSize[2]
    else:
        img_left = w_stdSize[2]

    if dataSize[3] < w_stdSize[3]:
        img_right = dataSize[3]
    else:
        img_right = w_stdSize[3]
    
    # Accessing original shape
    original_height, original_width = data_nparr.shape[:2]  # First two dimensions

    # Default dimensions based on dataSize
    default_dims = {
        "img_startLn": dataSize[0],
        "img_endLn": dataSize[1],
        "img_left": img_left,
        "img_right": img_right
    }

    # Validate the trimming dimensions
    try:
        validate_trims(default_dims["img_startLn"], default_dims["img_endLn"], 
                       default_dims["img_left"], default_dims["img_right"], 
                       original_width, original_height)
    except ValueError as e:
        # If dimensions are invalid, raise an error and terminate
        raise ValueError(f"Invalid trimming dimensions: {e}")

    # Calculate the new dimensions
    new_width = original_width - (default_dims["img_left"] + (original_width - default_dims["img_right"]))
    new_height = original_height - (default_dims["img_startLn"] + (original_height - default_dims["img_endLn"]))

    # Inform the user of the new dimensions (for Debug)
    if Debug:
        print(f"The new dimensions will be: {new_width}x{new_height}")

    # Trim based on validated dimensions
    img_trimmed = data_nparr[default_dims["img_startLn"]:default_dims["img_endLn"], 
                             default_dims["img_left"]:default_dims["img_right"], :]

    w_trimmed = w_ref[:175, default_dims["img_left"]:default_dims["img_right"], :]
    d_trimmed = d_ref[:, default_dims["img_left"]:default_dims["img_right"], :]

    # Return trimmed data
    trimmedData = [img_trimmed, w_trimmed, d_trimmed]
    
    if Debug:
        print(f"Trimmed shapes: {img_trimmed.shape}, W_ref: {w_trimmed.shape}, D_ref: {d_trimmed.shape}")

    return trimmedData

def validate_trims(img_startLn: int, img_endLn: int, img_left: int, img_right: int, original_width: int, original_height: int):
    """
    Validates user input for trimming dimensions.

    Parameters:
    - img_startLn: Pixels to trim from the top.
    - img_endLn: Pixels to trim from the bottom.
    - img_left: Pixels to trim from the left.
    - img_right: Pixels to trim from the right.
    - original_width: Original width of the image.
    - original_height: Original height of the image.

    Raises:
    - ValueError: If any of the dimensions are invalid.
    """

    if not (0 <= img_startLn < original_height and 0 <= img_endLn <= original_height):
        raise ValueError("Top/Bottom values must be within the image's height.")
    if not (0 <= img_left < original_width and 0 <= img_right <= original_width):
        raise ValueError("Left/Right values must be within the image's width.")
    if img_startLn >= img_endLn or img_left >= img_right:
        raise ValueError("Invalid trimming dimensions: start must be less than end.")