import numpy as np
import matplotlib.pyplot as plt
import spectral
spectral.settings.envi_support_nonlowercase_params = True

def trim(data_nparr: np.ndarray, w_ref: np.ndarray, d_ref: np.ndarray, dataSize: list, w_stdSize: list = [530, 820, 200, 900], Debug: bool= False) -> list:
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

    if dataSize[2] > w_stdSize[2]:
        img_left = dataSize[2]
    else:
        img_left = w_stdSize[2]

    if dataSize[3] < w_stdSize[3]:
        img_right = dataSize[3]
    else:
        img_right = w_stdSize[3]
    
    
    #accessing original shape
    original_height, original_width = data_nparr.shape[:2]  # First two dimensions

    # Default dimensions
    default_dims = {
        "img_startLn": dataSize[0],
        "img_endLn": dataSize[1],
        "img_left": img_left,
        "img_right": img_right
    }
    
    try:
        # Input-based trim
        img_startLn = int(input("Top (pixels to trim from the top): "))
        img_endLn = int(input("Bottom (pixels to trim from the bottom): "))
        img_left = int(input("Left (pixels to trim from the left): "))
        img_right = int(input("Right (pixels to trim from the right): "))

        # Validate inputs
        validate_trims(img_startLn, img_endLn, img_left, img_right, original_width, original_height)

    except (ValueError, TypeError):
        print("Invalid input detected. Reverting to default dimensions.")
        img_startLn = default_dims["img_startLn"]
        img_endLn = default_dims["img_endLn"]
        img_left = default_dims["img_left"]
        img_right = default_dims["img_right"]

    # Calculate the new dimensions
    new_width = original_width - (img_left + (original_width-img_right))
    new_height = original_height - (img_startLn + (original_height-img_endLn))


    # Inform the user of the new dimensions
    print(f"The new dimensions will be: {new_width}x{new_height}")
    input("Press Enter to continue or Ctrl+C to cancel...")  # Pause for confirmation

    # trim based on user input
    img_trimmed = data_nparr[img_startLn:img_endLn,img_left:img_right, :]

    w_startLn = w_stdSize[0]# pixel to start w ref data 
    w_endLn = w_ref.shape[0]-w_stdSize[1] #calculate last pixel number

    w_trimmed = w_ref[: ,img_left:img_right, :]
    d_trimmed = d_ref[: , img_left:img_right, :]
    

    trimmedData = [img_trimmed, w_trimmed, d_trimmed]
    if Debug == True:
        print(f"Trimmed shapes: {img_trimmed.shape},W_ref: {w_trimmed.shape},D_ref:{d_trimmed.shape}")

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


    # #list based trim
    # img_startLn = dataSize[0]
    # img_endLn = dataSize[1]
    # img_left = dataSize[2]
    # img_right = dataSize[3]