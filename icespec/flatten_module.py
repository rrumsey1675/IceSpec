# flatten matrix for each band

import numpy as np

def flatten(filtered_img):
    """
    Flattens a 3D hyperspectral image into a 2D array for processing.

    Args:
        filtered_img (np.ndarray): Hyperspectral image with irrelevant bands removed.

    Returns:
        np.ndarray: 2D array representation of the image.
    """

    bands= filtered_img.shape[2]

    reshaped_data = filtered_img.reshape(-1, filtered_img.shape[2])

    #flattened_arr = np.array(flattenedChannels)
    # print("Flattened shape: ", reshaped_data.shape)
    return reshaped_data