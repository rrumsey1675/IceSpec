## rumz March 2025

import numpy as np


def get_sums(img: np.ndarray, Debug=False, save=False, savePath="") -> np.ndarray:
    """
    Gets sum of each line per channel, then uses channel sums to remake image.

    Args:
     - img: (np.ndarray): Reflectance-corrected and bubble filled image.
     - Debug (bool, optional): If True, prints debug information about bubble counts.
     - save (bool, optional): If True, saves image to specified folder.
     - savePath (str, optional): used if save is True, specifes path to save folder and file name.

     Returns:
        np.ndarray: Sum Value image.
    """

    if img.ndim != 3:
        raise ValueError("Expected image with shape (height, width, channels)")


    height, _, channels = img.shape
    sums_img = np.zeros((height, channels), dtype=img.dtype)

    for channel in range(channels):
        if Debug:
            print(f"\n--- Channel {channel} ---")
        for row_index in range(height):
            row = img[row_index, :, channel]
            row_sum = np.sum(row)
            sums_img[row_index, channel] = row_sum
            if Debug:
                row_str = ' '.join(f"{val:.2f}" for val in row)
                print(f"Row {row_index} = {row_str} -> Sum = {row_sum:.2f}")

    return sums_img
        
        