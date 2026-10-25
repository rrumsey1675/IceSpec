## rumz May 2025

import numpy as np
import pandas as pd
import statistics


def get_sums(img: np.ndarray, px_per_cm: float, top_depth, Debug: bool = False) -> pd.DataFrame:
    """
    Calculates the sum of pixel values across each row for each spectral channel
    and maps the row index to depth in centimeters using a pixel-to-depth conversion factor.

    Args:
        img (np.ndarray): A 3D image array with shape (height, width, channels),
                          typically reflectance-corrected and bubble-filled.
        px_per_cm (float): Number of vertical pixels per centimeter, used to convert rows to depth.
        Debug (bool, optional): If True, prints debug information per row and channel. Defaults to False.

    Returns:
        pd.DataFrame: A table with depth in centimeters as the first column,
                      followed by summed values per channel for each row.
    """

    if img.ndim != 3:
        raise ValueError("Expected image with shape (height, width, channels)")

    height, _, channels = img.shape
    depths_cm = top_depth * 100 + (np.arange(height) / px_per_cm)

    row_sums = np.sum(img, axis=(1, 2))
    row_mean = np.mean(img, axis=(1, 2))
    row_std = np.std(img, axis=(1,2))
    row_median = np.median(img, axis=(1, 2))
    row_med_5  = np.percentile(img, 5, axis=(1, 2)) # median 5 per row
    row_med_95 = np.percentile(img, 95, axis=(1, 2)) # median 95 per row
    

    data = {"Depth_cm": depths_cm, "Sums": row_sums, "Means": row_mean, "STDev": row_std, "Medians": row_median, "Median_5": row_med_5, "Median_95": row_med_95}

    df = pd.DataFrame(data)
    
    # df = df[(df.select_dtypes(include='number') >= 0).all(axis=1)]

    return df
        
        