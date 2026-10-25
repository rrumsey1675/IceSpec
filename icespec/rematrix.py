import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def new_matrix(flattened, trimmedData, pc_num):
    """
    Reshapes a flattened hyperspectral image into a 3D array with principal components.

    Args:
        flattened (np.ndarray): 2D representation of the hyperspectral image.
        trimmedData (list): Original trimmed data for reference.
        pc_num (int, optional): Number of principal components to retain. Default is 5.

    Returns:
        np.ndarray: 3D array with reshaped principal components.
    """
    img_trimmed = trimmedData[0]
    flattened = flattened.astype(float).copy()

    # Create a stable mask
    mask = ~np.isnan(flattened).any(axis=1)

    
    if np.sum(mask) == 0:
        raise ValueError("No valid rows remain after removing rows with NaNs.")
    
    valid_data = flattened[mask]
    
    col_std = valid_data.std(axis=0)
    
    if np.sum(col_std > 0) == 0:
        raise ValueError("No columns with non-zero variance remain.")

    if not np.isfinite(valid_data).all():
        raise ValueError("Non-finite values in valid data detected.")

    # Remove zero-variance columns
    nonzero_variance_mask = valid_data.std(axis=0) > 0
    valid_data = valid_data[:, nonzero_variance_mask]

    # Perform PCA
    pca = PCA(n_components=pc_num)
    pca_transformed = pca.fit_transform(valid_data)

    # If mismatch still exists, directly print lengths
    if mask.sum() != pca_transformed.shape[0]:
        raise ValueError(f"Mismatch detected: mask sum = {mask.sum()}, PCA rows = {pca_transformed.shape[0]}")

    # Prepare reconstruction array
    output_shape = (*img_trimmed.shape[:2], pc_num)
    data_pca = np.full((flattened.shape[0], pc_num), np.nan)

    # Use np.where(mask) to correctly insert PCA-transformed data
    data_pca[np.where(mask)] = pca_transformed

    # Force fix any remaining NaNs (last resort)
    data_pca[np.isnan(data_pca)] = 0

    # Final reshape
    image_pca = data_pca.reshape(output_shape)

    return image_pca