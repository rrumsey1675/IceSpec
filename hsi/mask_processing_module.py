# mask_processing_module.py

import numpy as np
from hsi import masking_module as masker
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def process_data_with_mask(data_cropped, individual_segm_masks, inpaint_radius=3):
    """
    This function performs the following operations on hyperspectral data:
        - Inpaints masked areas in each spectral band.
        - Reshapes the inpainted data to 2D for Principal Component Analysis (PCA).
        - Removes rows with NaN (missing) values.
        - Conducts PCA on the valid data.
        - Reconstructs the PCA-transformed data into the original image shape.
        - Normalizes the PCA components for visualization.
        - Visualizes the contributions of spectral bands to PCA components.
        - Visualizes the first five PCA components as images.
    
    Args:
    - data_cropped (np.ndarray): Cropped hyperspectral data (Lines, Samples, and Bands).
    - individual_segm_masks (List[np.ndarray]): List of 2D boolean masks for inpainting, one for each spectral band.
    - inpaint_radius (int, optional): Radius of the circular neighborhood for inpainting. Default is 3.

    Returns:
    - pca_transformed (np.ndarray): 2D array of PCA-transformed data.
    """
    # Initialize an empty array to hold inpainted data with the same shape as the cropped data.
    inpainted_data = np.empty_like(data_cropped)

    # Iterate over each band and apply inpainting using the corresponding mask.
    for i, mask in enumerate(individual_segm_masks):
        inpainted_data[:, :, i] = masker.inpaint_masked_pixels(data_cropped[:, :, i], mask, inpaint_radius)

    # Reshape the inpainted data into a 2D array, flattening the spatial dimensions while preserving the bands.
    reshaped_data = inpainted_data.reshape(-1, inpainted_data.shape[2])

    # Discard rows with NaN values, resulting in only valid data for PCA.
    valid_data = reshaped_data[~np.isnan(reshaped_data).any(axis=1)]

    # Perform PCA on valid data to reduce dimensionality and capture the most significant features.
    pca = PCA(n_components=5)
    pca_transformed = pca.fit_transform(valid_data)

    # Calculate the shape of the reconstructed image with 5 PCA components.
    output_shape = (*data_cropped.shape[:2], 5)
    
    # Prepare the data for reconstruction by setting NaN values to a neutral value.
    data_pca = np.full((reshaped_data.shape[0], 5), np.nan)
    
    # Insert the PCA-transformed data into the array, replacing the valid (non-NaN) rows.
    data_pca[~np.isnan(reshaped_data).any(axis=1)] = pca_transformed
    
    # Reshape the array back into the original image shape with 5 PCA components.
    image_pca = data_pca.reshape(output_shape)

    # Normalize the 5 PCA components to the range (0, 1) for consistent visualization.
    normalized_image_pca = (image_pca - np.nanmin(image_pca)) / (np.nanmax(image_pca) - np.nanmin(image_pca))
    
    # Replace any remaining NaNs with 0 to ensure proper visualization.
    normalized_image_pca[np.isnan(normalized_image_pca)] = 0

    # Plot the spectral band contributions to the first 5 PCA components.
    for i, component in enumerate(pca.components_[:5], start=1):
        plt.figure()
        plt.plot(component)
        plt.title(f'Spectral Band Contribution to PCA Component {i}')
        plt.xlabel('Spectral band index')
        plt.ylabel('Spectral band contribution')
        plt.show()

    # Display the normalized images of the first five PCA components.
    for i in range(5):
        plt.figure()
        plt.imshow(normalized_image_pca[:, :, i], cmap='gray')
        plt.title(f'PCA Component {i+1} Image')
        plt.colorbar()
        plt.show()

    # Return the transformed PCA data as the function output.
    return pca_transformed