import matplotlib.pyplot as plt
import numpy as np
from spectral import *
from spectral import imshow


def show_rgb(img, dsetName, bands=(1, 6, 29), figsize=(100, 100)):
    """
    Displays an RGB preview of hyperspectral image data.

    Args:
        img (np.ndarray): Hyperspectral image data.
        dsetName (str): Name of the dataset being previewed.
        bands (tuple, optional): Indices of the three spectral bands to use for RGB visualization.
        figsize (tuple, optional): Figure size for display.

    Returns:
        int: Always returns 0 (no functional return value).
    """
    v = imshow(img, bands, stretch_all=True, title=dsetName, figsize=figsize)
    return 0


def show_pca_rgb(img, dsetName, bands=(0, 1, 2), figsize=(24, 36)):
    """
    Displays an RGB preview of PCA-transformed hyperspectral image data.

    Args:
        img (np.ndarray): PCA-transformed hyperspectral image data.
        dsetName (str): Name of the dataset being previewed.
        bands (tuple, optional): Indices of the three principal components to use for RGB visualization.
        figsize (tuple, optional): Figure size for display.

    Returns:
        int: Always returns 0 (no functional return value).
    """
    v = imshow(img, bands, stretch_all=True, title=dsetName, figsize=figsize)
    return 0


def interactive_preview(img, dsetName, rgb_bands=(16, 59, 96), figsize=(3, 5)):
    """
    Displays an interactive RGB preview of hyperspectral image data using matplotlib.
    Function requires "%matplotlib ipympl" in a Jupyter notebook.

    Args:
        img (np.ndarray): Hyperspectral image data.
        dsetName (str): Name of the dataset being previewed.
        rgb_bands (tuple, optional): Indices of the three spectral bands to use for RGB visualization.
        figsize (tuple, optional): Figure size for display.

    Returns:
        int: Always returns 0 (no functional return value).
    """
    # Extract the specified RGB bands
    rgbImage = img[:, :, list(rgb_bands)]

    # Normalize the image to [0, 1] range
    rgbImage = (rgbImage - np.min(rgbImage)) / (np.max(rgbImage) - np.min(rgbImage))

    fig = plt.figure(figsize=figsize)
    plt.title(dsetName)
    plt.imshow(rgbImage)
    plt.show(block=False)

    return 0
