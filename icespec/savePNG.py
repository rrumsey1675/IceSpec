import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os


def save_as_PNG(img, dsetName, savePath):
    """
    Saves a hyperspectral image as a PNG file using selected RGB bands.
    The image is normalized and saved with the current date appended to the filename.
    If the target directory does not exist, it is created automatically.

    Args:
        img (np.ndarray): Hyperspectral image data.
        dsetName (str): Name of the dataset (used for labeling the image).
        savePath (str): Path to save the PNG file (directory and filename prefix).

    Returns:
        None
    """
    # Select specific bands (16, 59, 96) to generate an RGB image.
    rgbImage = img[:, :, [16, 59, 96]]

    # Normalize the image to the range [0, 1] for proper visualization.
    rgbImage = (rgbImage - np.min(rgbImage)) / (np.max(rgbImage) - np.min(rgbImage))

    # Ensure that the target directory exists; create it if necessary.
    subPath = os.path.dirname(savePath)
    os.makedirs(subPath, exist_ok=True)

    # Append the current date to the filename in "M-D-Y" format.
    current_date = datetime.now().strftime("%-m-%-d-%y")
    savePath = f"{savePath}_{current_date}.png"
    print("SavePath =", savePath)

    # Create a figure for saving the image.
    fig = plt.figure(figsize=(2, 5))
    plt.title(dsetName)
    plt.imshow(rgbImage)
    plt.axis('off')  # Remove axes for a cleaner image.

    # Save the figure as a PNG file without extra margins or padding.
    plt.savefig(savePath, format='png', bbox_inches='tight', pad_inches=0)