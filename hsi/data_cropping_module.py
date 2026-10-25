# data_cropping_module.py

import numpy as np
import matplotlib.pyplot as plt

def data_cropping(removal_percent, full_data, display=True):
    """
    Takes a removal percentage and the full hyperspectral data to crop all margins og the data.
    
    Args:
    - removal_percent (float): Percentage of the image width to remove from all sides.
    - full_data (3D np.ndarray Float 32): 3D array containing all the hyperspectral data to be cropped (Lines, Samples and Bands)
    
    Returns:
        tuple:
           - representative_band_cropped (2D np.ndarray Float 32): 2D array representing the cropped data from the first band.
           - data_cropped (3D np.ndarray Float 32): 3D array representing the full cropped hyperspectral data (LInes, Samples and Bands)
    """
    
    representative_band = full_data[:,:,0] #Creates an array with the data from Lines and Samples based on the first Band.
    height, width, bands = full_data.shape  #Gets the shape of the data, saving it to the hight widthe and band variables.
    left_margin = int(removal_percent * width) #Creates an integer for the new left margin location.
    right_margin = width - left_margin  #USes widthe and new left margin to get the right margin location.
    top_margin = int(removal_percent * height) #Creates an integer for the new top margin location
    bottom_margin = height - top_margin #Uses the top margin and height to find the new bottom location.
    
    #Crops the representative band and full data by the calculated margins
    representative_band_cropped = representative_band[top_margin:bottom_margin, left_margin:right_margin]
    data_cropped = full_data[top_margin:bottom_margin, left_margin:right_margin, :]
    
    # Display the cropped representative band using matplotlib
    if display:
        plt.imshow(representative_band_cropped, cmap='gray')
        plt.show()

    return representative_band_cropped, data_cropped # Return the cropped band and the full cropped data

