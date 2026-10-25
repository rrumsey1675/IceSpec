# data_trimming_module.py
#avk Feb 24, 2024
# unit test was not performed
import numpy as np
import matplotlib.pyplot as plt

def data_trimming(full_data, removal_ypercent: float=10.0, topl: int=0, botl: int=0, band: int =30):
    """
    Takes a removal percentage and the full hyperspectral data to crop all margins og the data.
    
    Args:
    
    - full_data (3D np.ndarray Float 32): 3D array containing all the hyperspectral data to be cropped (Lines, Samples and Bands)
    - removal_percent (float): Percentage of the image width to remove from the edges sides.
    - topl: top line index in pixel
    - botl: pottom line index in pixel
    - band Image band id 1 to 224
    Returns:
        tuple:
           - representative_band_cropped (2D np.ndarray Float 32): 2D array representing the cropped data from the first band.
           - data_cropped (3D np.ndarray Float 32): 3D array representing the full cropped hyperspectral data (LInes, Samples and Bands)
    """
    
    representative_band = full_data[:,:,band] #Creates an array with the data from Lines and Samples based on the Band id.
    height, width, bands = full_data.shape  #Gets the shape of the data, saving it to the hight width and band variables.
    left_margin = int(removal_percent * width) #Creates an integer for the new left margin location.
    right_margin = width - left_margin  #Uses width and new left margin to get the right margin location.
    top_margin = topl #Creates an integer for the new top margin location
    bottom_margin = botl #Uses the top margin and height to find the new bottom location.
    
    #Crops the representative band and full data by the calculated margins
    representative_band_cropped = representative_band[top_margin:bottom_margin, left_margin:right_margin]
    data_cropped = full_data[top_margin:bottom_margin, left_margin:right_margin, :]
    
    # Display the cropped representative band using matplotlib
    plt.imshow(representative_band_cropped, cmap='gray')
    plt.show()

    return representative_band_cropped, data_cropped # Return the cropped band and the full cropped data
