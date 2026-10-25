# sigma_module.py

import numpy as np

def calculate_sigma(fwhm):
    """
    Calculate the standard deviation (siogma) of a Gaussian kernel given its full width at half maximum (fwhm).
    
    Args:
    - fwhm (float): The full with at half maximum value
    
    Returns:
        - float: The sigma of the Gaussian distribution. 
    """
    
    return fwhm / (2.0 * np.sqrt(2 * np.log(2))) #Calculate sigma from fwhm using the relationship between the two for a Gaussian distribution