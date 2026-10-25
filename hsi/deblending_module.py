# deblending_module.py

from hsi import sigma_module as calcsigma
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources

def deblending_image(band_data, band_index, fwhm_values):
    """
    Deblend a single spectral band image to separate overlapping sources.
    
    Args:
    - band_data: 2D numpy array representing the band image data.
    - band_index: Index of the spectral band (0-based).
    - fwhm_array: Array of FWHM values corresponding to each spectral band.
    
    Returns:
        - segm_mask: Segmentation mask of deblended sources.
    """
    # Get FWHM value for the current band from the array
    fwhm = fwhm_values
    sigma = calcsigma.calculate_sigma(fwhm)
    
    # Background estimation
    bkg_estimator = MedianBackground()
    bkg = Background2D(band_data, (25, 25), exclude_percentile=75, filter_size=(1, 1), bkg_estimator=bkg_estimator)
    data_subtracted = band_data - bkg.background

    # Convolve subtracted data with Gaussian kernel
    kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
    kernel.normalize()
    convolved_data = convolve(data_subtracted, kernel)

    # Detect sources
    mean, median, std = sigma_clipped_stats(data_subtracted)
    threshold = median + (2.0 * std)
    segments = detect_sources(convolved_data, threshold, npixels=5)
    
    # Create segmentation mask, handling possible no-detection case
    segm_mask = segments.data.astype(bool) if segments else np.zeros(band_data.shape, dtype=bool)
    
    return segm_mask