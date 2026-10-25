from icespec import sigma_module as calcsigma
import numpy as np
import scipy.fftpack
import copy
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import sigma_clipped_stats
from skimage.morphology import binary_opening, binary_closing
from skimage.morphology import opening, remove_small_objects
from scipy import ndimage as ndi
from skimage import morphology, segmentation, feature, measure, color
from skimage.morphology import rectangle
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources


def extract(img: np.array, fwhm: list) -> np.array:
    """
    Extracts bubbles from a hyperspectral image by applying masks to each spectral band.

    Args:
    - img (np.ndarray): 3D hyperspectral image array with shape (height, width, bands).
    - fwhm (list): List of FWHM values corresponding to each spectral band.

    Returns:
    - bubble_data (np.ndarray): Hyperspectral image array containing only the bubbles.
    """
    bubble_img = copy.deepcopy(img)
    
    
    sigma = calcsigma.calculate_sigma(fwhm)

    # Background estimation
    bkg_estimator = MedianBackground()
    bkg = Background2D(bubble_img, (15, 15), exclude_percentile=90, filter_size=(5, 5), bkg_estimator=bkg_estimator)
    data_subtracted = bubble_img - bkg.background

    # Convolve subtracted data with Gaussian kernel
    kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
    kernel.normalize()
    convolved_data = convolve(data_subtracted, kernel)

    # Detect sources
    mean, median, std = sigma_clipped_stats(data_subtracted)
    threshold = median + (1 * std)
    segments = detect_sources(convolved_data, threshold, npixels=5)

    # Create segmentation mask, handling possible no-detection case
    bubble_mask = segments.data.astype(bool) if segments else np.zeros(band_data.shape, dtype=bool)

    watershed_mask = fixOverlap(bubble_mask)
    cleaned_mask = clean_stripes_in_mask(watershed_mask)

    return cleaned_mask

def clean_stripes_in_mask(mask):
    """
    Removes vertical stripes from binary mask.

    Args:
    - mask (np.ndarray): Binary mask with vertical striping.

    Returns:
    - cleaned_mask (np.ndarray): Cleaned binary mask.
    """
    structuring_element = rectangle(1, 10)  # Vertical shape
    mask_cleaned = opening(mask, structuring_element)
    mask_cleaned = remove_small_objects(mask_cleaned, min_size=50)
    return mask_cleaned

def fixOverlap(img):

    distance = ndi.distance_transform_edt(img)
    
    coords = feature.peak_local_max(
        distance, 
        labels=img, 
        footprint=np.ones((3, 3)), 
        exclude_border=False
    )
    
    markers = np.zeros_like(img, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    
    labels_ws = segmentation.watershed(-distance, markers, mask=img)
    

    cleaned_ws = morphology.remove_small_objects(labels_ws, min_size=10)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    plt.imshow(color.label2rgb(cleaned_ws, image=img, bg_label=0))
    plt.title("Watershed Bubble Separation")
    plt.axis("off")
    plt.show()

    return cleaned_ws