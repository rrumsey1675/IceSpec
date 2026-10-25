from icespec import sigma_module as calcsigma
import numpy as np
import scipy.fftpack
import copy
import cv2
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import sigma_clipped_stats
from skimage.morphology import binary_opening, binary_closing
from skimage.morphology import opening, remove_small_objects
from scipy import ndimage as ndi
from skimage import morphology, segmentation, feature, measure, color
from skimage.segmentation import relabel_sequential
from scipy.ndimage import label
from skimage.morphology import rectangle
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources 
from typing import List, Tuple, Dict, Any

def remove (img: np.array, fwhm: list) -> np.array: 

    # === Bubble Masks ===
    bubble_only_image, bubble_masks = extract(img, fwhm)
    n = img.shape[2]
    mid = n // 2
    bubble_only_2d = bubble_masks[mid]

    # === Acquiring Median Across pixels ===
    median_2d = np.nanmedian(img, axis=2)

    # === Applying Mask ===
    img = img.astype(float, copy=False)
    mask = (bubble_only_2d > 0)
    mask = ndi.binary_dilation(mask, iterations=3) # removing a little extra bubble space just in case
    img = np.where(~mask[..., None], img, np.nan)

    # === Inpainting ===
    nan_mask = np.isnan(img)
    inpainted_img = np.copy(img)
    
    for band in range(img.shape[2]):
        band_data = img[:, :, band]
        band_mask = np.uint8(nan_mask[:, :, band]) * 255
        band_data_valid = np.where(np.isnan(band_data), 0, band_data)
        band_data_valid = np.clip(band_data_valid, 0, np.nanmax(band_data_valid))
        band_data_uint8 = ((band_data_valid / np.nanmax(band_data_valid)) * 255).astype(np.uint8)
        inpainted_band = cv2.inpaint(band_data_uint8, band_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        inpainted_band = (inpainted_band / 255.0) * np.nanmax(band_data_valid)
        inpainted_img[:, :, band] = np.where(nan_mask[:, :, band], inpainted_band, band_data)

    plt.figure(figsize=(80, 80))
    plt.imshow(inpainted_img[:,:,15])
    plt.title(f"Inpainted Image")
    plt.axis('off')
    plt.show()

    return inpainted_img
    

def extract(img: np.array, fwhm: list) -> np.array:
    """
    Extracts bubbles from a hyperspectral image by applying masks to each spectral band.

    Args:
    - img (np.ndarray): 3D hyperspectral image array with shape (height, width, bands).
    - fwhm (list): List of FWHM values corresponding to each spectral band.

    Returns:
    - bubble_data (np.ndarray): Hyperspectral image array containing only the bubbles.
    """
    labeled_masks, provenance = masking(img, fwhm)

    bubble_data = np.zeros_like(img)
    for i, mask in enumerate(labeled_masks):
        # Retain bubble pixels; set non-bubble pixels to zero
        bubble_data[:, :, i] = img[:, :, i] * (mask > 0).astype(img.dtype)

    return bubble_data, labeled_masks



def masking(img, fwhm):
    labeled_masks = []
    provenance = np.zeros(img.shape[:2], dtype=int)  # Will store band index per pixel
    next_label = 1

    for band_index in range(img.shape[2]):
        band_data = img[:, :, band_index]
        segm_mask = deblending_image(band_data, band_index, fwhm[band_index])
        #segm_mask = clean_stripes_in_mask(segm_mask)

        # Label bubbles in this band
        labeled, num_features = label(segm_mask)
        labeled[labeled > 0] += next_label - 1  # Offset IDs to keep unique
        labeled_masks.append(labeled)

        # Track provenance (first band to detect that pixel)
        provenance[(labeled > 0) & (provenance == 0)] = band_index + 1

        # Update label counter
        next_label = labeled.max() + 1

    return labeled_masks, provenance

def deblending_image(band_data, band_index, fwhm):
    """
    Deblends a single spectral band image to separate overlapping sources (bubbles).

    Args:
    - band_data: 2D numpy array representing the band image data.
    - band_index: Index of the spectral band (0-based).
    - fwhm: FWHM value for the current band.

    Returns:
    - segm_mask: Segmentation mask of deblended sources.
    """
    sigma = calcsigma.calculate_sigma(fwhm)

    # Background estimation
    bkg_estimator = MedianBackground()
    bkg = Background2D(band_data, (3, 3), exclude_percentile=2, filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data_subtracted = band_data - bkg.background

    # Convolve subtracted data with Gaussian kernel
    kernel = Gaussian2DKernel(sigma, x_size=1, y_size=1)
    kernel.normalize()
    convolved_data = convolve(data_subtracted, kernel)

    # Detect sources
    mean, median, std = sigma_clipped_stats(data_subtracted)
    threshold = median + (1 * std)
    segments = detect_sources(convolved_data, threshold, npixels=2)

    # Create segmentation mask, handling possible no-detection case
    segm_mask = segments.data.astype(bool) if segments else np.zeros(band_data.shape, dtype=bool)

    return segm_mask

def get_combined_bubble_mask(img: np.array, fwhm: list, clean: bool) -> np.array:
    """
    Creates a combined mask representing bubbles detected across all spectral bands.

    Args:
    - img (np.ndarray): 3D hyperspectral image array with shape (height, width, bands).
    - fwhm (list): List of FWHM values corresponding to each spectral band.
    - clean (bool): if True, cleans vertical striping in combined mask.

    Returns:
    - combined_mask (np.ndarray): 2D boolean array where True indicates bubble regions.
    """
    labeled_masks, provenance = masking(img, fwhm)

    # Combine labeled masks
    combined_labels = np.zeros(img.shape[:2], dtype=int)
    for labeled in labeled_masks:
        combined_labels = np.maximum(combined_labels, labeled)  # Merge without overlap

    # Optional cleaning before watershed
    if clean:
        combined_labels = morphology.remove_small_objects(combined_labels, min_size=5)
    
    # Watershed for overlap resolution
    # ws_labels = fixOverlap(combined_labels > 0)  # Pass binary for distance transform
    # ws_labels, _, _ = relabel_sequential(ws_labels)
    
    return combined_labels, provenance



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

def fixOverlap(mask):

    distance = ndi.distance_transform_edt(mask)
    
    coords = feature.peak_local_max(
        distance, 
        labels=mask, 
        footprint=np.ones((3, 3)), 
        exclude_border=False
    )
    
    markers = np.zeros_like(mask, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    
    labels_ws = segmentation.watershed(-distance, markers, mask=mask)
    

    cleaned_ws = morphology.remove_small_objects(labels_ws, min_size=10)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    plt.imshow(color.label2rgb(cleaned_ws, image=mask, bg_label=0))
    plt.title("Watershed Bubble Separation")
    plt.axis("off")
    plt.show()

    return cleaned_ws