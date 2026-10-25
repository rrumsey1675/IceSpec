from icespec import sigma_module as calcsigma
import numpy as np
from typing import List, Tuple
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources
from skimage.morphology import rectangle, binary_opening, remove_small_objects
from scipy import ndimage as ndi
from skimage import morphology, segmentation, feature, color
import matplotlib.pyplot as plt


def extract(img: np.ndarray, fwhm: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Extracts bubbles from a hyperspectral image by applying masks to each spectral band.

    Args:
        img (np.ndarray): 3D hyperspectral image array with shape (height, width, bands).
        fwhm (np.ndarray): 3D array of Full Width at Half Maximum values for the image.

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: A tuple containing:
            - bubble_data: Hyperspectral image array containing only the bubbles.
            - labeled_masks: List of labeled masks for each spectral band.
    """
    # Generate masks and provenance for each spectral band
    labeled_masks, provenance = masking(img, fwhm)

    # Initialize output array for bubble data
    bubble_data = np.full_like(img, np.nan)

    # Apply masks to retain bubble pixels, setting non-bubble pixels to zero
    for i, mask in enumerate(labeled_masks):
        bubble_data[:, :, i] = img[:, :, i] * (mask > 0).astype(img.dtype)

    return bubble_data, labeled_masks

def masking(img: np.ndarray, fwhm: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Creates segmentation masks for each spectral band and tracks pixel provenance.

    Args:
        img (np.ndarray): 3D hyperspectral image array with shape (height, width, bands).
        fwhm (np.ndarray): array of FWHM values for the image.

    Returns:
        Tuple[List[np.ndarray], np.ndarray]: A tuple containing:
            - labeled_masks: List of labeled segmentation masks for each band.
            - provenance: 2D array indicating the first band index detecting each pixel.
    """
    labeled_masks: List[np.ndarray] = []
    provenance = np.zeros(img.shape[:2], dtype=int)  # Tracks band index per pixel
    next_label = 1  # Counter for unique label IDs across bands

    sigma = calcsigma.calculate_sigma(fwhm) 

    # Process each spectral band
    for band_index in range(img.shape[2]):
        band_data = img[:, :, band_index]
        segm_mask = deblending_image(band_data, band_index, sigma[band_index])

        # [Note]: Optional stripe cleaning (unimplemented, kept for future use)
        # segm_mask = clean_stripes_in_mask(segm_mask)

        # Label bubbles in the current band
        labeled, num_features = ndi.label(segm_mask)
        print(f"\nFound {num_features} initial markers in {band_index} band")
        labeled[labeled > 0] += next_label - 1  # Offset labels to ensure uniqueness
        labeled_masks.append(labeled)

        # Update provenance with the first band detecting each pixel
        provenance[(labeled > 0) & (provenance == 0)] = band_index + 1

        # Update label counter for the next band
        next_label = labeled.max() + 1

    return labeled_masks, provenance

def deblending_image(band_data: np.ndarray, band_index: int, sigma: float) -> np.ndarray:
    """
    Deblends a single spectral band image to separate overlapping sources (bubbles).

    Args:
        band_data (np.ndarray): 2D array representing the band image data.
        band_index (int): Index of the spectral band (0-based).
        sigma (float): Sigma from fwhm

    Returns:
        np.ndarray: Boolean segmentation mask of deblended sources.
    """
    # Calculate sigma from FWHM using custom function
    # sigma = calcsigma.calculate_sigma(fwhm)

    # Estimate and subtract background
    bkg_estimator = MedianBackground()
    bkg = Background2D(band_data, box_size=(3, 3), exclude_percentile=2, filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data_subtracted = band_data - bkg.background

    # Convolve data with Gaussian kernel
    kernel = Gaussian2DKernel(sigma, x_size=1, y_size=1)
    kernel.normalize()
    convolved_data = convolve(data_subtracted, kernel)

    # Detect sources using statistical threshold
    mean, median, std = sigma_clipped_stats(data_subtracted)
    threshold = median + (1 * std)
    segments = detect_sources(convolved_data, threshold, npixels=2)

    # Create boolean segmentation mask
    segm_mask = segments.data.astype(bool) if segments else np.zeros_like(band_data, dtype=bool)

    return segm_mask

def get_combined_bubble_mask(img: np.ndarray, fwhm: np.ndarray, clean: bool = False, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a combined mask representing bubbles detected across all spectral bands.

    Args:
        img (np.ndarray): 3D hyperspectral image array with shape (height, width, bands).
        fwhm (np.ndarray): 3D array of FWHM values for the image.
        clean (bool): If True, removes small objects from the combined mask. Defaults to False.
        debug (bool): If True, visualizes the watershed result in fix_overlap. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - combined_labels: 2D array of combined labeled masks.
            - provenance: 2D array indicating the first band index detecting each pixel.
    """
    # Generate labeled masks and provenance
    labeled_masks, provenance = masking(img, fwhm)

    # Combine masks by taking the maximum label at each pixel
    combined_labels = np.zeros(img.shape[:2], dtype=int)
    for labeled in labeled_masks:
        combined_labels = np.maximum(combined_labels, labeled)

    # Optionally clean small objects
    if clean:
        combined_labels = morphology.remove_small_objects(combined_labels, min_size=5)

    # [Note]: Optional overlap resolution (unimplemented, kept for future use)
    # ws_labels = fix_overlap(combined_labels > 0, debug=debug)
    # ws_labels, _, _ = relabel_sequential(ws_labels)

    return combined_labels, provenance

def clean_stripes_in_mask(mask: np.ndarray) -> np.ndarray:
    """
    Removes vertical stripes from a binary mask using morphological operations.

    Args:
        mask (np.ndarray): Binary mask with potential vertical striping.

    Returns:
        np.ndarray: Cleaned binary mask with stripes and small objects removed.
    """
    # Define vertical structuring element for morphological opening
    structuring_element = rectangle(1, 10)
    mask_cleaned = binary_opening(mask, structuring_element)

    # Remove small objects from the cleaned mask
    mask_cleaned = remove_small_objects(mask_cleaned, min_size=50)

    return mask_cleaned

def fix_overlap(mask: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Resolves overlapping regions in a binary mask using watershed segmentation.

    Args:
        mask (np.ndarray): Binary mask indicating bubble regions.
        debug (bool): If True, displays a visualization of the watershed result. Defaults to False.

    Returns:
        np.ndarray: Labeled mask with resolved overlaps.
    """
    # Compute distance transform for watershed
    distance = ndi.distance_transform_edt(mask)

    # Identify local maxima as markers for watershed
    coords = feature.peak_local_max(
        distance,
        labels=mask,
        footprint=np.ones((3, 3)),
        exclude_border=False
    )

    # Create markers for each peak
    markers = np.zeros_like(mask, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    # Perform watershed segmentation
    labels_ws = segmentation.watershed(-distance, markers, mask=mask)

    # Remove small objects from the result
    cleaned_ws = morphology.remove_small_objects(labels_ws, min_size=10)

    # Optional visualization for debugging
    if debug:
        plt.figure(figsize=(15, 10))
        plt.imshow(color.label2rgb(cleaned_ws, image=mask, bg_label=0))
        plt.title("Watershed Bubble Separation")
        plt.axis("off")
        plt.show()

    return cleaned_ws