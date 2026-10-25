from icespec import sigma_module as calcsigma
import numpy as np
import scipy.fftpack
import copy
import cv2
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import sigma_clipped_stats
from skimage import segmentation, feature, measure, color
from skimage.morphology import disk, binary_opening, binary_closing, remove_small_objects, remove_small_holes, rectangle

from skimage.segmentation import relabel_sequential
from skimage.measure import label
from scipy import ndimage as ndi
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources
from skimage.feature import peak_local_max
from sklearn.decomposition import PCA
from typing import List, Tuple


def extract(img: np.ndarray, fwhm: List[float], debug: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Extract bubble regions from a hyperspectral image by applying per-band segmentation.

    Args:
        img (np.ndarray): 3D hyperspectral image array with shape (height, width, bands).
        fwhm (List[float]): List of full-width at half-maximum (FWHM) values for each band.
        debug (bool, optional): If True, enables debug visualizations. Default is False.

    Returns:
        Tuple:
            - bubble_data (np.ndarray): Hyperspectral image retaining only bubble pixels (others set to zero).
            - labeled_masks (List[np.ndarray]): List of labeled masks per band (integer IDs for bubble regions).
    """
    labeled_masks, provenance = masking(img, fwhm, debug=debug)

    bubble_data = np.zeros_like(img)
    for i, mask in enumerate(labeled_masks):
        bubble_data[:, :, i] = img[:, :, i] * mask.astype(img.dtype)

    return bubble_data, labeled_masks


def masking(img: np.ndarray, fwhm: List[float], debug: bool = False) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Generate labeled bubble masks for each spectral band.

    Args:
        img (np.ndarray): Hyperspectral image array (height, width, bands).
        fwhm (List[float]): List of FWHM values per spectral band.
        debug (bool, optional): If True, enables debug visualizations. Default is False.

    Returns:
        Tuple:
            - labeled_masks (List[np.ndarray]): List of per-band labeled masks (unique bubble IDs).
            - provenance (np.ndarray): 2D array storing first-detection band index for each pixel.
    """
    labeled_masks: List[np.ndarray] = []
    provenance = np.zeros(img.shape[:2], dtype=int)
    next_label = 1

    for band_index in range(img.shape[2]):
        band_data = img[:, :, band_index]
        segm_mask = deblending_image(band_data, band_index, fwhm[band_index])

        labeled = label(segm_mask)
        labeled[labeled > 0] += next_label - 1
        labeled_masks.append(labeled)

        provenance[(labeled > 0) & (provenance == 0)] = band_index + 1
        next_label = labeled.max() + 1

    return labeled_masks, provenance


def deblending_image(band_data: np.ndarray, band_index: int, fwhm: float) -> np.ndarray:
    """
    Create a bubble segmentation mask for a single spectral band using adaptive thresholding.

    Args:
        band_data (np.ndarray): 2D array representing the spectral band.
        band_index (int): Index of the spectral band.
        fwhm (float): Full-width at half-maximum (FWHM) for the current band.

    Returns:
        np.ndarray: Binary mask indicating detected bubble regions.
    """
    band_norm = cv2.normalize(band_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.GaussianBlur(band_norm, (5, 5), 0)

    adaptive_thresh = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=55,
        C=-2
    )

    segm_mask = adaptive_thresh > 0
    #segm_mask = remove_small_objects(segm_mask, min_size=10)
    return segm_mask


def get_combined_bubble_mask(img: np.ndarray, fwhm: List[float], clean: bool, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine per-band bubble masks into a single labeled mask.

    Args:
        img (np.ndarray): Hyperspectral image array (height, width, bands).
        fwhm (List[float]): FWHM values per band.
        clean (bool): If True, apply small-object removal before watershed segmentation.
        debug (bool, optional): If True, enables debug visualizations. Default is False.

    Returns:
        Tuple:
            - combined_mask (np.ndarray): Labeled mask of bubbles resolved across all bands.
            - provenance (np.ndarray): Array mapping pixels to their first-detection band.
    """
    labeled_masks, provenance = masking(img, fwhm, debug=debug)

    combined_labels = np.zeros(img.shape[:2], dtype=int)
    for labeled in labeled_masks:
        combined_labels = np.maximum(combined_labels, labeled)

    if clean:
        combined_labels = remove_small_objects(combined_labels, min_size=10)

    ws_labels = fixOverlap(combined_labels > 0, debug=debug)
    ws_labels, _, _ = relabel_sequential(ws_labels)
    return ws_labels, provenance


def clean_stripes_in_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remove vertical striping artifacts from a binary mask.

    Args:
        mask (np.ndarray): Binary segmentation mask with vertical striping.

    Returns:
        np.ndarray: Cleaned binary mask.
    """
    structuring_element = rectangle(1, 10)
    mask_cleaned = opening(mask, structuring_element)
    mask_cleaned = remove_small_objects(mask_cleaned, min_size=50)
    return mask_cleaned


def fixOverlap(mask: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Resolve overlapping bubble regions using watershed segmentation.

    Args:
        mask (np.ndarray): Binary mask of bubble regions.
        debug (bool, optional): If True, displays watershed segmentation results.

    Returns:
        np.ndarray: Labeled mask where overlaps are separated into distinct bubbles.
    """
    # Structuring element
    selem = disk(2)

    # --- Morphological opening then closing ---
    m = mask.copy()
    for _ in range(3):
        m = binary_opening(m, selem)
    # for _ in range(2):
    #     m = binary_closing(m, selem)

    m = remove_small_holes(m, area_threshold=max(0, 9))
    
    distance = ndi.distance_transform_edt(mask)
    ## Switch distance for radiance: of Band 75 // Median for all channels // PCA
    coords = feature.peak_local_max(distance, labels=mask, footprint=np.ones((3, 3)), exclude_border=False)

    markers = np.zeros_like(mask, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    labels_ws = segmentation.watershed(-distance, markers, mask=mask)
    cleaned_ws = remove_small_objects(labels_ws, min_size=10)

    if debug:
        plt.figure(figsize=(15, 10))
        plt.imshow(color.label2rgb(cleaned_ws, image=mask, bg_label=0))
        plt.title("Watershed Bubble Separation")
        plt.axis("off")
        plt.show()

    return cleaned_ws


    # labeled = measure.label(mask)
    # markers = np.zeros_like(mask, dtype=int)
    # current_marker = 1

    # for region in measure.regionprops(labeled):
    #     coords = np.array(region.coords)

    #     if coords.shape[0] > 2:
    #         pca = PCA(n_components=2)
    #         pca.fit(coords)
    #         transformed = pca.transform(coords)

    #         median_split = np.median(transformed[:, 0])
    #         left_cluster = coords[transformed[:, 0] <= median_split]
    #         right_cluster = coords[transformed[:, 0] > median_split]

    #         for cluster in [left_cluster, right_cluster]:
    #             if len(cluster) > 0:
    #                 r, c = np.mean(cluster, axis=0).astype(int)
    #                 markers[r, c] = current_marker
    #                 current_marker += 1
    #     else:
    #         r, c = np.mean(coords, axis=0).astype(int)
    #         markers[r, c] = current_marker
    #         current_marker += 1

    # labels_ws = segmentation.watershed(-mask.astype(float), markers, mask=mask)
    # cleaned_ws = remove_small_objects(labels_ws, min_size=10)

    # if debug:
    #     plt.figure(figsize=(15, 10))
    #     plt.imshow(color.label2rgb(cleaned_ws, image=mask, bg_label=0))
    #     plt.title("Watershed with PCA-Based Markers")
    #     plt.axis("off")
    #     plt.show()

    # return cleaned_ws

    ########------------------------
     # # noise removal
    # kernel = np.ones((3,3), np.uint8)
    # mask_uint8 = mask.astype(np.uint8)
    # opening = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations = 2)
    # # sure background area
    # sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
    # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg, sure_fg)

    # # plt.imshow("SureFG", sure_fg)
    # # plt.show()
    # # plt.imshow("SureBG", sure_bg)
    # # plt.show()
    # # plt.imshow("unknown", unknown)
    # # plt.show()
    
    # # Marker labelling
    # # Connected Components determines the connectivity of blob-like regions in a binary image.
    # ret, markers = cv2.connectedComponents(sure_fg)
    
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers+1
    
    # # Now, mark the region of unknown with zero
    # markers[unknown==255] = 0

    # labels_ws = segmentation.watershed(-mask.astype(float), markers, mask=mask)
    # # mask[labels_ws == -1] = [255,0,0]
    
    # #cleaned_ws = remove_small_objects(labels_ws, min_size=10)

    # if debug:
    #     plt.figure(figsize=(15, 10))
    #     plt.imshow(color.label2rgb(labels_ws, image=mask, bg_label=0))
    #     plt.title("Watershed Bubble Separation")
    #     plt.axis("off")
    #     plt.show()

    # return labels_ws
