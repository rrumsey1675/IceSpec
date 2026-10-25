# rumz Jan/25
# bubble processing code

import os
from typing import List, Tuple, Dict, Any
from pathlib import Path
from bubbles import get_bubbles_module as extract_bubbles
from bubbles import PCA_get_bubbles_module as PCA_extract_bubbles
from bubbles import bubbles_count_module as count
from bubbles import bubble_analysis_module as bubbleAnalysis
from bubbles import bubble_segmentation_module as getSegments
from bubbles import bubbles_per_cm
from bubbles import bubble_stats_module as stats
from bubbles import graphStats
from bubbles import bubble_watershed

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from scipy import ndimage as ndi
from skimage import segmentation, color, filters, measure, morphology
from skimage.segmentation import find_boundaries
from skimage.feature import peak_local_max

import spectral
spectral.settings.envi_support_nonlowercase_params = True
import numpy as np
import pandas as pd

# RGB band indices for visualization
rd = 96
gr = 59
bl = 16

def test_bubbles(img: np.ndarray, fwhm: List[float], clean: bool = True, debug: bool = False):
    bubble_only_image, bubble_masks = extract_bubbles.extract(img, fwhm, debug)

    return bubble_only_image, bubble_masks

def getBoundaries(segmented_img, x_coords, y_coords, filtered_img, Save: bool = False, savepath: str = ""):
    
    watershed_boundaries = find_boundaries(segmented_img, mode='inner')
    watershed_boundary_pixels = np.where(watershed_boundaries)
    
    def to01(a, lo=None, hi=None):
        if lo is None or hi is None:
            lo, hi = np.nanpercentile(a, (1, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = np.nanmin(a), np.nanmax(a)
        a = np.nan_to_num(a, nan=lo)
        out = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a, dtype=np.float32)
        return np.clip(out, 0, 1).astype(np.float32)
    
    global_lo, global_hi = np.nanpercentile(filtered_img, (1, 99))
    n = filtered_img.shape[2]
    mid = n // 2
    base = to01(filtered_img[:, :, mid], global_lo, global_hi)
    
    overlay = color.label2rgb(
        segmented_img,                           # labels
        image=base,                         
        bg_label=0,
        alpha=0.3
    )

    
    plt.close()
    plt.figure(figsize=(100, 100))
    plt.imshow(overlay)
    plt.scatter(x_coords, y_coords, c='white', s=10)
    plt.scatter(watershed_boundary_pixels[1], watershed_boundary_pixels[0], c='red', s=400, marker='.', label='Watershed Boundaries')
    plt.axis('off')
    scalebar = ScaleBar(
        0.01563,
        "cm",
        fixed_units="mm",
        fixed_value=5,
        loc="lower right",
        width_fraction=.005,
        pad=1, 
        border_pad=10,
        font_properties={"family": "serif", "size": 72}
    )
    plt.gca().add_artist(scalebar)
    plt.title("Bubble Watershed Overlay")
    if Save:
        DSName = savepath.parent.name
        filename = f"{DSName}BubbleBoundaries.png"
        plt.savefig(savepath / filename, format='png', dpi=300, bbox_inches='tight')

        readme_path = savepath / "README.md"
        with readme_path.open("a") as f:
            f.write(f"\n###  Output File: {filename}\n- {DSName}: Visualization of Bubble Boundaries\n")
    plt.show()

    return watershed_boundaries, watershed_boundary_pixels

def get_bubbles(img: np.ndarray, fwhm: List[float], clean: bool = True, debug: bool = False) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, pd.DataFrame]:
    """
    Extract bubble regions and generate masks from a hyperspectral ice core image.

    Args:
        img (np.ndarray): Hyperspectral image array of shape (height, width, bands).
        fwhm (List[float]): Full-width half-maximum (FWHM) values per spectral band.
        clean (bool, optional): If True, removes vertical striping artifacts in the combined mask. Default is True.
        debug (bool, optional): If True, displays visual output for Debugging

    Returns:
        Tuple:
            - bubble_only_image (np.ndarray): Image retaining only bubble pixels (other pixels set to zero).
            - bubble_masks (List[np.ndarray]): List of binary masks for bubble regions per spectral band.
            - bubble_only_2d (np.ndarray): Combined binary mask representing bubbles across all bands.
            - bubble_channel_df (pd.DataFrame): DataFrame mapping bubble IDs to spectral bands.
    """
    bubble_only_image, bubble_masks = extract_bubbles.extract(img, fwhm)
    bubble_only_2d, provenance = extract_bubbles.get_combined_bubble_mask(img, fwhm)

    bubble_channel_map: Dict[int, List[int]] = {}
    for bubble_id in np.unique(bubble_only_2d):
        if bubble_id == 0:
            continue
        bands = np.unique(provenance[bubble_only_2d == bubble_id])
        bubble_channel_map[bubble_id] = bands.tolist()

    # bubble_channel_df = pd.DataFrame(
    #     [{"bubble_id": k, "bands": v} for k, v in bubble_channel_map.items()]
    # )

    return bubble_only_image, bubble_masks, bubble_only_2d, bubble_channel_map


def PCA_bubbles(img: np.ndarray, fwhm: List[float]) -> np.ndarray:
    """
    Extract a bubble mask using PCA channel.

    Args:
        img (np.ndarray): Hyperspectral image array of shape (height, width, bands).
        fwhm (List[float]): Full-width half-maximum (FWHM) values per spectral band.

    Returns:
        np.ndarray: Binary mask highlighting bubbles detected in the PCA-reduced band image.
    """
    bubble_mask = PCA_extract_bubbles.extract(img, fwhm)
    return bubble_mask

def rad_watershed(img: np.ndarray, bubble_only_2d) -> np.ndarray:

    median_2d = np.nanmedian(img, axis=2)

    plt.close()
    plt.figure(figsize=(60, 60))
    plt.imshow(median_2d, cmap='gray')
    plt.title("Median 2d")
    plt.show()
    
    mask = (bubble_only_2d > 0)
    img = np.where(mask[..., None], img, np.nan)

    n = img.shape[2]
    mid = n // 2

    plt.close()
    plt.figure(figsize=(60, 60))
    plt.imshow(img[:,:,mid], cmap='gray')
    plt.title("bubble only 2d")
    plt.show()
    
    segmented_img, x_coords, y_coords = bubble_watershed.rad_watershed(img[:,:,mid], median_2d)

    return segmented_img, x_coords, y_coords

def get_segments(image: np.ndarray, top_depth: float, bottom_depth: float,
                 start_px: int = 1200, end_px: int = 7100, detect_end: bool = False) -> Tuple[List[Tuple[int, int]], float]:
    """
    Compute pixel boundaries for 1-cm depth segments in an ice core image.

    Args:
        image (np.ndarray): Hyperspectral image array of shape (height, width, bands).
        top_depth (float): Depth at the top of the core section (meters).
        bottom_depth (float): Depth at the bottom of the core section (meters).
        start_px (int, optional): Pixel where ice starts. Default is 1200.
        end_px (int, optional): Pixel where ice ends. Ignored if detect_end=True. Default is 7100.
        detect_end (bool, optional): If True, automatically detect the end of the ice region. Default is False.

    Returns:
        Tuple:
            - segment_boundaries (List[Tuple[int, int]]): List of pixel ranges per 1-cm segment.
            - px_per_cm (float): Conversion factor from pixels to centimeters.
    """
    
    return getSegments.get_ice_segments(image, top_depth, bottom_depth, start_px, end_px, detect_end)


def count_bubbles(bubble_masks: List[np.ndarray], min_size: int = 10,
                  Debug: bool = False, Save: bool = False, savepath: str = "") -> int:
    """
    Count detected bubbles across spectral masks and combined mask.

    Args:
        bubble_masks (List[np.ndarray]): List of binary masks for bubble regions.
        min_size (int, optional): Minimum bubble size in pixels. Default is 10.
        Debug (bool, optional): Print debug info if True. Default is False.
        Save (bool, optional): Save results if True. Default is False.
        savepath (str, optional): Path to save outputs if Save=True.

    Returns:
        int: Total number of unique bubbles detected.
    """
    bubble_counts = count.count_bubbles_per_band(bubble_masks, min_size, Debug)
    print("Bubble counts per band:", bubble_counts)

    return count.count_bubbles_in_combined_mask(bubble_masks, min_size, Debug, Save, savepath)


def analyze_bubbles(bubble_mask: np.ndarray, start_px: int, pixels_per_cm: float,
                    top_depth: float, min_size: float, Debug: bool = False,
                    Save: bool = False, savepath: str = "") -> Tuple[List[Dict[str, Any]], pd.DataFrame, float]:
    """
    Analyze bubble properties (area, centroid, orientation, elongation) from a mask.

    Args:
        bubble_mask (np.ndarray): 2D binary mask of detected bubbles.
        start_px (int): Pixel index where cropped image starts.
        pixels_per_cm (float): Conversion factor from pixels to centimeters.
        top_depth (float): Depth at top of core section (meters).
        min_size (float): Minimum bubble size to analyze.
        Debug (bool, optional): Print bubble properties if True.
        Save (bool, optional): Save visualization if True.
        savepath (str, optional): Output directory if Save=True.

    Returns:
        Tuple:
            - analyzed (List[Dict]): List of bubble property dictionaries.
            - bubble_df (pd.DataFrame): DataFrame of bubble region properties.
            - bubble_area_ratio (float): Ratio of bubble area to total image area (%).
    """
    analyzed, bubble_df, bubble_area_ratio = bubbleAnalysis.analyze_bubbles(bubble_mask, start_px, pixels_per_cm, top_depth, min_size)

    if Debug:
        for i, bubble in enumerate(analyzed, 1):
            print(f"Bubble {i}:")
            print(f"  Area: {bubble['area']}")
            print(f"  Centroid: {bubble['centroid']}")
            print(f"  Orientation: {bubble['orientation']} radians")
            print(f"  Elongation: {bubble['elongation']}")
            print(f"  Bounding Box: {bubble['bounding_box']}")

    bubbleAnalysis.visualize_bubble_analysis(bubble_mask, analyzed, Save, savepath)
    return analyzed, bubble_df, bubble_area_ratio


def get_stats(bubble_df: pd.DataFrame, px_per_cm: float) -> pd.DataFrame:
    """
    Compute summary statistics (mean size, elongation, etc.) for detected bubbles.

    Args:
        bubble_df (pd.DataFrame): DataFrame of bubble properties.
        px_per_cm (float): Conversion factor from pixels to centimeters.

    Returns:
        pd.DataFrame: Summary statistics of bubble features.
    """
    return stats.compute_bubble_statistics(bubble_df, px_per_cm)


def process_bubbles_per_cm(cropped_image: np.ndarray, analyzed_bubbles: List[Dict[str, Any]],
                           segment_boundaries: List[Tuple[int, int]], crop_start_px: int,
                           crop_end_px: int, original_start_px: int, pixels_per_cm: float,
                           top_depth_m: float) -> pd.DataFrame:
    """
    Count bubbles per centimeter depth segment in cropped ice image.

    Args:
        cropped_image (np.ndarray): Cropped hyperspectral image.
        analyzed_bubbles (List[Dict]): Bubble analysis results.
        segment_boundaries (List[Tuple[int, int]]): Pixel ranges for each 1-cm segment.
        crop_start_px (int): Start pixel of cropped image.
        crop_end_px (int): End pixel of cropped image.
        original_start_px (int): Original image's ice start pixel.
        pixels_per_cm (float): Pixels-to-centimeter conversion factor.
        top_depth_m (float): Depth at top of core section (meters).

    Returns:
        pd.DataFrame: Bubble counts per cm of depth.
    """
    print(f"crop_start_px: {crop_start_px}")
    print(f"crop_end_px: {crop_end_px}")
    print(f"crop_end_px: {crop_end_px}")
    print(f"original_start_px: {original_start_px}")
    print(f"pixels_per_cm: {pixels_per_cm}")
    print(f"top_depth_m: {top_depth_m}")
    
    
    return bubbles_per_cm.process_bubbles_per_cm(
        cropped_image, analyzed_bubbles, segment_boundaries,
        crop_start_px, crop_end_px, original_start_px,
        pixels_per_cm, top_depth_m
    )


def graph_bubble_stats(df_bubbles: pd.DataFrame, save: bool = False, savepath: str = "") -> None:
    """
    Generate plots of bubble statistics.

    Args:
        df_bubbles (pd.DataFrame): DataFrame containing bubble statistics per cm.
        save (bool, optional): If True, save figure to disk.
        savepath (str, optional): Output path for saved figure.
    """
    graphStats.graph(df_bubbles, save, savepath)