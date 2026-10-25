import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import segmentation, color, filters, measure, morphology
from skimage.morphology import remove_small_objects
from skimage.feature import peak_local_max
import numpy as np

def rad_watershed(bubble_only_2d: np.ndarray, median_2d) -> np.ndarray:
    # Validate inputs
    if not isinstance(bubble_only_2d, np.ndarray) or bubble_only_2d.ndim != 2:
        raise ValueError(f"img must be a 2D NumPy array, got shape {getattr(bubble_only_2d, 'shape', None)}")
    if not isinstance(median_2d, np.ndarray) or median_2d.ndim != 2:
        raise ValueError(f"median_2d must be a 2D NumPy array, got shape {getattr(median_2d, 'shape', None)}")
    
    H, W = bubble_only_2d.shape
    if median_2d.shape != (H, W):
        raise ValueError(f"median_2d shape {median_2d.shape} must match img spatial dimensions ({H}, {W})")

    labels = np.zeros((H, W), dtype=np.int32)
    imgf = bubble_only_2d.astype(np.float32, copy=False)

    # --- Define valid regions from median_2d ---
    valid_median = np.isfinite(median_2d)
    print(f"Median 2D shape: {median_2d.shape}; Valid Median 2D shape: {valid_median.shape}")
    if not np.any(valid_median):
        print("No valid pixels in median_2d, returning empty labels.")
        return labels
    med = np.nanmedian(median_2d[valid_median])
    print(f"Global Median Radiance (from median_2d): {med}")

    # --- Find peaks in median_2d for consistent markers ---
    coords = peak_local_max(median_2d, footprint=np.ones((3, 3)), threshold_abs=1)
    x_coords = coords[:, 1]
    y_coords = coords[:, 0]
    print(np.shape(coords))
    if len(coords) == 0:
        coords = peak_local_max(median_2d, footprint=np.ones((3, 3)), threshold_abs=med)
        x_coords = coords[:, 1]
        y_coords = coords[:, 0]
        if len(coords) == 0:
            print("No peaks found in median_2d, returning empty labels.")
            return labels, None, None

    # Create 2D mask for markers
    mask = np.zeros((H, W), dtype=bool)
    mask[tuple(coords.T)] = True
    markers, num_markers = ndi.label(mask)
    markers = markers.astype(np.int32)
    print(f"Found {num_markers} initial markers in median_2d")
    print(f"Markers stats: mean={np.mean(markers):.2f}, max={np.max(markers)}, "
          f"min={np.min(markers)}, std={np.std(markers):.2f}")
    marker_boundaries = segmentation.find_boundaries(markers, mode='inner')

    valid = np.isfinite(imgf) & valid_median


    # Create elevation map
    elev = -np.where(valid, imgf, med).astype(np.float32)
    
    plt.close()
    plt.figure(figsize=(60, 60))
    marker_boundary_pixels = np.where(marker_boundaries)
    #plt.scatter(x_coords, y_coords, c='red', s=4)
    plt.imshow(elev, cmap='gray')
    plt.scatter(marker_boundary_pixels[1], marker_boundary_pixels[0], c='red', s=10, marker='.', label='Marker Boundaries')
    plt.title("Elevation Map")
    #plt.savefig('/data/icecores/Output/AN/ALHIC/ALHIC2302/ALHIC2302-29/PCA_elevationMap.png', dpi=300, bbox_inches="tight")
    plt.show()

    # Run watershed
    labels = segmentation.watershed(elev, markers=markers, mask=valid)

    return labels, x_coords, y_coords