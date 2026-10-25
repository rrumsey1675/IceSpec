# rumz Jan 25

import numpy as np
import pandas as pd
from bubbles import bubbles_count_module as count_bubbles
from bubbles import bubble_analysis_module as analyze

def process_bubbles_per_cm(
    cropped_image: np.ndarray,
    bubble_properties: list[dict],
    segment_boundaries: list[tuple[int, int]],
    crop_start_px: int,
    crop_end_px: int,
    original_start_px: int,
    pixels_per_cm: float,
    top_depth: float,
    slice_thickness_cm: float = 3.0
) -> pd.DataFrame:
    """
    Processes bubble segmentation and counting for a cropped ice core image.

    Args:
    - cropped_image (np.ndarray): Cropped ice core image.
    - segment_boundaries (list of tuples): Original (start_px, end_px) per cm.
    - crop_start_px (int): Start pixel of the cropped image.
    - crop_end_px (int): End pixel of the cropped image.
    - original_start_px (int): Original ice start pixel before cropping.
    - pixels_per_cm (float): Conversion factor from pixels to cm.
    - top_depth (float): Core section depth for table.

    Returns:
    - pd.DataFrame: DataFrame with depth in cm and bubble counts.
    """
    crop_start_px = int(crop_start_px)
    crop_end_px = int(crop_end_px)
    
    # Adjust segment boundaries for the cropped image
    adjusted_segments, new_pixels_per_cm = adjust_boundaries_for_crop(segment_boundaries, crop_start_px, crop_end_px, 
                                                                     original_start_px, pixels_per_cm)

    # Count and normalize bubble data in each adjusted segment
    total_counts, normalized_counts, BND, ice_to_bubble_ratios, avg_bubble_sizes, avg_elongations, avg_eccentricities, avg_orientations, std_sizes, std_elongations, std_eccentricities, std_orientations, median_sizes, median_elongations, median_eccentricities, median_orientations, median5_sizes, median5_elongations, median5_eccentricities, median5_orientations, median95_sizes, median95_elongations, median95_eccentricities, median95_orientations, trimmed_mean_sizes, trimmed_mean_elongations, trimmed_mean_eccentricities, trimmed_mean_orientations = count_bubbles_in_segments(
        cropped_image, bubble_properties, adjusted_segments, new_pixels_per_cm, slice_thickness_cm
    )

    # Generate the output DataFrame
    bubble_segments_df = generate_bubble_dataframe(
        total_counts, normalized_counts, ice_to_bubble_ratios, avg_bubble_sizes,
        BND, avg_elongations, avg_eccentricities, avg_orientations, std_sizes, std_elongations, std_eccentricities, std_orientations, median_sizes, median_elongations, median_eccentricities, median_orientations, median5_sizes, median5_elongations, median5_eccentricities, median5_orientations, median95_sizes, median95_elongations, median95_eccentricities, median95_orientations, trimmed_mean_sizes, trimmed_mean_elongations, trimmed_mean_eccentricities, trimmed_mean_orientations, adjusted_segments, top_depth
    )
    return bubble_segments_df

def adjust_boundaries_for_crop(segment_boundaries, crop_start_px, crop_end_px, original_start_px, pixels_per_cm):
    """
    Adjusts the segment boundaries to match a cropped image.

    Args:
    - segment_boundaries (list of tuples): Original (start_px, end_px) per cm.
    - crop_start_px (int): New start pixel after cropping.
    - crop_end_px (int): New end pixel after cropping.
    - original_start_px (int): Original ice start pixel before cropping.
    - pixels_per_cm (float): Conversion factor from pixels to cm.

    Returns:
    - adjusted_segments (list of tuples): Updated (start_px, end_px) per cm.
    - new_pixels_per_cm (float): Adjusted conversion factor.
    """

    adjusted_segments = []
    for start, end in segment_boundaries:
        new_start = start - crop_start_px
        new_end = end - crop_start_px

        # Keep only fully visible segments
        if new_start >= 0 and new_end <= (crop_end_px - crop_start_px):
            adjusted_segments.append((new_start, new_end))

    # Update pixels_per_cm for the cropped image
    new_ice_height_px = crop_end_px - crop_start_px
    new_pixels_per_cm = new_ice_height_px / len(adjusted_segments)  

    return adjusted_segments, new_pixels_per_cm

def count_bubbles_in_segments(
    cropped_image: np.ndarray,
    bubble_properties: list[dict],
    adjusted_segments: list[tuple[int, int]],
    pixels_per_cm: float,
    slice_thickness_cm: float = 3.0
) -> tuple[
    list[int], list[float], list[float], list[float],
    list[float], list[float], list[float], list[float]
]:
    """
    Computes bubble counts and morphology for the centermost 5 cm x 5 cm area of each segment.

    Returns:
    - total_counts: Total bubbles in central tile
    - normalized_counts: Bubbles per 25 cm² (always 25 for this tile)
    - BND: Bubble number density (bubbles/cm³)
    - ice_to_bubble_ratios: % of tile that is bubble pixels
    - avg_bubble_sizes, avg_elongations, avg_eccentricities, avg_orientations
    """
    total_counts, normalized_counts, density, BND = [], [], [], []
    ice_to_bubble_ratios, avg_bubble_sizes = [], []
    avg_elongations, avg_eccentricities, avg_orientations = [], [], []
    std_sizes, std_elongations, std_eccentricities, std_orientations = [], [], [], []
    median_sizes, median_elongations, median_eccentricities, median_orientations = [], [], [], []
    median5_sizes, median5_elongations, median5_eccentricities, median5_orientations = [], [], [], []
    median95_sizes, median95_elongations, median95_eccentricities, median95_orientations = [], [], [], []
    trimmed_mean_sizes, trimmed_mean_elongations, trimmed_mean_eccentricities, trimmed_mean_orientations = [], [], [], []

    width_px = cropped_image.shape[1]
    tile_px = int(round(5.0 * pixels_per_cm))  # 5 cm in pixels
    tile_area_cm2 = 25.0  # 5 x 5 cm

    n_full_blocks = (len(adjusted_segments) // 5) * 5

    for i in range(0, n_full_blocks, 5):
        # Combine 5 consecutive 1-cm segments into a single 5-cm block
        group = adjusted_segments[i:i + 5]
        # print("Group: ", group)
        group_start = group[0][0]
        group_end = group[-1][1]
        group_height_px = max(0, group_end - group_start)
        if group_height_px <= 0:
            print("group_height_px <= 0")
            # Shouldn't happen
            total_counts.append(0)
            normalized_counts.append(0.0)
            density.append(0.0)
            BND.append(0.0)
            ice_to_bubble_ratios.append(0.0)
            avg_bubble_sizes.append(0.0)
            avg_elongations.append(0.0)
            avg_eccentricities.append(0.0)
            avg_orientations.append(0.0)
            continue

        # print("group_height_px = ", group_height_px)

        # Center of the 5-cm block
        x_center = width_px // 2
        y_center = group_start + group_height_px // 2

        # Compute 5x5 cm tile at center
        x_start = x_center - tile_px // 2
        x_end = x_start + tile_px
        if x_start < 0:
            x_end -= x_start  # shift right
            x_start = 0
        if x_end > width_px:
            shift = x_end - width_px
            x_start = max(0, x_start - shift)
            x_end = width_px

        y_start = y_center - tile_px // 2
        y_end = y_start + tile_px
        if y_start < group_start:
            # shift down to fit inside block
            y_end += (group_start - y_start)
            y_start = group_start
        if y_end > group_end:
            shift = y_end - group_end
            y_start = max(group_start, y_start - shift)
            y_end = group_end

        # In rare rounding cases, ensure full tile size vertically
        # if (y_end - y_start) < tile_px or (x_end - x_start) < tile_px:
        #     # No partial tiles for consistency
        #     print("partial tile")
        #     continue

        # Bubbles w/ centroids in the 5x5 tile
        tile_bubbles = [
            b for b in bubble_properties
            if (x_start <= b["x"] < x_end) and (y_start <= b["y"] < y_end)
        ]
        count_tile = len(tile_bubbles)
        # print(f"count_tile for group{group}: {count_tile}")
        total_counts.append(count_tile)

        # Normalized to 25 cm² (same as count cuz tile area is fixed)
        normalized_counts.append(count_tile)

        # Density (bubbles/cm²) and BND (bubbles/cm³)
        dens_cm2 = count_tile / tile_area_cm2
        density.append(dens_cm2)
        BND.append(dens_cm2 / slice_thickness_cm if slice_thickness_cm > 0 else 0.0)

        # Morphology
        if count_tile > 0:
            areas_px2 = [b["area"] for b in tile_bubbles]
            total_bubble_area_px2 = float(sum(areas_px2))

            
            areas_px2 = np.array([b["area"] for b in tile_bubbles])
            elongations = np.array([b["elongation"] for b in tile_bubbles])
            eccentricities = np.array([b["eccentricity"] for b in tile_bubbles])
            orientations = np.array([b["orientation"] for b in tile_bubbles])
            
            def mean_within_std(values: np.ndarray) -> float:
                """Return mean with values outside 1 std from mean removed."""
                mean = np.mean(values)
                std = np.std(values)
                mask = (values >= mean - std) & (values <= mean + std)
                filtered = values[mask]
                return float(np.mean(filtered)) if len(filtered) > 0 else float("nan")
            
            # Means
            mean_size = float(np.mean(areas_px2))
            mean_elongation = float(np.mean(elongations))
            mean_eccentricity = float(np.mean(eccentricities))
            mean_orientation = float(np.mean(orientations))
            
            # Standard deviations
            std_size = float(np.std(areas_px2, ddof=1))
            std_elongation = float(np.std(elongations, ddof=1))
            std_eccentricity = float(np.std(eccentricities, ddof=1))
            std_orientation = float(np.std(orientations, ddof=1))
            
            # Medians
            median_size = float(np.median(areas_px2))
            median_elongation = float(np.median(elongations))
            median_eccentricity = float(np.median(eccentricities))
            median_orientation = float(np.median(orientations))

            # 5th percentiles
            median5_size = float(np.percentile(areas_px2, 5))
            median5_elongation = float(np.percentile(elongations, 5))
            median5_eccentricity = float(np.percentile(eccentricities, 5))
            median5_orientation = float(np.percentile(orientations, 5))
            
            # 95th percentiles
            median95_size = float(np.percentile(areas_px2, 95))
            median95_elongation = float(np.percentile(elongations, 95))
            median95_eccentricity = float(np.percentile(eccentricities, 95))
            median95_orientation = float(np.percentile(orientations, 95))
            
            # Trimmed means (excluding points outside ±1 std)
            trimmed_mean_size = mean_within_std(areas_px2)
            trimmed_mean_elongation = mean_within_std(elongations)
            trimmed_mean_eccentricity = mean_within_std(eccentricities)
            trimmed_mean_orientation = mean_within_std(orientations)

            
        else:
            total_bubble_area_px2 = 0.0
            median_size = median_elongation = median_eccentricity = median_orientation = 0.0
            mean_size = mean_elongation = mean_eccentricity = mean_orientation = 0.0
            std_size = std_elongation = std_eccentricity = std_orientation = 0.0
            trimmed_mean_size = trimmed_mean_elongation = trimmed_mean_eccentricity = trimmed_mean_orientation = 0.0
            median5_size = median5_elongation = median5_eccentricity = median5_orientation = 0.0
            median95_size = median95_elongation = median95_eccentricity = median95_orientation = 0.0

        # % of 5x5 tile area that’s bubble pixels (pixel-based ratio)
        tile_area_px = tile_px * tile_px
        ice_to_bubble_ratio = (
            round((total_bubble_area_px2 / tile_area_px) * 100.0, 2)
            if tile_area_px > 0 else 0.0
        )

        ice_to_bubble_ratios.append(ice_to_bubble_ratio)
        avg_bubble_sizes.append(mean_size)
        avg_elongations.append(mean_elongation)
        avg_eccentricities.append(mean_elongation)
        avg_orientations.append(mean_orientation)

        std_sizes.append(std_size)
        std_elongations.append(std_elongation)
        std_eccentricities.append(std_elongation)
        std_orientations.append(std_orientation)

        median_sizes.append(median_size)
        median_elongations.append(median_elongation)
        median_eccentricities.append(median_elongation)
        median_orientations.append(median_orientation)

        median5_sizes.append(median5_size)
        median5_elongations.append(median5_elongation)
        median5_eccentricities.append(median5_elongation)
        median5_orientations.append(median5_orientation)

        median95_sizes.append(median95_size)
        median95_elongations.append(median95_elongation)
        median95_eccentricities.append(median95_elongation)
        median95_orientations.append(median95_orientation)

        trimmed_mean_sizes.append(trimmed_mean_size)
        trimmed_mean_elongations.append(trimmed_mean_elongation)
        trimmed_mean_eccentricities.append(trimmed_mean_elongation)
        trimmed_mean_orientations.append(trimmed_mean_orientation)

    #print("Depth_cm length: ", len(depths_cm))
    # print("Bubble_Count length: ", len(total_counts))
    # print("Avg_Bubble_Size length: ", len(avg_bubble_sizes))
    # print("Bubble_Area_Ratio length: ", len(ice_to_bubble_ratios))
    # print("BND length: ", len(BND))
    # print("Avg_Elongation length: ", len(avg_elongations))
    # print("Avg_Eccentricity length: ", len(avg_eccentricities))
    # print("Avg_Orientation length: ", len(avg_orientations))

    return (
        total_counts,          # bubbles in central 5x5 cm tile
        normalized_counts,     # bubbles per 25 cm²
        BND,                   # bubbles/cm³
        ice_to_bubble_ratios,  # % tile area that's bubbles
        avg_bubble_sizes,
        avg_elongations,
        avg_eccentricities,
        avg_orientations,
        std_sizes,
        std_elongations,
        std_eccentricities,
        std_orientations,
        median_sizes,
        median_elongations,
        median_eccentricities,
        median_orientations,
        median5_sizes,
        median5_elongations,
        median5_eccentricities,
        median5_orientations,
        median95_sizes,
        median95_elongations,
        median95_eccentricities,
        median95_orientations,
        trimmed_mean_sizes,
        trimmed_mean_elongations,
        trimmed_mean_eccentricities,
        trimmed_mean_orientations
    )
 



def generate_bubble_dataframe(
    total_counts: list[int],
    normalized_counts: list[int],
    ice_to_bubble_ratios: list[float],
    avg_bubble_sizes: list[float],
    BND: list[float],
    avg_elongations: list[float],
    avg_eccentricities: list[float],
    avg_orientations: list[float],
    std_sizes: list[float], 
    std_elongations: list[float], 
    std_eccentricities: list[float], 
    std_orientations: list[float], 
    median_sizes: list[float], 
    median_elongations: list[float],
    median_eccentricities: list[float],
    median_orientations: list[float],
    median5_sizes: list[float],
    median5_elongations: list[float],
    median5_eccentricities: list[float],
    median5_orientations: list[float],
    median95_sizes: list[float],
    median95_elongations: list[float],
    median95_eccentricities: list[float],
    median95_orientations: list[float],
    trimmed_mean_sizes: list[float],
    trimmed_mean_elongations: list[float],
    trimmed_mean_eccentricities: list[float],
    trimmed_mean_orientations: list[float],
    adjusted_segments: list[tuple[int, int]],
    top_depth: float
) -> pd.DataFrame:
    """
    Creates a DataFrame with depth-resolved bubble morphology and density metrics.

    Args:
    - total_counts: Number of bubbles in each segment.
    - normalized_counts: Average number of bubbles per cm^2, per cm of depth
    - ice_to_bubble_ratios: Percent area of each segment occupied by bubbles.
    - avg_bubble_sizes: Average bubble size (in px) in each segment.
    - BND: Bubble Number Density per cm^3 for each segment.
    - avg_elongations: Average elongation per segment.
    - avg_eccentricities: Average eccentricity per segment.
    - avg_orientations: Average orientation per segment.
    - adjusted_segments: Pixel ranges for each 1 cm segment.
    - top_depth: Depth (in meters) at top of the core section.

    Returns:
    - pd.DataFrame with all metrics per 1 cm depth increment.
    """
    start_cm = int(top_depth * 100)
    depths_cm = np.arange(start_cm, start_cm + (len(adjusted_segments) // 5) * 5, 5)

    # print("Depth_cm length: ", len(depths_cm))
    # print("Depth_cm: ", depths_cm)
    # print("Bubble_Count length: ", len(total_counts))
    # print("Avg_Bubble_Size length: ", len(avg_bubble_sizes))
    # print("Bubble_Area_Ratio length: ", len(ice_to_bubble_ratios))
    # print("BND length: ", len(BND))
    # print("Avg_Elongation length: ", len(avg_elongations))
    # print("Avg_Eccentricity length: ", len(avg_eccentricities))
    # print("Avg_Orientation length: ", len(avg_orientations))

    df = pd.DataFrame({
        "Depth_cm": depths_cm,
        "Bubble_Count": total_counts,
        "Normalized_Count": normalized_counts,
        "Bubble_Number_Density": BND,
        "Avg_Bubble_Size": avg_bubble_sizes,
        "Bubble_Area_Ratio": ice_to_bubble_ratios,
        "Avg_Elongation": avg_elongations,
        "Avg_Eccentricity": avg_eccentricities,
        "Avg_Orientation": avg_orientations,
        "STD_Bubble_Size": std_sizes,
        "STD_Elongation": std_elongations,
        "STD_Eccentricity": std_eccentricities,
        "STD_Orientation": std_orientations,
        "Median_Bubble_Size": median_sizes,
        "Median_Elongation": median_elongations,
        "Median_Eccentricity": median_eccentricities,
        "Median_Orientation": median_orientations,
        "Median5_Bubble_Size": median5_sizes,
        "Median5_Elongation": median5_elongations,
        "Median5_Eccentricity": median5_eccentricities,
        "Median5_Orientation": median5_orientations,
        "Median95_Bubble_Size": median95_sizes,
        "Median95_Elongation": median95_elongations,
        "Median95_Eccentricity": median95_eccentricities,
        "Median95_Orientation": median95_orientations,
        "Trimmed_Mean_Bubble_Size": trimmed_mean_sizes,
        "Trimmed_Mean_Elongation": trimmed_mean_elongations,
        "Trimmed_Mean_Eccentricity": trimmed_mean_eccentricities,
        "Trimmed_Mean_Orientation": trimmed_mean_orientations
    })

    return df