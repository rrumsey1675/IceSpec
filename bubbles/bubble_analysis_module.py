# rumz / Jan 25
# shape, elongation, orientation

import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple


def analyze_bubbles(
    bubble_mask: np.ndarray,
    crop_start_px: int,
    pixels_per_cm: float,
    top_depth: float,
    min_size: int = 5
) -> Tuple[List[Dict], pd.DataFrame, float]:
    """
    Analyze bubble size, elongation, orientation, and depth from a segmentation mask.

    Args:
        bubble_mask (np.ndarray): 2D array (boolean or labeled) indicating bubble regions.
        crop_start_px (int): Pixel index where the cropped image starts relative to full core image.
        pixels_per_cm (float): Conversion factor from pixels to centimeters.
        top_depth (float): Depth (in meters) at the top of the core section.
        min_size (int, optional): Minimum bubble area (pixels) to include. Defaults to 5.

    Returns:
        Tuple:
            - bubble_properties (List[Dict]): List of bubble attributes including size, centroid, elongation, orientation, etc.
            - df (pd.DataFrame): DataFrame of bubble properties for further analysis.
            - bubble_area_ratio (float): Percentage of the image area occupied by bubbles.
    """
    # Ensure the mask is labeled
    labeled_mask = label(bubble_mask)[0] if bubble_mask.dtype == bool else bubble_mask.astype(int)
    properties = regionprops(labeled_mask)

    bubble_properties: List[Dict] = []
    bubble_num = 1

    px_per_cm = 63.976377952755904
    px_per_mm = px_per_cm / 10.0
    mm_per_px = 1.0 / px_per_mm
    
    

    for region in properties:
        if region.area >= min_size:
            if region.area < 500:
                elongation = (
                    region.axis_major_length / region.axis_minor_length
                    if region.axis_minor_length > 0 else 1.0
                )
                radius = np.sqrt(region.area / np.pi)
                if elongation < 2.5:
                    orientation_rad = float(region.orientation)                # radians from row-axis
                    # Store both raw and display-friendly versions:
                    # orientation_xy_deg = (-np.degrees(orientation_rad)) % 180.0  # 0°=vertical, +/-90°=horizontal
                    orientation_xy_deg = np.degrees(orientation_rad)

                    area_mm2 = region.area * (mm_per_px ** 2)
                    
                    bubble_info = {
                        "Bubble": bubble_num,
                        "area": round(area_mm2, 4),
                        "centroid": (round(region.centroid[0], 0), round(region.centroid[1], 0)),  # (row, col)
                        "x": round(region.centroid[1], 4),  # col
                        "y": round(region.centroid[0], 4),  # row
                        "radius": round(radius, 2),
                        "orientation_rad": round(orientation_rad, 4),   # raw radians
                        "orientation": round(orientation_xy_deg),  # <-- use this in DF/plots
                        "elongation": round(elongation, 2),
                        "eccentricity": round(region.eccentricity, 2),
                        "bounding_box": [round(x, 0) for x in region.bbox],
                        "major_axis_length": float(region.axis_major_length),
                        "minor_axis_length": float(region.axis_minor_length),
                    }
                    bubble_properties.append(bubble_info)
                    bubble_num += 1


    # Add depth calculation to bubbles
    bubble_properties = tag_bubbles_with_depth(bubble_properties, crop_start_px, pixels_per_cm, top_depth)
    min_area = min(b["area"] for b in bubble_properties)
    max_area = max(b["area"] for b in bubble_properties)
    print(f"Minimum bubble size: {min_area}")
    print(f"Maximum bubble size: {max_area}")
    df = pd.DataFrame(bubble_properties)

    # Calculate bubble area ratio
    img_area = bubble_mask.shape[0] * bubble_mask.shape[1]
    bubble_area = np.sum(bubble_mask > 0)
    bubble_area_ratio = round((bubble_area / img_area) * 100, 2)

    print("Bubble Count:", len(df))
    return bubble_properties, df, bubble_area_ratio


def tag_bubbles_with_depth(
    bubble_properties: List[Dict],
    crop_start_px: int,
    pixels_per_cm: float,
    top_depth: float
) -> List[Dict]:
    """
    Annotate bubbles with depth based on y-coordinate and cropping offset.

    Args:
        bubble_properties (List[Dict]): List of bubble property dictionaries.
        crop_start_px (int): Pixel index where cropped image starts.
        pixels_per_cm (float): Conversion factor from pixels to centimeters.
        top_depth (float): Depth (in meters) at the top of the core section.

    Returns:
        List[Dict]: Updated bubble property list including depth in centimeters.
    """
    print(f"Top Depth = {top_depth}m")
    print(f"Crop Start Px = {crop_start_px}")
    print(f"Px Per cm = {pixels_per_cm}")
    
    
    for bubble in bubble_properties:
        y_px = bubble["y"]
        #print(f"Y Px: {y_px} = {bubble["y"]}")
        y_cm = y_px/pixels_per_cm
        #print(f"Y cm: {y_cm}")
        depth_cm = (top_depth * 100) + (y_cm)
        #print(f"Y Depth cm = {depth_cm}")
        #print("\n")
        bubble["depth_cm"] = round(depth_cm, 2)
    return bubble_properties


def visualize_bubble_analysis(
    mask: np.ndarray,
    analyzed: List[Dict],
    Save: bool = False,
    savepath: Path = Path("")
) -> None:
    """
    Visualize bubbles with orientation ellipses overlaid on mask.

    Args:
        mask (np.ndarray): 2D binary or labeled bubble mask.
        analyzed (List[Dict]): List of bubble property dictionaries from `analyze_bubbles`.
        Save (bool, optional): If True, saves visualization to disk. Defaults to False.
        savepath (Path, optional): Directory path to save output image and update README.md. Defaults to "".
    """
    # Generate some random data
    data = np.random.rand(10, 10)
    
    # Generate a list of random RGB colors
    num_colors = 256
    random_colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(num_colors)]
    
    # Create a custom colormap from the random colors
    random_cmap = ListedColormap(random_colors)

    
    plt.figure(figsize=(100, 100))
    plt.imshow(mask, interpolation='none')
    plt.title("Bubble Analysis")

    for bubble in analyzed:
        y, x = bubble["centroid"]  # (row, col)
        width  = bubble["major_axis_length"]
        height = bubble["minor_axis_length"]
        angle_deg = bubble["orientation"]  # 0° horizontal, 90° vertical
    
        ellipse = patches.Ellipse(
            (x, y),          # note (x, y) i.e., (col, row)
            width, height,
            angle=angle_deg,
            edgecolor='red',
            facecolor='none',
            lw=1,
            zorder=2
        )
        plt.gca().add_patch(ellipse)

    if Save:
        DSName = savepath.parent.name
        filename = f"{DSName}bubbleMap.png"
        plt.savefig(savepath / filename, format='png', dpi=300, bbox_inches='tight')

        readme_path = savepath / "README.md"
        with readme_path.open("a") as f:
            f.write(f"\n###  Output File: {filename}\n- {DSName}: Visualization of Bubble Analysis\n")

    plt.show()
