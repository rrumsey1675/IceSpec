# rumz Jan 25

import numpy as np

from pathlib import Path

def log_and_average(value: float, filepath: str) -> float:
    """
    Appends a numeric value to the given file and returns the average
    of all values currently stored in that file.

    Args:
        value (float): The numeric value to append.
        filepath (str): Path to the text file for logging.

    Returns:
        float: Average of all values in the file (including the new one).
    """
    file = Path(filepath)
    file.parent.mkdir(parents=True, exist_ok=True)

    # Append value to file
    with file.open("a") as f:
        f.write(f"{value}\n")

    # Read back all values and compute average
    with file.open("r") as f:
        numbers = [float(line.strip()) for line in f if line.strip()]

    avg = sum(numbers) / len(numbers)
    return avg


def get_ice_segments(image: np.ndarray, core_top_depth: float, core_bottom_depth: float, 
                               start_px: int = 1200, end_px: int = 7100, detect_end: bool = False):
    """
    Determines the pixel boundaries for 1-cm ice segments in an image.

    Args:
    - image (np.ndarray): The full hyperspectral image (height, width, bands).
    - core_top_depth (float): Depth at the top of the core section (in meters).
    - core_bottom_depth (float): Depth at the bottom of the core section (in meters).
    - start_px (int, optional): Fixed pixel location where ice starts. Default is 1200.
    - end_px (int, optional): Fixed pixel location where ice ends. Ignored if detect_end=True.
    - detect_end (bool, optional): If True, automatically detects the end of the ice region.
    - Debug (bool, optional): If True, outputs more info to notebook

    Returns:
    - segment_boundaries (list of tuples): List of (start_px, end_px) tuples for each 1-cm segment.
    - px_per_cm (float): Conversion factor from pixels to cm.
    """
    # Detect the end of the ice region if enabled
    if detect_end:
        intensity_profile = np.sum(image, axis=(1,2))  # Summing across width & bands
        threshold = np.max(intensity_profile) * 0.05  # 5% of max intensity
        valid_px = np.where(intensity_profile > threshold)[0]
        end_px = valid_px[-1] if len(valid_px) > 0 else image.shape[0]  # Last bright row
    
    if end_px is None:
        raise ValueError("Either provide `end_px` or set `detect_end=True` to determine it dynamically.")
    
    # Compute conversion factor
    core_length_cm = (core_bottom_depth - core_top_depth) * 100
    print(f"core bottom depth: {core_bottom_depth} - core top depth: {core_top_depth} = core length: {core_length_cm/100}")
    print(f"core length cm: {core_length_cm}")
    ice_height_px = end_px - start_px
    print(f"end px: {end_px} - start px: {start_px} = total px height: {ice_height_px}")
    px_per_cm = ice_height_px / core_length_cm  # Pixels per cm
    print(f"ice height px: {ice_height_px} / core length cm: {core_length_cm} = px per cm: {px_per_cm}")


    px_per_cm = 63.976377952755904  # Pixels per cm
    print(f"standard px height: {325} / standard length: {5.08} = px per cm: {px_per_cm}")

    current_avg = log_and_average(px_per_cm, "/hsidata/hsiCode/stat/px_per_cm.txt")

    # Define segment boundaries
    segment_boundaries = []
    for cm in range(int(core_length_cm)):
        px_start = start_px + int(cm * px_per_cm)
        px_end = start_px + int((cm + 1) * px_per_cm)
        if px_end <= end_px:
            segment_boundaries.append((px_start, px_end))

    return segment_boundaries, px_per_cm