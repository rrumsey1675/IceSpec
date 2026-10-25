#!/usr/bin/env python3

# ───────────────────────────────────────────────
# Script ID   : spectralBubbles_v3.3
# Description : Full pipeline for trimmed, filtered HSI image bubble analysis. 
#               Updated to use watershed segmentation.
# Author      : Roisin Rumsey
# Date        : 2025-10-16
# Notes       : Outputs plots as PNG and data as CSV files.
# ───────────────────────────────────────────────

SCRIPT_ID = "spectralBubbles_v3.3"
notes = "Bubble Analysis // SNR Updated"

"""
This script processes hyperspectral images of ice and extracts information about bubbles,
including segmentation, measurement, and visualization.

Version 3.3:
- Added watershed segmentation for bubble detection.
- Outputs bubble statistics, plots, and derived datasets.
- Commented and has 'helper functions'
- 3.2 bubbles analyzed in 5x5cm units in center of core
- 3.3 bubbles stats by median rather than mean
"""

# ───────────────────────────────────────────────
# Import standard libraries
# ───────────────────────────────────────────────
import warnings
import sys
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    # ───────────────────────────────────────────────
    # 1. SETUP: Add processing directories to Python path
    # ───────────────────────────────────────────────
    processing_path = os.path.abspath(os.path.join('processing'))
    if processing_path not in sys.path:
        sys.path.append(processing_path)
    
    # Import project-specific libraries
    import set_icespec_path as tool
    import set_bubbles_path as tool2

    # Add code directories dynamically
    icespec_code_path = tool.add_icespec_code_dir()
    bubble_code_path = tool2.add_bubble_code_dir()

    from icespec import icespec_process2_module as processice
    from bubbles import bubbles_process_module as processbubbles
    from icespec import savePNG

    # ───────────────────────────────────────────────
    # 2. ARGUMENT PARSING & DATASET PATH SELECTION
    # ───────────────────────────────────────────────
    DataID = sys.argv[1]
    NumID = int(DataID.split('-')[-1])
    
    datasetPath = getCSV(NumID)
    outputDir = Path("/data/icecores/Output")  # base output directory
    
    # ── Batch hooks (argv → env fallback) ─────────────────────
    batch_id = sys.argv[2] if len(sys.argv) > 2 else os.getenv("BATCH_ID", "unbatched")
    batch_ts = os.getenv("BATCH_TIMESTAMP", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    script_id = os.getenv("SCRIPT_ID", "unknown_script")
    
    Debug = False
    # ───────────────────────────────────────────────
    # 3. DATA RETRIEVAL AND RAW IMAGE EXPORT
    # ───────────────────────────────────────────────
    dataPath, top_depth, bottom_depth, core_length, save_path, px_dims, metadata = processice.getDataPath(datasetPath, DataID, outputDir)
    outputPath = processice.make_dir(save_path, SCRIPT_ID, metadata, f"{DataID}-{batch_ts}", notes, batch_id)

    # Load ENVI hyperspectral data
    data_nparr, w_ref, d_ref, wvls, fwhm = processice.getEnviData(dataPath)

    # Save raw RGB composite
    save_rgb_image(data_nparr, outputPath, f"{DataID}raw.png", DataID, "RGB Image of Raw Data")

    # ───────────────────────────────────────────────
    # 4. IMAGE PRE-PROCESSING (Trim → Reflectance → Filter)
    # ───────────────────────────────────────────────
    segment_boundaries, pixels_per_cm = processbubbles.get_segments(data_nparr, top_depth, bottom_depth, start_px=px_dims[0], end_px=px_dims[1])
    stdSize = [530, 820, 200, 900]  # Standard crop dimensions
    trimmedData = processice.trim(data_nparr, w_ref, d_ref, px_dims, stdSize, Debug=Debug)

    # Save trimmed RGB image
    save_rgb_image(trimmedData[0], outputPath, f"{DataID}trimmed.png", DataID, "RGB Image of Trimmed Raw Data")

    # Reflectance correction and band filtering
    correctedData = processice.reflectance(trimmedData, Debug)
    filtered_img, wvls, fwhm = processice.remove_bands(correctedData, trimmedData[1], wvls, fwhm, Debug=False, outputPath=outputPath, save=True)
    save_rgb_image(filtered_img, outputPath, f"{DataID}filtered.png", DataID, "Filtered Data (Post Reflectance Correction)")

    # ───────────────────────────────────────────────
    # 5. BUBBLE DETECTION & ANALYSIS
    # ───────────────────────────────────────────────
    bubbles_only, bubble_masks, bubble_only_2d, bubble_channel_map = processbubbles.get_bubbles(filtered_img, fwhm, clean=False, debug=Debug)
    
    # Save bubble-channel map
    df_map = pd.DataFrame([
    {"Bubble_ID": b_id, "Bands": bands, "Num_Bands": len(bands)}
    for b_id, bands in bubble_channel_map.items()
    ])
    DSName = outputPath.parent.name
    df_map.to_csv(f"{outputPath}/{DSName}bubbles_channel_map.csv", index=False)
    append_readme(outputPath, f"{DSName}bubbles_channel_map.csv", "CSV of Bubble Channel Mapping")

    # Apply mask and Watershed Segmentation to bubbles
    segmented_img, x_coords, y_coords = processbubbles.rad_watershed(filtered_img, bubble_only_2d)
    watershed_boundaries, watershed_boundary_pixels = processbubbles.getBoundaries(segmented_img, x_coords, y_coords, filtered_img, Save=True, savepath=outputPath)

    # Analyze bubbles and calculate area ratio
    analyzed_bubbles, bubble_df, bubble_area_ratio = processbubbles.analyze_bubbles(
        bubble_only_2d, px_dims[0], pixels_per_cm, top_depth, min_size=10, Debug=Debug, Save=True, savepath=outputPath
    )
    append_readme(outputPath, None, f"Bubble Area Ratio: {bubble_area_ratio}%")

    # Generate bubble statistics
    stats_df = processbubbles.get_stats(bubble_df, pixels_per_cm)

    # Save bubble properties and stats
    save_dataframe(bubble_df, outputPath, f"{DSName}_bubbleData.csv", "CSV of all detected bubbles and their properties")
    df_bubbles_per_cm = processbubbles.process_bubbles_per_cm(filtered_img, analyzed_bubbles, segment_boundaries, px_dims[0], px_dims[1], 2470, pixels_per_cm, top_depth)
    save_dataframe(df_bubbles_per_cm, outputPath, f"{DSName}bubbles_per_cm.csv", "Bubble stats per cm")
    save_dataframe(stats_df, outputPath, f"{DSName}_bubble_stats.csv", "Bubble stats across whole core section")

    # Plot bubble statistics
    processbubbles.graph_bubble_stats(df_bubbles_per_cm, save=True, savepath=outputPath)


# ───────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np # Make sure numpy is imported if not already

def save_rgb_image(img, output_path, filename, data_id, description):
    """Save an RGB slice of hyperspectral data at 300 DPI and log it."""
    
    # --- Data Processing (Same as before) ---
    n_bands = img.shape[2]
    if n_bands >= 97:
        band_indices = [96, 59, 16]
    else:
        # Pick 3 roughly evenly spaced bands
        band_indices = [max(0, n_bands - 1), n_bands // 2, 0]
        # Ensure we have at most 3 unique indices, sorted descending
        band_indices = sorted(list(set(band_indices)), reverse=True)[:3]
        
    rgb_slice = img[:, :, band_indices].astype(np.float32)
    # Normalize the data to the 0-1 range for plotting
    rgb_norm = (rgb_slice - rgb_slice.min()) / (rgb_slice.max() - rgb_slice.min())
    
    # --- Matplotlib Figure Creation and Saving (NEW) ---
    
    # 1. Create a new figure and axes
    # We set figsize based on the image shape to ensure the output image
    # has the same aspect ratio as the data. 
    # The image shape is (Height, Width, Channels)
    H, W, C = rgb_norm.shape
    
    # Determine the size of the plot in inches, making sure the final 
    # saved image is scaled correctly by the DPI (300).
    # We use 1/300 as the scaling factor.
    fig_width = W / 300.0
    fig_height = H / 300.0
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 2. Display the image
    ax.imshow(rgb_norm)
    
    # 3. Remove all axes, ticks, and borders so only the image data remains
    ax.axis('off')
    
    # 4. Remove the white border/margin around the image
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # 5. Save the figure with the desired DPI
    full_output_path = f"{output_path}/{filename}"
    plt.savefig(full_output_path, 
                dpi=300, 
                bbox_inches='tight', # Ensures no extra padding is added 
                pad_inches=0)      # Sets the padding to zero
                
    # 6. Close the figure to free memory
    plt.close(fig)
    
    # --- Logging (Same as before) ---
    # append_readme(output_path, filename, f"{data_id}: {description}") # Assuming this function exists elsewhere
    print(f"Image saved to {full_output_path} at 300 DPI.")


def append_readme(output_path, filename, description):
    """Append output file info or notes to README.md."""
    readme_path = output_path / "README.md"
    with readme_path.open("a") as f:
        if filename:
            f.write(f"\n###  Output File: {filename}\n- {description}\n")
        else:
            f.write(f"\n###  {description}\n")


def save_dataframe(df, output_path, filename, description):
    """Save a DataFrame to CSV and record it in README.md."""
    df.to_csv(f"{output_path}/{filename}", index=False)
    append_readme(output_path, filename, description)

def getCSV(NumID):
    """Grab correct csv file for image data."""
    if NumID <= 228:
        datasetPath = '/data/icecores/HSI_2024.csv'
    elif 229 <= NumID < 335:
        datasetPath = '/data/icecores/HSI_2023_1.csv'
    elif 335 <= NumID < 363:
        datasetPath = '/data/icecores/HSI_June2025.csv'
    elif 363 <= NumID < 586:
        datasetPath = '/data/icecores/HSI_Sept2025.csv'
    elif 586 <= NumID < 774:
        datasetPath = '/data/icecores/HSI_Denali2025.csv'
    elif 774 <= NumID < 896:
        datasetPath = '/data/icecores/HSI_Jan2024.csv'
    else:
        raise ValueError(f"NumID {NumID} is outside supported ranges (<=228, 229–334, 335–362, 363–585, 586–750, 751–872). \n Please check DataID and try again.")
    return datasetPath


# ───────────────────────────────────────────────
if __name__ == "__main__":
    main()
