#!/usr/bin/env python3

# ───────────────────────────────────────────────
# Script ID   : spectralLayers_v2.3
# Description : Full pipeline for trimmed, filtered HSI image spectral sum comparison.
# Author      : Roisin Rumsey
# Date        : 2025-10-06
# Notes       : Outputs plt figs and sums in csv files
# ───────────────────────────────────────────────

SCRIPT_ID = "spectralLayers_v2.3"
notes = "Spectral Layering Analysis // SNR Updated"

"""
This script processes hyperspectral images of ice and compares spectral sums across rows.

Version 2.3:
- Added watershed segmentation for bubble detection.
- Outputs layer csvs and images.
- Commented and has 'helper functions'
- Added Batch Log
- Added PCA sums: V2.3
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
    from Sums import sums_process_module as processSums
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
    dataPath, top_depth, bottom_depth, core_length, save_path, px_dims, metadata= processice.getDataPath(datasetPath, DataID, outputDir)
    outputPath = processice.make_dir(save_path, SCRIPT_ID, metadata, f"{DataID}-{batch_ts}", notes, batch_id)

    # Load ENVI hyperspectral data
    data_nparr, w_ref, d_ref, wvls, fwhm = processice.getEnviData(dataPath)

    # Save raw RGB composite
    save_rgb_image(data_nparr, outputPath, f"{DataID}raw.png", DataID, "RGB Image of Raw Data")

    # ───────────────────────────────────────────────
    # 4. IMAGE PRE-PROCESSING (Trim → Reflectance → Filter)
    # ───────────────────────────────────────────────
    segment_boundaries, px_per_cm = processbubbles.get_segments(data_nparr, top_depth, bottom_depth, end_px=px_dims[1])
    stdSize = [530, 820, 200, 900]  # Standard crop dimensions
    trimmedData = processice.trim(data_nparr, w_ref, d_ref, px_dims, stdSize, Debug=Debug)

    # Save trimmed RGB image
    save_rgb_image(trimmedData[0], outputPath, f"{DataID}trimmed.png", DataID, "RGB Image of Trimmed Raw Data")

    # Reflectance correction and band filtering
    correctedData = processice.reflectance(trimmedData, Debug)
    filtered_img, wvls, fwhm = processice.remove_bands(correctedData, trimmedData[1], wvls, fwhm, Debug=False, outputPath=outputPath, save=True)
    save_rgb_image(filtered_img, outputPath, f"{DataID}filtered.png", DataID, "Filtered Data (Post Reflectance Correction)")

    # ───────────────────────────────────────────────
    # 5. BUBBLE REMOVAL & LAYER PROCESSING
    # ───────────────────────────────────────────────
    inpainted_img = processice.bubbles(filtered_img,  fwhm)
    
    import copy 
    inpaint_copy = copy.deepcopy(inpainted_img) 
    pc_image = processice.pc_infill(inpaint_copy, num_pcs=5, fill_mode="tile")
    size = pc_image.shape[1]
    start = max((size - 50) // 2, 0)
    end = start + 50
    pc_image = np.take(pc_image, range(start, min(end, size)), axis=1) # just middle 50 px
    save_rgb_image(pc_image, outputPath, f"{DataID}PCA_Image.png", DataID, "RGB Image of PCA Data")

    sums_img, full_img_sums = processSums.channel_sums(pc_image, px_per_cm, top_depth, Debug, True, outputPath, True)
    combined_df, per_section = processSums.sum_comparison(pc_image, px_per_cm, top_depth, Debug, True, outputPath, True)
    
    sums_img, full_img_sums = processSums.channel_sums(inpainted_img, px_per_cm, top_depth, Debug, True, outputPath)
    combined_df, per_section = processSums.sum_comparison(inpainted_img, px_per_cm, top_depth, Debug, True, outputPath)

    # ───────────────────────────────────────────────
    # 6. README & BATCH LOG
    # ───────────────────────────────────────────────
    readme_path = outputPath / "README.md"
    
    # Ensure README exists, then append a simple provenance block
    try:
        if not readme_path.exists():
            readme_path.write_text("# Processing Report\n", encoding="utf-8")
    
        with readme_path.open("a", encoding="utf-8") as f:
            f.write("\n## Batch Provenance\n")
            f.write(f"- ScriptID: {script_id}\n")
            f.write(f"- BatchID: {batch_id}\n")
            f.write(f"- BatchTimestamp: {batch_ts}\n")
            f.write(f"- DataID: {DataID}\n")
    except Exception as e:
        # don’t crash the job over README write issues
        print(f"[warn] README write failed: {e}")
    
    # Hints for batch_runner to capture into CSV
    print(f"output_dir: {outputPath}")
    print(f"readme: {readme_path}")

# ───────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────
def save_rgb_image(img, output_path, filename, data_id, description):
    """Save an RGB slice of hyperspectral data and log it in README."""
    n_bands = img.shape[2]
    if n_bands >= 97:
        band_indices = [96, 59, 16]
    else:
        # Pick 3 roughly evenly spaced bands
        band_indices = [max(0, n_bands - 1), n_bands // 2, 0]
        band_indices = sorted(set(band_indices), reverse=True)[:3]
    rgb_slice = img[:, :, band_indices].astype(np.float32)
    rgb_norm = (rgb_slice - rgb_slice.min()) / (rgb_slice.max() - rgb_slice.min())
    plt.imsave(f"{output_path}/{filename}", rgb_norm)
    append_readme(output_path, filename, f"{data_id}: {description}")


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
    elif 751 <= NumID < 873:
        datasetPath = '/data/icecores/HSI_Jan2024.csv'
    else:
        raise ValueError(f"NumID {NumID} is outside supported ranges (<=228, 229–334, 335–362, 363–585, 586–750, 751–872). \n Please check DataID and try again.")
    return datasetPath

if __name__ == "__main__":
    main()