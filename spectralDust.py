#!/usr/bin/env python3

# ───────────────────────────────────────────────
# Script ID   : spectralDust_v1
# Description : Full pipeline for trimmed, filtered HSI image spectral dust comparison.
# Author      : Roisin Rumsey
# Date        : 2026-1-17
# Notes       : Outputs plt figs and dust in csv files
# ───────────────────────────────────────────────

SCRIPT_ID = "spectralDust_v1"
notes = "Spectral Dust Layering Analysis"

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
from sklearn.decomposition import PCA


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

    # ───────────────────────────────────────────────
    # 5. BUBBLE REMOVAL & LAYER PROCESSING
    # ───────────────────────────────────────────────
    inpainted_img = processice.bubbles(correctedData,  fwhm)
    
    import copy 
    inpaint_copy = copy.deepcopy(inpainted_img) 

    violet = inpaint_copy[:, :, 9] 
    infrared = inpaint_copy[:, :, 215] 
    green = inpaint_copy[:, :, 110] 

    DSName = outputPath.parent.name

    # --- Violet ---
    height, _ = violet.shape
    depths_cm = top_depth * 100 + (np.arange(height) / px_per_cm)
    row_sums = np.sum(violet, axis=1)
    row_mean = np.mean(violet, axis=1)
    row_median = np.median(violet, axis=1)
    data = {"Depth_cm": depths_cm, "Sums": row_sums, "Means": row_mean, "Medians": row_median}
    violet_df = pd.DataFrame(data)
    violet_df = violet_df[(violet_df.select_dtypes(include='number') >= 0).all(axis=1)]
    save_dataframe(violet_df, outputPath, f"{DSName}violet_df.csv", "Violet wavelength data")

    # --- Infrared ---
    height, _ = infrared.shape
    depths_cm = top_depth * 100 + (np.arange(height) / px_per_cm)
    row_sums = np.sum(infrared, axis=1)
    row_mean = np.mean(infrared, axis=1)
    row_median = np.median(infrared, axis=1)
    data = {"Depth_cm": depths_cm, "Sums": row_sums, "Means": row_mean, "Medians": row_median}
    infrared_df = pd.DataFrame(data)
    infrared_df = infrared_df[(infrared_df.select_dtypes(include='number') >= 0).all(axis=1)]
    save_dataframe(infrared_df, outputPath, f"{DSName}infrared_df.csv", "infrared wavelength data")

    # --- Green ---
    height, _ = green.shape
    depths_cm = top_depth * 100 + (np.arange(height) / px_per_cm)
    row_sums = np.sum(green, axis=1)
    row_mean = np.mean(green, axis=1)
    row_median = np.median(green, axis=1)
    data = {"Depth_cm": depths_cm, "Sums": row_sums, "Means": row_mean, "Medians": row_median}
    green_df = pd.DataFrame(data)
    green_df = green_df[(green_df.select_dtypes(include='number') >= 0).all(axis=1)]
    save_dataframe(green_df, outputPath, f"{DSName}green_df.csv", "green wavelength data")


    # --- Merged ---
    from functools import reduce
    dfs = [violet_df, green_df, infrared_df]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='Depth_cm', how='outer', suffixes=('', '_drop')), dfs) 
    df_merged.columns = [
        'Depth_cm', 
        'Sums_1', 'Means_1', 'Medians_1',
        'Sums_2', 'Means_2', 'Medians_2',
        'Sums_3', 'Means_3', 'Medians_3'
    ]
    df_merged['Diff_1_2_Means'] = df_merged['Means_1'] - df_merged['Means_2']
    df_merged['Diff_1_3_Means'] = df_merged['Means_1'] - df_merged['Means_3']
    df_merged['Diff_2_3_Means'] = df_merged['Means_2'] - df_merged['Means_3']
    df_merged['Means_Spread'] = df_merged[['Means_1', 'Means_2', 'Means_3']].max(axis=1) - \
                                 df_merged[['Means_1', 'Means_2', 'Means_3']].min(axis=1)
    differences_only = df_merged[df_merged['Means_Spread'] > 0]
    df_merged['BG_NDI'] = (df_merged['Means_1'] - df_merged['Means_2'])/(df_merged['Means_1'] + df_merged['Means_2'])
    df_merged['VN_NDI'] = (df_merged['Means_2'] - df_merged['Means_3'])/(df_merged['Means_2'] + df_merged['Means_3'])
    save_dataframe(df_merged, outputPath, f"{DSName}df_merged.csv", "all 3 wavelength data")

    # --- Plots ---
    plt.figure(figsize=(8, 20))
    plt.plot(df_merged['Means_1'], df_merged['Depth_cm'], label='420nm')
    plt.plot(df_merged['Means_2'], df_merged['Depth_cm'], label='520nm')
    plt.plot(df_merged['Means_3'], df_merged['Depth_cm'], label='980nm')
    plt.gca().invert_yaxis()  # Put 0 depth at the top
    plt.xlabel('Mean Value')
    plt.ylabel('Depth (cm)')
    plt.legend()
    plt.title('Comparison of Means across Depths')
    out_path = f"{outputPath}/WVLsMean_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 20))
    plt.plot(df_merged['BG_NDI'], df_merged['Depth_cm'], label='BG_NDI', linestyle='dashed')
    plt.plot(df_merged['VN_NDI'], df_merged['Depth_cm'], label='VN_NDI', linestyle='dashed')
    plt.gca().invert_yaxis()  # Put 0 depth at the top
    plt.xlabel('Mean Value')
    plt.ylabel('Depth (cm)')
    plt.legend()
    plt.title('Comparison of Means across Depths')
    out_path = f"{outputPath}/NDI_SignalComparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.scatter(df_merged['BG_NDI'], df_merged['VN_NDI'])
    plt.xlabel('BG_NDI')
    plt.ylabel('VN_NDI')
    plt.title('NDI Correlation Scatter Plot')
    out_path = f"{outputPath}/NDI_Correlation.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.scatter(df_merged['Means_1'], df_merged['Means_3'])
    plt.title('Means 1,3 Correlation Scatter Plot')
    out_path = f"{outputPath}/meanSignalCorrelation1-3.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.scatter(df_merged['Means_1'], df_merged['Means_2'])
    plt.title('Means 1,2 Correlation Scatter Plot')
    out_path = f"{outputPath}/meanSignalCorrelation1-2.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.scatter(df_merged['Means_2'], df_merged['Means_3'])
    plt.title('Means 2,3 Correlation Scatter Plot')
    out_path = f"{outputPath}/meanSignalCorrelation2-3.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ------ PCA -----
    inpaint_PCAcopy = copy.deepcopy(inpainted_img)
    n_components = 5
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(inpaint_PCAcopy)
    hsi_compressed = pca_results.reshape(H, W, n_components)

    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.plot(pca.components_[i], label=f'PC {i+1} Contributions')
    
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Original Channel (Band) Index')
    plt.ylabel('Loading Weight (Influence)')
    plt.title('Which channels contain the Principal Components?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = f"{outputPath}/PC_signalContribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    pc1 = hsi_compressed[:, :, 0]
    pc2 = hsi_compressed[:, :, 1]
    pc3 = hsi_compressed[:, :, 2]

    #print("PC1 Shape: ", pc1.shape)
    height, _ = pc1.shape
    depths_cm = top_depth * 100 + (np.arange(height) / px_per_cm)
    row_sums = np.sum(pc1, axis=1)
    row_mean = np.mean(pc1, axis=1)
    row_median = np.median(pc1, axis=1)
    # print("PC1 Row Sum Shape: ", row_sums.shape)
    # print("PC1 Row Mean Shape: ", row_mean.shape)
    # print("PC1 Row Median Shape: ", row_median.shape)
    data = {"Depth_cm": depths_cm, "Sums": row_sums, "Means": row_mean, "Medians": row_median}
    pc1_df = pd.DataFrame(data)
    # print("PC1 DF Shape: ", pc1_df.shape)
    # print(pc1_df)
    # pc1_df = pc1_df[(pc1_df.select_dtypes(include='number') >= 0).all(axis=1)]
    # print("PC1 DF Shape: ", pc1_df.shape)

    height, _ = pc2.shape
    depths_cm = top_depth * 100 + (np.arange(height) / px_per_cm)
    row_sums = np.sum(pc2, axis=1)
    row_mean = np.mean(pc2, axis=1)
    row_median = np.median(pc2, axis=1)
    # print("PC2 Row Sum Shape: ", row_sums.shape)
    data = {"Depth_cm": depths_cm, "Sums": row_sums, "Means": row_mean, "Medians": row_median}
    pc2_df = pd.DataFrame(data)

    height, _ = pc3.shape
    depths_cm = top_depth * 100 + (np.arange(height) / px_per_cm) 
    row_sums = np.sum(pc3, axis=1)
    row_mean = np.mean(pc3, axis=1)
    row_median = np.median(pc3, axis=1)     
    data = {"Depth_cm": depths_cm, "Sums": row_sums, "Means": row_mean, "Medians": row_median} 
    pc3_df = pd.DataFrame(data)

    plt.figure(figsize=(8, 20))
    plt.scatter(pc1_df["Means"], pc1_df["Depth_cm"], label='pc1')
    plt.gca().invert_yaxis()  # Put 0 depth at the top
    plt.xlabel('Intensity Value')
    plt.ylabel('Depth (cm)')
    plt.title("Spectral Intensity By Depth: pc1")
    out_path = f"{outputPath}/pc1byDepth.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 20))
    plt.plot(pc2_df["Means"], pc2_df["Depth_cm"], label='pc2')
    plt.gca().invert_yaxis()  # Put 0 depth at the top
    plt.xlabel('Intensity Value')
    plt.ylabel('Depth (cm)')
    plt.title("Spectral Intensity By Depth: pc2")
    out_path = f"{outputPath}/pc2byDepth.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 20))
    plt.plot(pc3_df["Means"], pc3_df["Depth_cm"], label='pc3')
    plt.gca().invert_yaxis()  # Put 0 depth at the top
    plt.xlabel('Intensity Value')
    plt.ylabel('Depth (cm)')
    plt.title("Spectral Intensity By Depth: pc3")
    out_path = f"{outputPath}/pc3byDepth.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    PCAdfs = [pc1_df, pc2_df, pc3_df]

    df_PCAmerged = reduce(lambda left, right: pd.merge(left, right, on='Depth_cm', how='outer', suffixes=('', '_drop')), PCAdfs)
    
    df_PCAmerged.columns = [
        'Depth_cm', 
        'Sums_1', 'Means_1', 'Medians_1',
        'Sums_2', 'Means_2', 'Medians_2',
        'Sums_3', 'Means_3', 'Medians_3'
    ]

    plt.figure(figsize=(8, 20))
    plt.plot(df_PCAmerged['Means_1'], df_PCAmerged['Depth_cm'], label='pc1')
    plt.plot(df_PCAmerged['Means_2'], df_PCAmerged['Depth_cm'], label='pc2')
    plt.plot(df_PCAmerged['Means_3'], df_PCAmerged['Depth_cm'], label='pc3')
    plt.gca().invert_yaxis()  # Put 0 depth at the top
    plt.xlabel('Mean Value')
    plt.ylabel('Depth (cm)')
    plt.legend()
    plt.title('Principal Components of Means by Depth')
    out_path = f"{outputPath}/principalComponentsByDepth.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    save_dataframe(df_PCAmerged, outputPath, f"{DSName}df_PCAmerged.csv", "first 3 Principal Components data")

    

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
    plt.imsave(f"{output_path}/{filename}", rgb_norm, dpi=300)
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
    elif 873 <= NumID < 909:
        datasetPath = '/data/icecores/HSI_Denali2025.csv'
    else:
        raise ValueError(f"NumID {NumID} is outside supported ranges (<=228, 229–334, 335–362, 363–585, 586–750, 751–872). \n Please check DataID and try again.")
    return datasetPath

if __name__ == "__main__":
    main()