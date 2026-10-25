#!/usr/bin/env python3

# ───────────────────────────────────────────────
# Script ID   : spectralSums_v1
# Description : Full pipeline for trimmed, filtered HSI image spectral sum comparison.
# Author      : Roisin Rumsey
# Date        : 2025-05-22
# Notes       : Outputs plt figs and sums in csv files
# ───────────────────────────────────────────────

SCRIPT_ID = "spectralSums_v1"
notes = "Spectral Sum Analysis"

"""
This script processes hyperspectral images of ice and compares spectral sums across rows.

V. 1
"""

# Importing Python Libraries
import warnings
import sys
import os

def main():
    # Add the processing folder to the Python path
    processing_path = os.path.abspath(os.path.join('processing'))
    if processing_path not in sys.path:
        sys.path.append(processing_path)
    
    #Importing Python Libraries and Packages
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import set_icespec_path as tool
    import set_bubbles_path as tool2
    icespec_code_path=tool.add_icespec_code_dir()
    bubble_code_path=tool2.add_bubble_code_dir()
    tool.add_icespec_code_dir()
    tool2.add_bubble_code_dir()
    from icespec import icespec_process2_module as processice
    from bubbles import bubbles_process_module as processbubbles
    from Sums import sums_process_module as processSums
    from icespec import savePNG
    from icespec import ageConversion
    
    DataID = sys.argv[1]
    NumID = int(DataID.split('-')[-1])

    if NumID > 228:
        datasetPath = '/data/icecores/HSI_Jan2024.csv' #path to csv file on computer or server
    else:
        datasetPath = '/data/icecores/HSI_2024.csv'
    
    outputDir = '/data/icecores/Output' #png output path
    batch_id = sys.argv[2] if len(sys.argv) > 2 else "unbatched"
        
    
    Debug = False
    Save = True
    
    dataPath, top_depth, bottom_depth, core_length, save_path, px_dims, metadata = processice.getDataPath(datasetPath, DataID, outputDir)
    outputPath = processice.make_dir(save_path, SCRIPT_ID, metadata, notes=notes, batch_id=batch_id)
    outputPath = outputPath
    print("Output Path: ", outputPath)
    data_nparr, w_ref, d_ref, fwhm = processice.getEnviData(dataPath)
    filename = f"{DataID}raw.png"
    rgb = data_nparr
    rgb_slice = rgb[:, :, [96, 59, 16]].astype(np.float32)
    rgb_norm = (rgb_slice - rgb_slice.min()) / (rgb_slice.max() - rgb_slice.min())
    plt.imsave(f"{outputPath}/{filename}", rgb_norm)
    readme_path = outputPath / "README.md"
    with readme_path.open("a") as f:
        f.write(f"\n###  Output File: {filename}\n- {DataID}: RGB Image of Raw Data\n")

if __name__ == "__main__":
    main()