#!/usr/bin/env python3

# ───────────────────────────────────────────────
# Script ID   : spectralHDR_v1
# Description : Full pipeline for creating High Dynamic Range Images of core sections in HSI_2024.csv
# Author      : Roisin Rumsey
# Date        : 2025-06-4
# Notes       : Outputs HDR images as .pngs
# ───────────────────────────────────────────────

SCRIPT_ID = "spectralHDR_v1"
notes = "High Dynamic Range Image creation"

"""
This script processes hyperspectral images of ice and creates HDR images.

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
    import set_icespec_path as tool
    import set_bubbles_path as tool2
    icespec_code_path=tool.add_icespec_code_dir()
    bubble_code_path=tool2.add_bubble_code_dir()
    tool.add_icespec_code_dir()
    tool2.add_bubble_code_dir()
    from icespec import icespec_process2_module as processice
    from bubbles import bubbles_process_module as processbubbles
    from HDR import hdr_processing_module as processhdr
    from icespec import savePNG
    from icespec import ageConversion
    from HDR import hdr_csv_capture_module as cap

    datasetPath = '/data/icecores/HSI_2024.csv' #path to csv file on computer or server
    outputDir = '/data/icecores/Output' #png output path
    
    CoreID = sys.argv[1]
    SectionID = sys.argv[2]
    batch_id = sys.argv[3] if len(sys.argv) > 3 else "unbatched"
    Save = True
    Debug = False

    dataList = processhdr.getDataPath(CoreID, SectionID, datasetPath, outputDir, Debug)
    outputPath = processhdr.make_dir(dataList, SCRIPT_ID)

    imgs = processhdr.getEnviData(dataList)
    trimmedImgs = processhdr.trim(imgs, dataList)
    correctedImgs = processhdr.reflectance(trimmedImgs, Debug)

    combined_img = processhdr.combine_imgs(correctedImgs, Debug, save=True, savepath=outputPath)

    filtered_img, fwhm = processice.remove_bands(combined_img, trimmedImgs[1][1], imgs[2][3])
    inpainted_img = processice.bubbles(filtered_img, imgs[2][3])
    flattened_img = processice.flatten_data(inpainted_img)
    rematrix_img = processice.resize_matrix(flattened_img, trimmedImgs[1], pc_num = 3) #pc_num can be changed
    eq_img = processice.equalize(rematrix_img, save=True, savePath=outputPath)
    DSName = outputPath.parent.name
    filename = f"{DSName}_equalizedHDR.png"
    eq_img = processice.equalize(rematrix_img, save=True, savePath=f"{outputPath}/{filename}")
    readme_path = outputPath / "README.md"
    with readme_path.open("a") as f:
        f.write(f"\n###  Output File: {filename}\n- {DSName}: HDR image equalized\n")

if __name__ == "__main__":
    main()