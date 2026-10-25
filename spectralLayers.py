#!/usr/bin/env python3

"""
This script processes hyperspectral images of ice and extracts information in regards to the bubbles.

V. 1.1.4
"""

# Importing Python Libraries
import warnings
import sys
import os

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
from icespec import savePNG

datasetPath = '/data/icecores/HSI_2024.csv' #path to csv file on computer or server
outputDir = '/data/icecores/Output' #png output path
DataID = sys.argv[1]
Debug = False

dataPath, top_depth, bottom_depth, core_length, save_path, px_dims = processice.getDataPath(datasetPath, DataID, outputDir)
outputPath = f"{save_path}/"
data_nparr, w_ref, d_ref, fwhm = processice.getEnviData(dataPath)

dataSize = px_dims
stdSize = [530, 820, 200, 900] #standard sizing

trimmedData = processice.trim(data_nparr, w_ref, d_ref, dataSize, stdSize, Debug=True) #return arr of 3 trimmed
left, middle, right = processice.extract_sections(trimmedData)
left_correctedData = processice.reflectance(left, Debug)
mid_correctedData = processice.reflectance(middle, Debug)
right_correctedData = processice.reflectance(right, Debug)

left_filtered_img = processice.remove_bands(left_correctedData, trimmedData[1])
mid_filtered_img = processice.remove_bands(mid_correctedData, trimmedData[1])
right_filtered_img = processice.remove_bands(right_correctedData, trimmedData[1])

left_inpainted_img = processice.bubbles(left_filtered_img,  fwhm)
mid_inpainted_img = processice.bubbles(mid_filtered_img,  fwhm)
right_inpainted_img = processice.bubbles(right_filtered_img,  fwhm)

left_flattened_img = processice.flatten_data(left_inpainted_img)
mid_flattened_img = processice.flatten_data(mid_inpainted_img)
right_flattened_img = processice.flatten_data(right_inpainted_img)

left_rematrix_img = processice.resize_matrix(left_flattened_img, left, pc_num = 5) #pc_num can be changed
mid_rematrix_img = processice.resize_matrix(mid_flattened_img, middle, pc_num = 5) #pc_num can be changed
right_rematrix_img = processice.resize_matrix(right_flattened_img, right, pc_num = 5) #pc_num can be changed

rematrix_img = np.concatenate((left_rematrix_img, mid_rematrix_img, right_rematrix_img), axis=1)
eq_img = processice.equalize(rematrix_img)

start_cm = int(top_depth * 100)
depths_cm = np.arange(start_cm, start_cm + len(segment_boundaries))
depth_col = "Depth"
age_col = "Ice Age"
header = 40
age_df = ageConversion.map_depths_to_ages("/data/icecores/ICD/WDC06A/WD2014_26Aug2014.xlsx", depths_cm, top_depth, depth_col, age_col, header)
age_df.to_csv(f"{outputPath}{DataID}_ages_in_cm.csv", index=False)