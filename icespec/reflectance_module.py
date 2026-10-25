#rumz 5/8/24
# X = (S-B)/(W-B)
# https://medium.com/@achyutpaudel50/hyperspectral-image-processing-in-python-custom-roi-selection-with-mouse-78fbaf7520aa

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import spectral
spectral.settings.envi_support_nonlowercase_params = True
import spectral.io.envi as envi
import csv

# corrected_data = np.divide(
#             np.subtract(img_data, d_ref),
#             np.subtract(w_ref, d_ref))

# add a col size check

def correction(trimmedData: list, Debug: bool):

    img_data = trimmedData[0]
    w_ref = trimmedData[1]
    d_ref = trimmedData[2]

    rows = img_data.shape[0] #lines scanned
    samples = img_data.shape[1] #samples 1-1024 max 
    bands = img_data.shape[2] # also channels 1-224

    #Calculate mean value per channel for dark reference
    d_ref_mean=d_ref.mean(axis=0)
    #Calculate mean value per channel for white reference
    w_ref_mean=w_ref.mean(axis=0)

    # Define text file names
    dr_csv='./refTestData/darkRef.csv'
    #dr_csv='/Users/r.rumsey/src/refTestData/darkRef.csv'
    wr_csv='./refTestData/whiteRef.csv'
    #wr_csv='/Users/r.rumsey/src/refTestData/whiteRef.csv'
    before_ch1_csv='./refTestData/before1Ref.csv'
    #before_ch1_csv='/Users/r.rumsey/src/refTestData/before1Ref.csv'
    before_ch100_csv='./refTestData/before2Ref.csv'
    #before_ch100_csv='/Users/r.rumsey/src/refTestData/before2Ref.csv'
    before_ch224_csv='./refTestData/before3Ref.csv'
    #before_ch224_csv='/Users/r.rumsey/src/refTestData/before3Ref.csv'
    after_ch1_csv='./refTestData/after1Ref.csv'
    #after_ch1_csv='/Users/r.rumsey/src/refTestData/after1Ref.csv'
    after_ch100_csv='./refTestData/after2Ref.csv'
    #after_ch100_csv='/Users/r.rumsey/src/refTestData/after2Ref.csv'
    after_ch224_csv='./refTestData/after3Ref.csv'
    #after_ch224_csv='/Users/r.rumsey/src/refTestData/after3Ref.csv'

    if(Debug):
        d_ref_mean[250:350,:].tofile(dr_csv, sep= ',')


    if(Debug):
        w_ref_mean[250:350,:].tofile(wr_csv, sep= ',')

    denominator = w_ref_mean - d_ref_mean
    print(f"W ref: {w_ref_mean.shape}, D ref: {d_ref_mean.shape}, Denom:  {denominator.shape}")

    before1Arr = img_data[50,250:350,0]
    before2Arr = img_data[50,250:350,99]
    before3Arr = img_data[50,250:350,223]
    
    ref_img = np.empty((bands, rows, samples))

    # for x in range(samples):
    #     for y in range(rows):
    #         for z in range(bands):
    #             ref_img[y,x,z] = ((img_data[y,x,z] - d_ref_mean[x,z])/denominator[x,z])
    
    #Transpose data to speed up operations
    img_data=np.transpose(img_data, (2, 0, 1))
    i=0
    for im_ch in img_data:
        print(f"Channel: {i}")
        ref_img[i]=(im_ch-d_ref_mean[:,i])/denominator[:,i]
        i+=1


    print("Reflectance image transposed: ", ref_img.shape)
    #Transpose image back to the original dimentions
    ref_img=np.transpose(ref_img, (1, 2, 0))

    after1Arr = ref_img[50,250:350,0]
    after2Arr = ref_img[50,250:350,99]
    after3Arr = ref_img[50,250:350,223]

    if(Debug):
        before1Arr.tofile(before_ch1_csv, sep= ',')
        before2Arr.tofile(before_ch100_csv, sep= ',')
        before3Arr.tofile(before_ch224_csv, sep= ',')
        after1Arr.tofile(after_ch1_csv, sep= ',')
        after2Arr.tofile(after_ch100_csv, sep= ',')
        after3Arr.tofile(after_ch224_csv, sep= ',')

   
    
    return ref_img
