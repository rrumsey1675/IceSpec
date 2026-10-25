# rumz 5/8/24
# X = (S-B)/(W-B)
# https://medium.com/@achyutpaudel50/hyperspectral-image-processing-in-python-custom-roi-selection-with-mouse-78fbaf7520aa

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# corrected_data = np.divide(
#             np.subtract(img_data, d_ref),
#             np.subtract(w_ref, d_ref))

def correction(img_data, w_ref, d_ref):

    columns = img_data.shape[1]
    rows = img_data.shape[0]
    d_av = np.average(d_ref)

    col_arr = []

    for i in range(columns):
        col_arr.append(np.average(w_ref[:,i]))
        i += 1

    for x in range(columns-1):
        for y in range(rows-1):
            img_data[x,y] = (img_data[x,y] - d_av)/(col_arr[x] - d_av)

    

    return img_data