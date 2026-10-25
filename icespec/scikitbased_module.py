"""This module is based on scikit-image. 

"""
#import libraries

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import img_as_float
from skimage import exposure

def norm_imgfloat(image_data):
    """converts np.array image (lines, samples, 3 -bands) to RGB format float
    prints displayed image
    returns image (lines, samples, 3 -bands) with float values in a range from 0 to 1
    """
    # Contrast stretching
    p2, p98 = np.percentile(image_data, (2, 98))
    img_rescale = exposure.rescale_intensity(image_data, in_range=(p2, p98))
    # Equalization
    img_eq = exposure.equalize_hist(image_data)
    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(image_data, clip_limit=0.03)
    return img_rescale, img_eq, img_adapteq


matplotlib.rcParams['font.size'] = 8

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.
https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

'''
layer detection based on Hough transform 
recommended by Daniel J. Breton, 14 JUN 2021

See for more details:
https://scikit-image.org/docs/stable/auto_examples/edges/plot_line_hough_transform.html
'''
from skimage.transform import probabilistic_hough_line
from matplotlib import cm

def find_layers(image,threshold=10,line_length=5,line_gap=10):
    """
    input single channel image (lines, samples, band) with float values in a range from 0 to 1
    """

    # reading image
    #fname = 'UXM-Hough.png'
    #image = imread(fname, as_gray=True)
    
    # invert image
    image = 1-image 

    lines = probabilistic_hough_line(image,threshold=threshold,line_length=line_length,line_gap=line_gap)
    return lines
