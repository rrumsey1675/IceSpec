# 3/8/24 rumz
# this code is based off the cheese notebook from Mikhail Zhizhin
#this file is currently hardcoded with data from ICF_SPC14-1317_0006
#called by hsi_process1_module for bubble detection

#import packages
import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import scipy
from skimage import io
from skimage import img_as_uint
from astropy.convolution import convolve
from astropy.visualization import simple_norm
from photutils.background import Background2D, MedianBackground
from photutils.datasets import make_100gaussians_image
from photutils.segmentation import (SourceFinder, make_2dgaussian_kernel)

#display image function
def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 28))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

#Zhu Li do the thing!
def cheese(path):
    image = io.imread(path) 
    #image_show(image)
    image_gray = color.rgb2gray(image) 
    #image_show(image_gray);

    # make a simulated image
    data = image_gray

    #mess with treshold, finder, kernel, and Background

    # subtract the background
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                    bkg_estimator=bkg_estimator)
    data -= bkg.background

    # convolve the data
    kernel = make_2dgaussian_kernel(3., size=5)  # FWHM = 3.
    convolved_data = convolve(data, kernel)

    # detect the sources
    threshold = 1.5 * bkg.background_rms  # per-pixel threshold
    finder = SourceFinder(npixels=10, progress_bar=True)
    segm = finder(convolved_data, threshold)

    # plot the image and the segmentation image
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(28, 12))
    norm = simple_norm(data, 'sqrt', percent=99.)
    ax1.imshow(data, origin='lower', interpolation='nearest',
            norm=norm)
    ax2.imshow(segm.data, origin='lower', interpolation='nearest',
            cmap=segm.cmap)
    plt.tight_layout()
    #plt.savefig('SPC14-1317_2023-01-06_21-05-33_bubbles.png',bbox_inches='tight')

    segm_mask = segm.data > 0
    print(segm_mask.shape)

    image_bw = img_as_uint(segm_mask)

    io.imsave('/Users/r.rumsey/src/rumzData/ICF_SPC14-1317_0006/bubbles_mask.png',image_bw)

    mask = '/Users/r.rumsey/src/rumzData/ICF_SPC14-1317_0006/bubbles_mask.png'

    return mask
