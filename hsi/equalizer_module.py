#based on scikit example
#https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
# https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_local_equalize.html 

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from skimage import data, img_as_float
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage import exposure
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import rank



matplotlib.rcParams['font.size'] = 8

def output(image1, image2, image3, output_path):
    imagefloat1 = img_as_float(image1)
    imagefloat2 = img_as_float(image2)
    imagefloat3 = img_as_float(image3)
    # plt.title("Histogram Equalization")
    # plt.imshow(imagefloat, cmap=plt.cm.gray)

    fig, axes = plt.subplots(1, 3)

    # Plot images on subplots
    axes[0].imshow(imagefloat1, cmap='gray')
    axes[0].set_title('Histogram Equalization')
    axes[0].axis('off')

    axes[1].imshow(imagefloat3, cmap='gray')
    axes[1].set_title('Adaptive Equalization')
    axes[1].axis('off')

    axes[2].imshow(imagefloat2, cmap='gray')
    axes[2].set_title('Local Equalization')
    axes[2].axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path+"/equalizedImage.jpg", bbox_inches ="tight")
    #plt.show()
    
    return imagefloat3


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

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


def analyze_data(image, output_path):
    #Images of type float must be between -1 and 1.
    #color bands
    # red = 96
    # green=59
    # blue=16
    red = 96
    green=59
    blue=180

    # Load image sent when calling function
    img = image[:,:, 0]
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))


    # Equalization
    img_eq = exposure.equalize_hist(img)

    #Local Equalization
    neighborhood = disk(30)
    #img_eq = rank.equalize(imgfloat, footprint=neighborhood)


    #RED / index 9
    img_rescaleMIN_MAX_r = exposure.rescale_intensity(image[:,:,0], in_range='image') # use min and max from image 
    #img_adapteq = exposure.equalize_adapthist(hsi_data[:,:,red], clip_limit=0.03)
    img_localeq_r = rank.equalize(img_rescaleMIN_MAX_r, footprint=neighborhood)
    img_adapteq_r = exposure.equalize_adapthist(img_rescaleMIN_MAX_r, clip_limit=0.03)

    #GREEN
    img_rescaleMIN_MAX_g = exposure.rescale_intensity(image[:,:,1], in_range='image') # use min and max from image 
    #img_adapteq = exposure.equalize_adapthist(hsi_data[:,:,red], clip_limit=0.03)
    img_localeq_g = rank.equalize(img_rescaleMIN_MAX_g, footprint=neighborhood)
    img_adapteq_g = exposure.equalize_adapthist(img_rescaleMIN_MAX_g, clip_limit=0.03)

    #BLUE
    img_rescaleMIN_MAX_b = exposure.rescale_intensity(image[:,:,2], in_range='image') # use min and max from image 
    #img_adapteq = exposure.equalize_adapthist(hsi_data[:,:,red], clip_limit=0.03)
    img_localeq_b = rank.equalize(img_rescaleMIN_MAX_b, footprint=neighborhood)
    img_adapteq_b = exposure.equalize_adapthist(img_rescaleMIN_MAX_b, clip_limit=0.03)

    #Combine into RGB aray
    img_adapteq = np.dstack((img_adapteq_r,img_adapteq_g,img_adapteq_b))
    img_localeq = np.dstack((img_localeq_r,img_localeq_g,img_localeq_b))
    img_localeq = rgb2gray(img_localeq)
    
    logarithmic_corrected = exposure.adjust_log(img_adapteq, 1)

    #gray scale
    # Adaptive Equalization
    #img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    # Display results
    # fig = plt.figure(figsize=(24, 15))
    # spec = gridspec.GridSpec(ncols=4, nrows=2,
    #                      width_ratios=[1, 1, 1, 1], wspace=0.5,
    #                      hspace=0.5, height_ratios=[3, 1])
    
    # axes = np.zeros((2, 4), dtype=object)
    # axes[0, 0] = fig.add_subplot(spec[0])
    # for i in range(1, 4):
    #     axes[0, i] = fig.add_subplot(spec[0+i], sharex=axes[0,0], sharey=axes[0,0])
    # for i in range(0, 4):
    #     axes[1, i] = fig.add_subplot(spec[4+i])

    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    # ax_img.set_title('Low contrast image')

    # y_min, y_max = ax_hist.get_ylim()
    # ax_hist.set_ylabel('Number of pixels')
    # ax_hist.set_yticks(np.linspace(0, y_max, 5))

    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
    # ax_img.set_title('Contrast stretching')

    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
    # ax_img.set_title('Histogram equalization')

    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
    # ax_img.set_title('Adaptive equalization')

    # ax_cdf.set_ylabel('Fraction of total intensity')
    # ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # # prevent overlap of y-axis labels
    # fig.tight_layout()
    #plt.show()
    plt.clf()

    op = output(img_eq, img_localeq, logarithmic_corrected, output_path)

    return op