# rumz 5/6/24 // 6/7/24 retry
# https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html

### Feb 2025
### NOT IN USE

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import skimage.io
from skimage import morphology
from skimage.color import rgb2gray

def process(eq_img):
    image = eq_img
    image = image[:,:,0]
    sq = morphology.disk(6)
    eroded_img = morphology.erosion(image, sq)
    fig, axes = plt.subplots(1, 2)

    # Plot images on subplots
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(eroded_img, cmap='gray')
    axes[1].set_title('Erosion')
    axes[1].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    dilated_img = morphology.dilation(image, sq)
    fig, axes = plt.subplots(1, 2)

    # Plot images on subplots
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(dilated_img, cmap='gray')
    axes[1].set_title('Dilation')
    axes[1].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    opened_img = morphology.opening(image, sq)
    fig, axes = plt.subplots(1, 2)

    # Plot images on subplots
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(opened_img, cmap='gray')
    axes[1].set_title('Opening')
    axes[1].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    closed_img = morphology.closing(image, sq)
    fig, axes = plt.subplots(1, 2)

    # Plot images on subplots
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(closed_img, cmap='gray')
    axes[1].set_title('Closing')
    axes[1].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()


    return eroded_img