#rumz 6/7/24
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

def output(image1, image2, image3, image4, image5, save=False, savePath=""):
    """
    Plot 5 enhanced images side by side for comparison.

    Args:
        image1 (np.ndarray): Contrast Stretching Image (RGB)
        image2 (np.ndarray): Histogram Equalization Image (RGB)
        image3 (np.ndarray): Adaptive Equalization Image (RGB)
        image4 (np.ndarray): Local Equalization Image (grayscale)
        image5 (np.ndarray): Logarithmic Corrected Image (RGB)
    """
    # Ensure all inputs are float for proper display
    from skimage.color import rgb2gray
    
    # Convert RGB images to grayscale safely
    def to_grayscale(img):
        return rgb2gray(img) if img.ndim == 3 else img
    
    image1_gray = to_grayscale(image1)
    image2_gray = to_grayscale(image2)
    image3_gray = to_grayscale(image3)
    image5_gray = to_grayscale(image5)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 10))
    
    axes[0].imshow(image1_gray, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Contrast Stretching')
    axes[0].axis('off')
    
    axes[1].imshow(image2_gray, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Histogram Equalization')
    axes[1].axis('off')
    
    axes[2].imshow(image3_gray, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Adaptive Equalization')
    axes[2].axis('off')
    
    axes[3].imshow(image4, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('Local Equalization (Grayscale)')
    axes[3].axis('off')
    
    axes[4].imshow(image5_gray, cmap='gray', vmin=0, vmax=1)
    axes[4].set_title('Logarithmic Correction')
    axes[4].axis('off')
    
    plt.tight_layout()
    if save == True:
        plt.savefig(f"{savePath}_equalized.png", dpi=300, bbox_inches='tight')
    plt.show()

def remove_column_striping(image: np.ndarray) -> np.ndarray:
    corrected = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        channel = image[:, :, i].astype(np.float32)
        column_medians = np.median(channel, axis=0)
        channel -= column_medians
        channel -= channel.min()
        channel /= channel.max()
        corrected[:, :, i] = channel
    return corrected


def clip_intensity(img: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    img_out = np.zeros_like(img)
    for c in range(3):
        p_low, p_high = np.percentile(img[:, :, c], (low, high))
        img_c = np.clip(img[:, :, c], p_low, p_high)
        img_c = (img_c - p_low) / (p_high - p_low + 1e-8)
        img_out[:, :, c] = img_c
    return img_out

def compare_disk_sizes(image: np.ndarray, radii: list[int] = [5, 15, 30, 50]) -> None:
    """
    Compares local histogram equalization using different disk sizes.

    Args:
        image (np.ndarray): Input RGB image
        radii (list[int]): List of disk radii to test
    """
    gray = rgb2gray(image)
    gray_ubyte = img_as_ubyte(gray)

    fig, axes = plt.subplots(1, len(radii), figsize=(4 * len(radii), 8))
    for ax, r in zip(axes, radii):
        neighborhood = disk(r)
        img_eq = rank.equalize(gray_ubyte, footprint=neighborhood)
        ax.imshow(img_eq, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'disk({r})')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img -= img.min()
    img /= (img.max() + 1e-8)
    return img

def analyze_data(image: np.ndarray, save, savePath) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Step 1: Remove vertical striping
    img_cleaned = remove_column_striping(image)
    # img_cleaned = image

    # Step 2: Contrast Stretching
    p2, p98 = np.percentile(img_cleaned, (2, 98))
    img_rescale = exposure.rescale_intensity(img_cleaned, in_range=(p2, p98))

    #compare_disk_sizes(img_rescale)

    # Step 3: Histogram Equalization (per channel)
    img_eq_r = exposure.equalize_hist(img_rescale[:, :, 0])
    img_eq_g = exposure.equalize_hist(img_rescale[:, :, 1])
    img_eq_b = exposure.equalize_hist(img_rescale[:, :, 2])
    img_eq = np.dstack((img_eq_r, img_eq_g, img_eq_b))

    # Step 4: Adaptive Equalization (CLAHE)
    clip_limit = 0.01
    img_r = normalize(img_rescale[:, :, 0])
    img_g = normalize(img_rescale[:, :, 1])
    img_b = normalize(img_rescale[:, :, 2])
    
    img_adapteq_r = exposure.equalize_adapthist(img_r, clip_limit=clip_limit)
    img_adapteq_g = exposure.equalize_adapthist(img_g, clip_limit=clip_limit)
    img_adapteq_b = exposure.equalize_adapthist(img_b, clip_limit=clip_limit)
    
    img_adapteq = np.dstack((img_adapteq_r, img_adapteq_g, img_adapteq_b))

    # Step 5: Local Equalization (grayscale)
    neighborhood = disk(30)
    gray_for_local = rgb2gray(img_rescale)
    gray_for_local = normalize(gray_for_local)
    gray_ubyte = img_as_ubyte(gray_for_local)
    img_localeq = rank.equalize(gray_ubyte, footprint=neighborhood)
    img_localeq = img_as_float(img_localeq)

    # Step 6: Logarithmic Correction
    logarithmic_corrected = exposure.adjust_log(img_adapteq)

    # Step 7: Clip visual intensity to avoid overblown colors
    img_rescale_clip = clip_intensity(img_rescale)
    img_eq_clip = clip_intensity(img_eq)
    img_adapteq_clip = clip_intensity(img_adapteq)
    log_corrected_clip = clip_intensity(logarithmic_corrected)

    # Step 8: Output the visual comparison
    output(img_rescale_clip, img_eq_clip, img_adapteq_clip, img_localeq, log_corrected_clip, save, savePath)

    return img_adapteq_clip, img_localeq