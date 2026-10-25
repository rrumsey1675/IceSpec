#rumz // 6/7/24

import matplotlib.pyplot as plt
import spectral

def data_preview(og_img, corrected_img):
    """
    Displays an gray preview of hyperspectral images at different bands, comparing reflecatnace before and after.

    Args:
        og_img (np.ndarray): Hyperspectral image before ref correction.
        corrected_img (np.ndarray): Hyperspectral image after ref correction.
        

    Returns:
        None
    """

    # image1 = spectral.imshow(og_img, (2, 1, 0), stretch_all = True, title = "Low-range, no correction")
    # image2 = spectral.imshow(og_img, (124, 123, 122), stretch_all = True, title = "Mid-range, no correction")
    # image3 = spectral.imshow(og_img, (217, 216, 215), stretch_all = False, title = "Hi-range, no correction")

    # image4 = spectral.imshow(corrected_img, (2, 1, 0), stretch_all = True, title = "Low-range, w/ correction")
    # image5 = spectral.imshow(corrected_img, (124, 123, 122), stretch_all = True, title = "Mid-range, w/ correction")
    # image6 = spectral.imshow(corrected_img, (217, 216, 215), stretch_all = False, title = "Hi-range, w/ correction")

    image1 = og_img[:,:,0]
    image2 = og_img[:,:,125]
    image3 = og_img[:,:,220]

    image4 = corrected_img[:,:,0]
    image5 = corrected_img[:,:,125]
    image6 = corrected_img[:,:,220]

    fig, axes = plt.subplots(2, 3)

    # Plot images on subplots
    axes[0,0].imshow(image1, cmap='gray')
    axes[0,0].set_title('Band: 0, no correction')
    axes[0,0].axis('off')

    axes[0,1].imshow(image2, cmap='gray')
    axes[0,1].set_title('Band: 125, no correction')
    axes[0,1].axis('off')

    axes[0,2].imshow(image3, cmap='gray')
    axes[0,2].set_title('Band: 220, no correction')
    axes[0,2].axis('off')

    axes[1,0].imshow(image4, cmap='gray')
    axes[1,0].set_title('Band: 0, w/ correction')
    axes[1,0].axis('off')

    axes[1,1].imshow(image5, cmap='gray')
    axes[1,1].set_title('Band: 125, w/ correction')
    axes[1,1].axis('off')

    axes[1,2].imshow(image6, cmap='gray')
    axes[1,2].set_title('Band: 220, w/ correction')
    axes[1,2].axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.show()