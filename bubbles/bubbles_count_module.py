# rumz / Jan 25
# get bubble count

from scipy.ndimage import label
from skimage.measure import regionprops
import numpy as np
import matplotlib.pyplot as plt

def count_bubbles(mask, min_size = 10, Debug=False):
    """
    Counts the number of bubbles in a 2D boolean mask.

    Args:
    - mask (np.ndarray): 2D boolean array where True represents bubbles.
    - min_size (int): Minimum size (in pixels) for bubbles to be counted.
    - Debug (bool): If True, visualizes the labeled mask with a title.

    Returns:
    - bubble_count (int): Number of bubbles detected in the mask.
    """
    labeled_mask, num_features = label(mask)
    properties = regionprops(labeled_mask)
    
    bubble_count = 0
    for region in properties:
        if region.area >= min_size:  # Filter by size
            bubble_count += 1

    if Debug == True:
        visualize_labeled_mask(mask, "Bubble Count")
    
    return num_features

def count_bubbles_per_band(masks, min_size, Debug):
    """
    Counts bubbles in each band's mask and returns the counts.

    Args:
    - masks (list of np.ndarray): List of 2D boolean masks for each band.
    - Debug (bool): If True, displays info to notebook.

    Returns:
    - bubble_counts (list): List of bubble counts for each band.
    """
    bubble_counts = [count_bubbles(mask, min_size, Debug) for mask in masks]
    return bubble_counts

def count_bubbles_in_combined_mask(masks, min_size, Debug, Save=False, savepath=""):
    """
    Counts bubbles in a combined mask across all bands.

    Args:
    - masks (list of np.ndarray): List of 2D boolean masks for each band.
    - Save (bool, optional): If True, saves data to Output folder
    - savepath (str, optional): Path used to save data, Needed if Save is True

    Returns:
    - combined_bubble_count (int): Number of unique bubbles in the combined mask.
    """
    combined_mask = np.any(np.stack(masks, axis=-1), axis=-1)
    visualize_labeled_mask(combined_mask, "Combined Bubble Count", Save, savepath)
    return count_bubbles(combined_mask, min_size, Debug)

def visualize_labeled_mask(mask, title:str, Save=False, savepath=""):
    labeled_mask, _ = label(mask)
    plt.imshow(labeled_mask, cmap='nipy_spectral')
    plt.colorbar(label="Bubble Label")
    plt.title(title)

    if Save == True:
        
        DSName = savepath.parent.name
        filename = f"{DSName}_CombinedBubbleMask.png"
        plt.savefig(f"{savepath}/{filename}", format='png', dpi=300, bbox_inches='tight')
        readme_path = savepath / "README.md"
        with readme_path.open("a") as f:
            f.write(f"\n###  Output File: {filename}\n- {DSName}: Visualization of Combined Bubble Mask\n")
    
    plt.show()