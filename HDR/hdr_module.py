# rumz 4/25

import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

def combine_hsi(
    cube_reflect: np.ndarray,
    cube_low: np.ndarray,
    cube_high: np.ndarray,
    Debug: bool = False,
    visualize_bands: list[int] = [96, 59, 16],  # choose band indices to visualize
    save: bool = False,
    savepath: str = ""
) -> np.ndarray:
    """
    Combines three hyperspectral cubes into one using HDR strategy.

    Args:
        cube_reflect (np.ndarray): Reflective image cube (shape: H x W x B)
        cube_low (np.ndarray): Transmitted light, low exposure
        cube_high (np.ndarray): Transmitted light, high exposure
        Debug (bool): If True, show weight maps for selected bands
        visualize_bands (list[int]): Bands to visualize if Debug is True

    Returns:
        np.ndarray: Combined hyperspectral cube
    """

    DSName = savepath.parent.name

    ref_norm = normalize_cube(cube_reflect)
    low_norm = normalize_cube(cube_low)
    high_norm = normalize_cube(cube_high)

    w_ref = get_weight(ref_norm)
    w_low = get_weight(low_norm)
    w_high = get_weight(high_norm)

    weight_sum = w_ref + w_low + w_high + 1e-6
    w_ref /= weight_sum
    w_low /= weight_sum
    w_high /= weight_sum

    if Debug:
        for band in visualize_bands:
            plt.figure(figsize=(15, 3))
            plt.suptitle(f"Band {band} - Weight Maps", fontweight='bold')
        
            plt.subplot(1, 3, 1)
            plt.imshow(w_ref[:, :, band], cmap='viridis')
            plt.title("Reflectance", fontweight='bold')
            plt.axis('off')
        
            plt.subplot(1, 3, 2)
            plt.imshow(w_low[:, :, band], cmap='plasma')
            plt.title("Transmitted Low", fontweight='bold')
            plt.axis('off')
        
            plt.subplot(1, 3, 3)
            plt.imshow(w_high[:, :, band], cmap='inferno')
            plt.title("Transmitted High", fontweight='bold')
            plt.axis('off')

            plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()

    combined = (
        w_ref * cube_reflect +
        w_low * cube_low +
        w_high * cube_high
    )

    if Debug and combined.shape[2] >= 96:
        rgb = combined[:, :, visualize_bands].astype(np.float32)
        for i in range(3):
            channel = rgb[:, :, i]
            c_min, c_max = np.percentile(channel, [1, 99])
            rgb[:, :, i] = np.clip((channel - c_min) / (c_max - c_min + 1e-6), 0, 1)

        rgb = np.power(rgb, 0.8)

        plt.figure(figsize=(24, 36))
        plt.imshow(rgb)
        #plt.title("Final Combined RGB Composite")
        plt.axis('off')
        if save == True:
            filename = f"{DSName}_HDR.png"
            plt.savefig(f"{savepath}/{filename}", dpi=300, bbox_inches='tight')
            readme_path = savepath/ "README.md"
            with readme_path.open("a") as f:
                f.write(f"\n### Output File: {filename}\n- {DSName}: High Dynamic Range Image\n")
        plt.show()

    return combined

def normalize_cube(cube):
    return cube / (np.max(cube, axis=(0, 1), keepdims=True) + 1e-6)

def get_weight(cube):
    return 1.0 - np.abs(cube - 0.5) * 2.0  # Peak weight near 0.5