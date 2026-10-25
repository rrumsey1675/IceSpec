## rumz March 2025

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def get_sums(img: np.ndarray, Debug=False, save=False, savePath="", PCA=False) -> np.ndarray:
    """
    Build HxW images where each row is the sum/median/mean of that row across
    all channels and columns in the original image.

    Args:
        img (np.ndarray): Input image with shape (H, W, C).
        Debug (bool): If True, prints debug info.
        save (bool): If True, saves PNGs and logs to README.md in savePath.
        savePath (str): Directory to save outputs.

    Returns:
        np.ndarray: Sum image of shape (H, W) (row-wise sums broadcast across width).
    """
    if img.ndim != 3:
        raise ValueError("Expected image with shape (height, width, channels)")

    H, W, C = img.shape
    # Use float64 for stable reductions
    imgf = img.astype(np.float64, copy=False)

    # Row-wise reductions over width and channels -> shape (H,)
    row_sums  = np.sum(imgf, axis=(1, 2))        # total per row
    row_meds  = np.median(imgf, axis=(1, 2))     # median per row
    row_means = np.mean(imgf, axis=(1, 2))       # mean per row

    # Broadcast each (H,) to (H, W) by repeating across columns
    sums_img   = np.repeat(row_sums[:,  None], W, axis=1)
    median_img = np.repeat(row_meds[:,  None], W, axis=1)
    mean_img   = np.repeat(row_means[:, None], W, axis=1)

    if Debug:
        # show a few sample rows
        for i in range(min(5, H)):
            print(f"Row {i}: sum={row_sums[i]:.4g}, median={row_meds[i]:.4g}, mean={row_means[i]:.4g}")
    

    def show_and_save(arr: np.ndarray, title: str, fname: str, desc: str) -> None:
        plt.figure()
        plt.imshow(arr, cmap='gray')
        plt.axis('off')
        # plt.title(title)
        plt.tight_layout()
        if save:
            p = Path(savePath)
            p.mkdir(parents=True, exist_ok=True)
            plt.savefig(p / fname, dpi=300, bbox_inches='tight', pad_inches = 0)
            with (p / "README.md").open("a") as f:
                f.write(f"\n### Output File\n- {fname}: {desc}\n")
        plt.show()
        plt.close()

    if PCA:
        show_and_save(sums_img,   "Row Sum Image",   "PCArow_sums.png",
                      "HxW image with each row filled by the total intensity of that row across all channels.")
        show_and_save(median_img, "Row Median Image", "PCArow_medians.png",
                      "HxW image with each row filled by the median intensity of that row across all channels.")
        show_and_save(mean_img,   "Row Mean Image",   "PCArow_means.png",
                      "HxW image with each row filled by the mean intensity of that row across all channels.")
    else:
        show_and_save(sums_img,   "Row Sum Image",   "row_sums.png",
                      "HxW image with each row filled by the total intensity of that row across all channels.")
        show_and_save(median_img, "Row Median Image", "row_medians.png",
                      "HxW image with each row filled by the median intensity of that row across all channels.")
        show_and_save(mean_img,   "Row Mean Image",   "row_means.png",
                      "HxW image with each row filled by the mean intensity of that row across all channels.")

    return sums_img

        
        