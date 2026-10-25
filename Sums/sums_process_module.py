# rumz 2/21/25
# Processing code for hyperspectral ice core data

import os
import numpy as np
import matplotlib.pyplot as plt
import spectral
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple

# Importing custom modules from the Sums package
from Sums import channel_sum_module as sum_mod
from Sums import channel_sum_calculations as sum_calc

from dtaidistance import dtw, dtw_visualisation as dtwvis
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

# Default RGB band indices for visualization
rd, gr, bl = 96, 59, 16


def channel_sums(img: np.ndarray, px_per_cm, top_depth, Debug=False, save=False, savePath: Optional[Path] = None, PCA: bool = False ) -> np.ndarray:
    """
    Gets sum of each line per channel, then uses channel sums to remake image.

    Args:
     - img: (np.ndarray): Reflectance-corrected and bubble filled image.
     - Debug (bool, optional): If True, prints debug information about bubble counts.
     - save (bool, optional): If True, saves image to specified folder.
     - savePath (str, optional): used if save is True, specifes path to save folder and file name.

     Returns:
        np.ndarray: Sum Value image.
        pd.dataframe: Image row-wise data; columns:[Depth_cm, Sums, Means, Medians]
    """
    sums_img = sum_mod.get_sums(img, Debug, save, savePath, PCA)
    full_img_sums = sum_calc.get_sums(img, px_per_cm, top_depth, Debug)
    if save:
        if PCA: 
            csv_name = "PCA_Sums.csv"
        else:
            csv_name = "Sums.csv"
        savePath = Path(savePath)
        out_csv = savePath / csv_name
        full_img_sums.to_csv(out_csv, index=False)

        readme_path = savePath / "README.md"
        with readme_path.open("a") as f:
            f.write("\n### Output File\n")
            f.write(f"- {csv_name}: Row-wise sums/means/medians for image.\n")

    return sums_img, full_img_sums

def sum_comparison(
    img: np.ndarray,
    px_per_cm: float,
    top_depth: float,
    Debug: bool = False,
    save: bool = False,
    savePath: Optional[Path] = None,
    PCA: bool = False,
    section_width: int = 40,
    n_sections: int = 5,
    csv_name: str = "sections_row_summaries.csv"
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Slice the image into n_sections vertical bands, compute row-wise stats via
    sum_calc.get_sums for each band, and write ONE combined CSV.

    Returns:
        combined_df: long DataFrame with columns:
            [Depth_cm, Sums, Means, Medians, Section, X_start_px, X_end_px]
        per_section: list of per-section DataFrames (same columns minus Section/coords)
    """
    if img.ndim != 3:
        raise ValueError("Expected image with shape (height, width, channels)")

    if save and savePath is None:
        raise ValueError("save=True requires savePath")

    H, W, C = img.shape
    section_width = int(section_width)
    n_sections = int(n_sections)

    # ensure width is reasonable
    if section_width > W:
        section_width = max(1, W // n_sections)

    # Evenly spaced section centers across width; clamp starts to [0, W - section_width]
    centers = np.linspace(section_width / 2, W - section_width / 2, n_sections)
    starts = np.clip((centers - section_width / 2).astype(int), 0, max(0, W - section_width))

    per_section: List[pd.DataFrame] = []
    meta_rows = []  # Keep (section_id, x_start, x_end) for annotation

    for i, s in enumerate(starts, start=1):
        sl = slice(s, s + section_width)
        section_img = img[:, sl, :]
        df = sum_calc.get_sums(section_img, px_per_cm, top_depth, Debug)
        df = df.copy()  # avoid SettingWithCopy warnings
        df["Section"] = i
        df["X_start_px"] = int(s)
        df["X_end_px"] = int(s + section_width - 1)
        per_section.append(df)
        meta_rows.append((i, int(s), int(s + section_width - 1)))

    combined_df = pd.concat(per_section, ignore_index=True)

    # Save one CSV + README entry
    if save:
        if PCA:
            csv_name = "PCA-SumsSections.csv"
            savePath = Path(savePath)
            out_csv = savePath / csv_name
            combined_df.to_csv(out_csv, index=False)
    
            readme_path = savePath / "README.md"
            with readme_path.open("a") as f:
                f.write("\n### Output File\n")
                f.write(f"- {csv_name}: Row-wise sums/means/medians for {n_sections} vertical sections.\n")
                for sec_id, x0, x1 in meta_rows:
                    f.write(f"  - Section {sec_id}: X [{x0}:{x1}] (width={section_width})\n")
        else: 
            csv_name = "SumsSections.csv"
            savePath = Path(savePath)
            out_csv = savePath / csv_name
            combined_df.to_csv(out_csv, index=False)
    
            readme_path = savePath / "README.md"
            with readme_path.open("a") as f:
                f.write("\n### Output File\n")
                f.write(f"- {csv_name}: Row-wise sums/means/medians for {n_sections} vertical sections.\n")
                for sec_id, x0, x1 in meta_rows:
                    f.write(f"  - Section {sec_id}: X [{x0}:{x1}] (width={section_width})\n")

    try:
        if n_sections == 5:
            graphSums(*per_section, Debug=Debug, save=save, savePath=savePath, PCA=PCA)
    except NameError:
        # graphSums not in scope; ignore
        pass

    return combined_df, per_section


def graphSums(sums1: pd.DataFrame, sums2: pd.DataFrame, sums3: pd.DataFrame, sums4: pd.DataFrame, sums5: pd.DataFrame, Debug=False, save=False, savePath: Optional[Path] = None, PCA: bool = False):
    """
    Graphs the imgSums

    Args:
     - sums1: (pd.DataFrame): dataframe containing row sums for image segment 1 by depth(cm)
     - sums2: (pd.DataFrame): dataframe containing row sums for image segment 2 by depth(cm)
     - sums3: (pd.DataFrame): dataframe containing row sums for image segment 3 by depth(cm)
     - Debug (bool, optional): If True, prints debug information about bubble counts.
     - save (bool, optional): If True, saves image to specified folder.
     - savePath (str, optional): used if save is True, specifes path to save folder and file name.
    """
    fig, axs = plt.subplots(1, 5, figsize=(8, 20), sharey=True)

    x1 = sums1.iloc[:, 1]
    y1 = sums1["Depth_cm"]

    x2 = sums2.iloc[:, 1]
    y2 = sums2["Depth_cm"]

    x3 = sums3.iloc[:, 1]
    y3 = sums3["Depth_cm"]

    x4 = sums4.iloc[:, 1]
    y4 = sums4["Depth_cm"]

    x5 = sums5.iloc[:, 1]
    y5 = sums5["Depth_cm"]

    axs[0].scatter(x1, y1, s=1, color='tab:blue')
    axs[0].set_title('Strip 1',  fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Depth (cm)', fontweight='bold')
    axs[0].grid(True)
    axs[0].invert_yaxis()
    
    axs[1].scatter(x2, y2, s=1, color='tab:red')
    axs[1].set_title('Strip 2',  fontsize=12, fontweight='bold')
    axs[1].grid(True)
    axs[1].invert_yaxis()
    
    axs[2].scatter(x3, y3, s=1, color='tab:orange')
    axs[2].set_title('Strip 3',  fontsize=12, fontweight='bold')
    axs[2].grid(True)
    axs[2].invert_yaxis()
    
    axs[3].scatter(x4, y4, s=1, color='tab:green')
    axs[3].set_title('Strip 4',  fontsize=12, fontweight='bold')
    axs[3].grid(True)
    axs[3].invert_yaxis()
    
    axs[4].scatter(x5, y5, s=1, color='tab:red')
    axs[4].set_title('Strip 5',  fontsize=12, fontweight='bold')
    axs[4].grid(True)
    axs[4].invert_yaxis()
    
    plt.suptitle("Spectral Sums vs Depth", y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        if PCA:
            plt.savefig(f"{savePath}/PCAsumGraph.png", dpi=300, bbox_inches='tight')
            readme_path = savePath.parent / "README.md"
            with readme_path.open("a") as f:
                f.write(f"\n### Output File\n- PCAsumGraph.png: Plot of row-wise pixel intensity sums for three image sections\n")
        else:
            plt.savefig(f"{savePath}/sumGraph.png", dpi=300, bbox_inches='tight')
            readme_path = savePath.parent / "README.md"
            with readme_path.open("a") as f:
                f.write(f"\n### Output File\n- sumGraph.png: Plot of row-wise pixel intensity sums for three image sections\n")

    plt.show()

    ## DTW
    s1 = x1.values[0::5]
    s2 = x2.values[0::5]
    s3 = x3.values[0::5]
    s4 = x4.values[0::5]
    s5 = x5.values[0::5]

    depths = y5 / 100
    depths = depths.values[0::5]

    s1 = np.nan_to_num(s1)
    s2 = np.nan_to_num(s2)
    s3 = np.nan_to_num(s3)
    s4 = np.nan_to_num(s4)
    s5 = np.nan_to_num(s5)

    # depths = np.nan_to_num(depths)

    distance, paths = dtw.warping_paths(s1, s5)
    best_path = dtw.best_path(paths)
    #similarity_score = distance / len(best_path)
    
    # distance1, paths1 = dtw.warping_paths(s2, s3)
    # best_path1 = dtw.best_path(paths1)
    # similarity_score1 = distance1 / len(best_path1)
    
    # distance2, paths2 = dtw.warping_paths(s3, s4)
    # best_path2 = dtw.best_path(paths2)
    # similarity_score2 = distance2 / len(best_path2)
    
    # distance3, paths3 = dtw.warping_paths(s3, s5)
    # best_path3 = dtw.best_path(paths3)
    # similarity_score3 = distance3 / len(best_path3)

    # plt.figure(figsize=(12, 8))

    # # Original Time Series Plot
    # ax1 = plt.subplot2grid((2, 2), (0, 0))
    # ax1.plot(s1, label='Series 1', color='blue')
    # ax1.plot(s5, label='Series 2', linestyle='--',color='orange')
    # ax1.set_title('Original Time Series')
    # ax1.legend()
    
    # # Shortest Path Plot (Cost Matrix with the path)
    # # In this example, only the path is plotted, not the entire cost matrix.
    
    # ax2 = plt.subplot2grid((2, 2), (0, 1))
    # ax2.plot(np.array(best_path)[:, 0], np.array(best_path)[:, 1], 'green', marker='o', linestyle='-')
    # ax2.set_title('Shortest Path (Best Path)')
    # ax2.set_xlabel('Series 1')
    # ax2.set_ylabel('Series 2')
    # ax2.grid(True)
    
    # # Point-to-Point Comparison Plot
    # ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    # ax3.plot(s1, label='Series 1', color='blue', marker='o')
    # ax3.plot(s5+12000, label='Series 2 + offset', color='orange', marker='x', linestyle='--')
    # for a, b in best_path:
    #     ax3.plot([a, b], [s1[a], s5 + offset[b]], color='grey', linestyle='-', linewidth=1, alpha = 0.5)
    # ax3.set_title('Point-to-Point Comparison After DTW Alignment')
    # ax3.legend()
    
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    offset = 10000  # vertical spacing between the series
    s2_offset = s5 + offset
    # s3_offset = s5 + (offset * 2)
    # s4_offset = s4 + (offset * 3)
    # s5_offset = s5 + (offset * 4)
    # combined_df = combined_df.sort_values(by='Depth_cm')
    # combined_df['Depth_cm'] = combined_df['Depth_cm'] / 100
    # depths = combined_df["Depth_cm"].values[0::5]  # assumes same for both series
    
    # Flip so Depth is y-axis
    plt.figure(figsize=(8, 20))

    filtered_path = []

    for (a1, b1), (a2, b2) in zip(best_path[:-1], best_path[1:]):
        if (a2 - a1 == 1) and (b2 - b1 == 1):
            filtered_path.append((a2, b2))
    
    # Draw alignment lines
    for a, b in filtered_path:
        plt.plot([s1[a], s2_offset[b]], [depths[a], depths[b]], color='gray', alpha=0.5, linewidth=0.5)
    # for a, b in best_path3:
    #     plt.plot([s2_offset[a], s3_offset[b]], [depths[a], depths[b]], color='gray', alpha=0.5, linewidth=0.5)
    # for a, b in best_path2:
    #     plt.plot([s3_offset[a], s4_offset[b]], [depths[a], depths[b]], color='gray', alpha=0.5, linewidth=0.5)
    # for a, b in best_path3:
    #     plt.plot([s4_offset[a], s5_offset[b]], [depths[a], depths[b]], color='gray', alpha=0.5, linewidth=0.5)
    
    # Plot Series
    plt.plot(s1, depths, label='Left', color='blue', marker='.', markersize=2)
    plt.plot(s2_offset, depths, label='Right (+ offset: 10000)', color='orange', marker='.', markersize=2)
    # plt.plot(s3_offset, depths, label='Right', color='green', marker='.', markersize=2)
    # plt.plot(s4_offset, depths, label='Strip 4 + offset', color='red', marker='x', linestyle='--', markersize=2)
    # plt.plot(s5_offset, depths, label='Strip 5 + offset', color='purple', marker='.', markersize=2)
    
    plt.gca().invert_yaxis()  # Depth increases downward
    
    plt.ylabel('Depth (m)', fontsize=14)
    plt.xlabel('Cumulative Pixel Intensity', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    sectionDir = os.path.dirname(savePath)
    sectionName = os.path.basename(sectionDir)

    plt.title(f'{sectionName}: Left vs. Right Side Intensity with Dynamic Time Warping', fontsize=18, fontweight='bold')
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{savePath}/DTW.png", dpi=300, bbox_inches='tight')
    plt.show()
