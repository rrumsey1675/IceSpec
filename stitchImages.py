import os
import csv
import subprocess
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image  

def get_image_height(png_path: str) -> int:
    ### ADD CM per PX code
    """Return height of the image in pixels."""
    with Image.open(png_path) as img:
        return img.size[1]

def load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[["Folder", "TopDepth", "BottomDepth"]]
    df = df.dropna()
    df["TopDepth"] = df["TopDepth"].astype(float)
    df["BottomDepth"] = df["BottomDepth"].astype(float)
    return df

def find_section_images(root_dir: str, metadata_df: pd.DataFrame) -> List[Tuple[float, str, float]]:
    """
    For each folder listed in metadata, find the first PNG image.
    Returns a list of tuples: (TopDepth, image_path, depth_per_pixel)
    """
    image_info = []
    for _, row in metadata_df.iterrows():
        folder = Path(root_dir) / row["Folder"]
        pngs = sorted(folder.glob("*.png"))
        if pngs:
            height_px = get_image_height(pngs[0])
            depth_range = row["BottomDepth"] - row["TopDepth"]
            dpp = depth_range / height_px  # meters per pixel
            image_info.append((row["TopDepth"], str(pngs[0]), dpp))
    return sorted(image_info, key=lambda x: x[0])  # sort by TopDepth

def create_gray_filler(height_px: int, width_px: int, output_path: str):
    """Create a gray filler image using ImageMagick."""
    command = ["convert", "-size", f"{width_px}x{height_px}", "xc:gray", output_path]
    subprocess.run(command, check=True)

def stitch_with_fillers(images: List[Tuple[float, str, float]], output_path: str):
    stitched_images = []
    prev_bottom = None
    for i, (top, img_path, dpp) in enumerate(images):
        if prev_bottom is not None and top > prev_bottom:
            # Gap detected
            gap_m = top - prev_bottom
            gap_px = int(round(gap_m / dpp))
            with Image.open(img_path) as sample:
                width = sample.size[0]
            filler_path = f"filler_{i}.png"
            create_gray_filler(gap_px, width, filler_path)
            stitched_images.append(filler_path)

        stitched_images.append(img_path)
        prev_bottom = top + get_image_height(img_path) * dpp  # or use BottomDepth directly

    # Stitch all images
    command = ["convert"] + stitched_images + ["-append", output_path]
    subprocess.run(command, check=True)
    print(f"Saved stitched image to {output_path}")

if __name__ == "__main__":
    ROOT_DIR = "path/to/ice_core_sections"
    METADATA_CSV = "path/to/metadata.csv"
    OUTPUT_PATH = "stitched_core.png"

    metadata = load_metadata(METADATA_CSV)
    images = find_section_images(ROOT_DIR, metadata)
    stitch_with_fillers(images, OUTPUT_PATH)
