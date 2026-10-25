## rumz 6/4/25

import os
from pathlib import Path
from datetime import datetime
from typing import Optional
from typing import List, Tuple, Dict, Any

def OutputPath(
    dataList,
    script_id: str,
    notes: str = "",
    batch_id: str = ""
) -> Path:
    """
    Creates and returns a timestamped folder for saving processed images or files,
    and logs metadata in a README.md file for reproducibility.

    Args:
        dataList (List): List of image metadata.
        script_id (str): Identifier for the script generating the output.
        notes (str, optional): Additional human-readable notes for this run.
        batch_id (str, optional): Optional batch identifier if running multiple sections at once.

    Returns:
        Path: The created output directory path.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(dataList[0][4]) / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    readme_path = output_dir / "README.md"
    with readme_path.open("w") as f:
        f.write(f"# Output Metadata\n")
        f.write(f"- Timestamp : {timestamp}\n")
        f.write(f"- Script ID : {script_id}\n")
        if batch_id:
            f.write(f"- Batch ID  : {batch_id}\n")
        
        # General Core Metadata
        f.write("\n## Core Metadata\n")
        base_path = Path(dataList[0][0])
        f.write(f"- dataName   : {base_path.name}\n")
        f.write(f"- dataDir    : {base_path.parent.name}\n")
        f.write(f"- coreName   : {dataList[0][10]}\n")
        f.write(f"- section    : {dataList[0][11]}\n")
        f.write(f"- top_depth  : {dataList[0][1]} m\n")
        f.write(f"- bottom_depth : {dataList[0][2]} m\n")
        f.write(f"- core length  : {round(dataList[0][3], 2)} m\n")
        f.write(f"- px_dims      : {dataList[0][5]}\n")
        f.write(f"- CSV File     : {dataList[0][12]}\n")

        # Reflectance
        f.write("\n## Reflectance Image Data\n")
        f.write(image_entry(dataList[0]))

        # Transmitted Low
        f.write("\n## Transmitted Low Image Data\n")
        f.write(image_entry(dataList[1]))

        # Transmitted High
        f.write("\n## Transmitted High Image Data\n")
        f.write(image_entry(dataList[2]))

        if notes:
            f.write(f"\n## Notes\n{notes.strip()}\n")

    return output_dir

def image_entry(data: List) -> str:
    return (
        f"- DataID      : {data[9]}\n"
        f"- topLight    : {data[6]}%\n"
        f"- bottomLight : {data[7]}%\n"
        f"- exposure    : {data[8]}\n"
    )