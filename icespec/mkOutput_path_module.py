## rumz 5/22//25

import os
from pathlib import Path
from datetime import datetime
from typing import Optional

def OutputPath(
    core_section_path: str,
    script_id: str,
    metadata: dict,
    batch_ts: str = "",
    notes: str = "",
    batch_id: str = ""
) -> Path:
    """
    Creates and returns a timestamped folder for saving processed images or files,
    and logs metadata in a README.md file for reproducibility.

    Args:
        core_section_path (str): Path to the core section directory.
        script_id (str): Identifier for the script generating the output.
        metadata (dict): Dictionary of metadata values related to the dataset.
        notes (str, optional): Additional human-readable notes for this run.
        batch_id (str, optional): Optional batch identifier if running multiple sections at once.

    Returns:
        Path: The created output directory path.
    """
    if batch_ts == "":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    else:
        timestamp = batch_ts
    output_dir = Path(core_section_path) / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    readme_path = output_dir / "README.md"
    with readme_path.open("w") as f:
        f.write(f"# Output Metadata\n")
        f.write(f"- Timestamp : {timestamp}\n")
        f.write(f"- Script ID : {script_id}\n")
        if batch_id:
            f.write(f"- Batch ID  : {batch_id}\n")
        f.write(f"\n## Core Metadata\n")
        for key, value in metadata.items():
            f.write(f"- {key} : {value if value is not None else 'N/A'}\n")
        if notes:
            f.write(f"\n## Notes\n{notes.strip()}\n")

    return output_dir