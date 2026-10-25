# Hyperspectral Ice Core Analysis
## Codebase Developer Guide

---

## 1. Project Overview

This codebase provides a research and analysis platform for hyperspectral imaging of ice cores. It processes hyperspectral image data (HSIUM format) to extract scientific measurements including bubble detection, dust loading, and stratigraphic layer analysis.

The platform is designed for glaciological research and supports both single-core interactive workflows and large-scale batch processing across many cores simultaneously.

---

## 2. Repository Structure

| Directory | Purpose |
|-----------|---------|
| `Notebooks/` | Jupyter notebooks for interactive exploration, visualization, and prototyping of analysis workflows. |
| `icespec/` | Sample acquisition and preprocessing algorithms. Handles raw hyperspectral data ingestion, calibration, and preparation for downstream analysis. |
| `bubbles/` | Algorithms for bubble detection and bubble data processing. Identifies and characterizes air bubbles within ice core images. |
| `Sums/` | Algorithms for basic layer flagging. Detects and marks stratigraphic layers based on spectral summation methods. |
| `Dust/` | Algorithms for detecting and quantifying dust loading in ice cores. |
| `stat/` | Statistical analysis scripts. Includes both single-core and batch-processing variants (see Section 4). |
| `targets.txt` | Reference file listing core IDs for batch processing runs. |

---

## 3. Data Format & Sample IDs

Each hyperspectral acquisition is assigned a unique sample ID of the form:

```
HSIUM-<n>
```

where `HSI` stands for **Hyperspectral Image**, `UM` for **UMaine** (University of Maine, where this research is conducted), and `<n>` is a sequential integer. IDs currently span from `HSIUM-1` to `HSIUM-908`.

These IDs are **not filenames**. They are lookup keys into a central CSV file that holds all sample metadata. When a script receives an ID like `HSIUM-42`, it queries the CSV to retrieve everything needed for that acquisition, including:

- The hyperspectral image filename and its file path
- Core depth information
- Image acquisition settings (exposure, binning, etc.)
- Any other per-sample metadata

This means scripts never need to know where image files live on disk directly — they only need a valid sample ID, and the CSV provides the rest.

> **Example:** passing `HSIUM-42` to a script causes it to look up row `HSIUM-42` in the CSV, find the associated image file and settings, and run the analysis on that acquisition.

---

## 4. Running Analysis Scripts

### 4.1 Single-Core Scripts

Scripts in the `stat/` folder **without** "batch" in their filename are designed to process one core at a time. They are invoked directly from the command line:

```bash
./<filename>.py HSIUM-<n>
```

Examples:

```bash
./analyze_layers.py HSIUM-42
./dust_stats.py HSIUM-317
```

> **Note:** Make sure the script is executable before running. If needed: `chmod +x <filename>.py`

### 4.2 Batch Processing Scripts

Scripts with "batch" in their filename are designed for processing many cores in sequence. Rather than accepting a single core ID as an argument, they read from `targets.txt`:

```bash
./batch_analyze.py
```

#### targets.txt Format

Each line should contain one core ID:

```
HSIUM-42
HSIUM-317
HSIUM-605
```

> **Note:** Blank lines and lines starting with `#` are treated as comments and ignored by the batch runner.

### 4.3 Jupyter Notebooks

The `Notebooks/` directory contains interactive Jupyter notebooks for exploratory analysis and visualization. Launch them via:

```bash
jupyter notebook Notebooks/
```

Notebooks are a good starting point for understanding a new analysis method before it is formalized into a standalone script.

---

## 5. Module Details

### 5.1 `icespec` — Preprocessing

The `icespec` module handles the earliest stages of the pipeline: ingesting raw hyperspectral images, applying calibrations, and producing cleaned spectral data arrays ready for analysis. Any new core should be run through `icespec` preprocessing before being passed to downstream modules.

Key responsibilities:
- Raw HSIUM file loading and validation
- Dark current and flat field correction
- Spectral calibration and normalization
- Spatial registration and cropping

### 5.2 `bubbles` — Bubble Detection

The `bubbles` module provides algorithms for identifying air bubbles within hyperspectral ice core images and extracting quantitative properties such as bubble size distribution, number density, and spatial patterns.

Key responsibilities:
- Detection of bubble boundaries using spectral contrast
- Morphological filtering to remove noise and artifacts
- Bubble size and shape characterization
- Output of per-bubble data tables for statistical analysis

### 5.3 `Sums` — Layer Flagging

The `Sums` module implements basic stratigraphic layer detection by computing spectral summations across the core depth axis. Anomalous bands in the summed signal are flagged as candidate layer boundaries.

Key responsibilities:
- Computation of spectral band sums along depth profiles
- Threshold-based flagging of candidate stratigraphic layers
- Output of layer depth positions for further review

### 5.4 `Dust` — Dust Loading Detection

The `Dust` module detects and quantifies particulate loading in ice core sections. Elevated dust concentrations are often associated with glacial periods, volcanic events, or windblown transport events, making this analysis critical for paleoclimate interpretation.

Key responsibilities:
- Spectral signature analysis for dust-indicative wavelength ranges
- Spatial mapping of high-dust zones
- Quantitative dust loading estimates per depth interval

### 5.5 `stat` — Statistical Analysis

The `stat` folder houses standalone Python scripts for computing and summarizing analytical results. These scripts sit downstream of the module algorithms and typically operate on output data rather than raw imagery.

| Script Type | Description |
|-------------|-------------|
| No "batch" in name | Single-core analysis. Run with: `./<n>.py HSIUM-<n>` |
| Contains "batch" | Multi-core analysis. Reads core list from `targets.txt`. Run with: `./<n>.py` |

---

## 6. Typical Analysis Workflow

The following sequence illustrates a standard end-to-end analysis run for a new ice core:

1. Add the sample ID(s) to `targets.txt` (for batch runs) or prepare to pass one on the command line.
2. Run `icespec` preprocessing to calibrate and normalize the raw hyperspectral image.
3. Run bubble detection via the `bubbles` module scripts or relevant `stat` scripts.
4. Run layer flagging via the `Sums` module.
5. Run dust loading detection via the `Dust` module.
6. Aggregate results using the appropriate `stat` scripts (single-core or batch).
7. Use Jupyter notebooks in `Notebooks/` to visualize and explore results interactively.

---

## 7. Code Conventions & Best Practices

### 7.1 File Naming

- Single-core stat scripts: descriptive name with no "batch" keyword, e.g. `bubble_stats.py`
- Batch stat scripts: include "batch" in the name, e.g. `batch_bubble_stats.py`
- Sample IDs always use the `HSIUM-<n>` format (e.g. `HSIUM-42`) — never use filenames or paths directly in place of IDs.

### 7.2 targets.txt Management

Keep `targets.txt` clean and well-commented. Use `#` to mark groups or flag cores that should be temporarily skipped:

```
# Spring 2023 acquisitions
HSIUM-42
HSIUM-43

# Deep core samples — 2024 season
HSIUM-317
# HSIUM-318  # skip — corrupted scan
```

### 7.3 Adding New Algorithms

When contributing a new algorithm:

- Place preprocessing/acquisition code in `icespec/`
- Place detection/segmentation algorithms in the relevant module folder (`bubbles/`, `Sums/`, or `Dust/`)
- Add a single-core stat script in `stat/` for testing on individual cores before writing a batch variant
- Document expected inputs and outputs at the top of each script
- Consider adding a companion Jupyter notebook in `Notebooks/` for visualization

---

## 8. Quick Reference

| Task | Command / Location |
|------|--------------------|
| Run single-core stat script | `./<script>.py HSIUM-<n>` |
| Run batch stat script | `./<batch_script>.py` (reads `targets.txt`) |
| Add samples for batch run | Edit `targets.txt` — one `HSIUM-<n>` per line |
| Preprocess a new core | Scripts in `icespec/` |
| Detect bubbles | Scripts in `bubbles/` |
| Flag stratigraphic layers | Scripts in `Sums/` |
| Detect dust loading | Scripts in `Dust/` |
| Interactive / exploratory work | Jupyter notebooks in `Notebooks/` |