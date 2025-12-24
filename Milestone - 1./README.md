# Milestone 1: Dataset Preparation & Preprocessing Setup

## 1. Introduction

Milestone 1 focuses on preparing the raw scanned document dataset and implementing the preprocessing pipeline required for forensic scanner identification. The objective is to structure the dataset, extract metadata, analyze scanner characteristics, and generate standardized preprocessed images that will be used in later milestones for feature extraction and machine learning.

## 2. Dataset Organization

The raw dataset consists of **1400 TIFF images** scanned using **7 different scanner models** at two DPI levels (150 and 300 DPI).

The dataset was structured in the following hierarchical format:

```
Dataset/
   └── ScannerModel/
         └── DPI/
              └── *.tif
```

This organized structure enables automated traversal and processing of all scanned documents.

## 3. Dataset Inventory Creation

A Python script (`inventory.py`) was developed to generate an inventory of all dataset images.

For each file, the following metadata was extracted:
- Scanner model
- DPI (150 / 300)
- Filename
- File format
- Image dimensions (width × height)
- File size (bytes)
- Pixel mode (RGB)
- File path

The inventory is saved as `dataset_inventory.csv`.

### Purpose of Dataset Inventory
- Validates dataset completeness
- Ensures correct folder structure
- Helps in identifying image properties
- Enables statistical analysis before preprocessing

## 4. Dataset Analysis

Using `analysis.py`, statistical insights were generated from the inventory:

- **Total images:** 1400
- **Scanner models:** 7
- **Images per scanner:** 200 each
- **DPI distribution:** 700 (150 DPI), 700 (300 DPI)
- **Image format:** All TIFF
- **Mode:** All RGB
- **Resolution:** Around 1860×2630 pixels

This analysis confirmed that the dataset is consistent and suitable for controlled preprocessing.

## 5. Image Preprocessing Pipeline

The preprocessing script (`preprocess.py`) standardizes all scanned images for feature extraction. It includes the following steps:

### 5.1 Grayscale Conversion
All RGB TIFF images are converted into grayscale using OpenCV. This reduces complexity and normalizes luminance information.

### 5.2 Image Denoising
Non-Local Means (NLM) denoising is applied to remove content noise while preserving structure. This step produces a clean denoised image.

### 5.3 Residual Image Computation
```
Residual = Original grayscale image – Denoised image
```

The residual highlights scanner-specific noise patterns such as:
- Sensor noise
- Periodic banding
- Mechanical noise
- Internal processing artefacts

This residual is crucial for forensic identification.

### 5.4 Image Normalization
Residual images are normalized and converted to uint8 format (0–255). This ensures uniformity for feature extraction.

### 5.5 Standard Image Resizing
Denoised and residual images are resized to **256 × 256 pixels**. This ensures consistent dimensions for all further processing.

### 5.6 Saving Outputs
Processed files are saved inside:

```
processed/
   ├── denoised/
   └── residual/
```

The directory structure inside these folders mirrors the original dataset folders.

## 6. Milestone 1 Deliverables

The following files were created and added to the repository:

- `inventory.py` – Dataset metadata extraction
- `analysis.py` – Dataset statistical analysis
- `preprocess.py` – Image preprocessing pipeline
- `dataset_inventory.csv` – Complete dataset metadata
- `requirements.txt` – Python library dependencies
- `processed/` – Folder containing denoised and residual images

## 7. Outcome of Milestone 1

By the end of Milestone 1:

1. The dataset is fully validated and structured properly  
2. All images are preprocessed uniformly  
3. Residual noise maps (scanner fingerprints) are extracted   
4. A consistent pipeline is established for end-to-end scanner identification

---

