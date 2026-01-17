# Scanner Device Identification from Scanned Documents

## Project Overview
This project focuses on identifying the **scanner brand/model** used to digitize documents by analyzing scanner-specific noise patterns and artifacts present in scanned images.  
A dataset of **600+ high-resolution scanned document images** was collected and preprocessed for machine learning–based classification.

---

## Dataset Description

### 1. Scanned Document Dataset
- **Total Images:** 600+
- **Scan Resolution:** 300 DPI  
- **File Format:** TIFF (`.tif`) — lossless, preserves fine-grain noise
- **Document Size:** A4 (typically 2480 × 3508 pixels at 300 DPI)
- **Color Format:** RGB and Grayscale
- **Bit Depth:** 8-bit per channel

#### Scanner Brands / Models
- **HP**
- **Canon**
- **Epson**

TIFF format was chosen to retain:
- Original pixel values
- Scanner-specific noise signatures
- Illumination and texture artifacts

---

## Dataset Labeling

Each image is labeled with its corresponding scanner brand/model.

### Directory Structure
dataset/
│
├── Canon/
│ └── s2_25.tif
├── Epson/
│ └── s8_47.tif
└── Hp/
└── s11_3.tif


### Label Mapping Example

| Image Path           | Brand  |
|----------------------|--------|
| `./Canon/s2_25.tif`  | canon  |
| `./Epson/s8_47.tif`  | epson  |
| `./Hp/s11_3.tif`     | hp     |

---

## Image Property Analysis (Week 1)

Before preprocessing, each image was analyzed for:

- **Resolution:** ~2480 × 3508 pixels
- **DPI:** 300 DPI (verified)
- **Format:** TIFF (`.tif`)
- **Color Channels:** RGB or Grayscale
- **Bit Depth:** 8-bit

This analysis ensured dataset consistency and quality before model training.

---

## Image Preprocessing (Week 2)

### 1. Image Resizing
- All images resized to **224 × 224** pixels
- Ensures uniform input size for deep learning models

### 2. Grayscale Conversion
- Converted all images to **grayscale**
- Reduces input channels from **3 → 1**
- Benefits:
  - Faster training
  - Lower memory usage
  - Focus on texture and noise patterns instead of color

### 3. Dataset Structuring
- Images converted to **NumPy arrays**
- Final shape:(224, 224, 1)