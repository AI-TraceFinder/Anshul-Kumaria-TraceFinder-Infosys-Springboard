📌 AI TraceFinder — Milestone 1
Image Pre-processing & Noise Residual Extraction

This repository contains the work completed as part of Milestone 1 of the AI-TraceFinder internship project.
The goal of this milestone is to prepare the dataset for downstream tasks such as scanner identification and image forensics.

🚀 Milestone 1 Objectives

✔ Convert all input images into a standard format
✔ Resize & normalize images
✔ Convert to grayscale
✔ Extract noise residuals using Gaussian denoising
✔ Save output images batch-wise (safe for large datasets)
✔ Generate metadata CSVs
✔ Organize all results into a clean folder structure

📁 Folder Structure
AI_TraceFinder/
│
├── raw/                        # Original dataset (not uploaded)
│
├── processed/
│   ├── images/                 # Preprocessed 512x512 grayscale images
│   └── noise_maps/             # Extracted noise residuals
│
├── metadata/
│   ├── dataset_labels.csv      # Mapping: scanner → original → processed → noise
│   └── image_stats.csv         # Processing status for each image
│
└── preprocess.py               # Main preprocessing pipeline script

⚙️ Preprocessing Pipeline

The preprocessing script performs the following steps:

Load images batch-wise (default: 20 images at a time)

Convert to grayscale

Resize to 512×512 pixels

Normalize pixel intensity to 0–1

Perform Gaussian denoising (σ=1)

Compute:

noise_residual = original - denoised


Save:

Clean preprocessed image

Noise residual map

Store metadata (CSV)

🧪 Run Preprocessing Script

Make sure the folder structure is:

raw/
processed/
metadata/
preprocess.py


Then run:

python preprocess.py


Outputs will be generated inside:

processed/images/
processed/noise_maps/
metadata/

🔧 Batch Processing

To avoid crashes on low-RAM systems, processing is done in batches:

BATCH_SIZE = 20


You may adjust this:

Low RAM → BATCH_SIZE = 10

High RAM → BATCH_SIZE = 50

📊 Sample Outputs (Placeholders)

You can add your own screenshots later:

📷 processed_sample.png  
🎛️ noise_map_sample.png

📜 Metadata Description
dataset_labels.csv
Field	Description
scanner	Folder name (raw dataset category)
original	Original image path
processed	Output processed image path
noise	Extracted noise residual path
image_stats.csv

Tracks all successes/errors during processing.

🧑‍💻 Author
Tejaswini Dhamane
AI-TraceFinder Internship
Milestone 1 Completed Successfully ✔
