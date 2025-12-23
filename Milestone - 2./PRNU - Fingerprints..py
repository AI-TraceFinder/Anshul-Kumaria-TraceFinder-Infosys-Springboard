import os
import cv2
import numpy as np
from prnu_utils import extract_prnu_residual

DATASET_PATH = r"C:\Users\M.varshith reddy\Downloads\MileStone-1\Processed"
OUTPUT_PATH = "prnu_fingerprints"

os.makedirs(OUTPUT_PATH, exist_ok=True)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Dictionary to collect PRNU residuals per scanner MODEL
scanner_prnu_map = {}

for dataset in os.listdir(DATASET_PATH):
    dataset_path = os.path.join(DATASET_PATH, dataset)

    if not os.path.isdir(dataset_path):
        continue

    for scanner_label in os.listdir(dataset_path):
        scanner_path = os.path.join(dataset_path, scanner_label)

        if not os.path.isdir(scanner_path):
            continue

        # 🔥 Extract scanner MODEL (before first '-')
        scanner_model = scanner_label.split("-")[0]

        if scanner_model not in scanner_prnu_map:
            scanner_prnu_map[scanner_model] = []

        for file in os.listdir(scanner_path):
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue

            img_path = os.path.join(scanner_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            prnu = extract_prnu_residual(img)
            scanner_prnu_map[scanner_model].append(prnu)

# 🔥 Average PRNU per scanner model
for scanner_model, prnu_list in scanner_prnu_map.items():
    if len(prnu_list) == 0:
        continue

    fingerprint = np.mean(prnu_list, axis=0)

    save_path = os.path.join(OUTPUT_PATH, f"{scanner_model}.npy")
    np.save(save_path, fingerprint)

    print(f"Saved FINAL fingerprint for {scanner_model} ({len(prnu_list)} images)")
