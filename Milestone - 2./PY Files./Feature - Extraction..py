import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern

# ----------------------------
# PATHS
# ----------------------------
PROCESSED_PATH = r"C:\Users\M.varshith reddy\Downloads\MileStone-1\Processed"
OUTPUT_CSV = "features_with_prnu.csv"

DATASET_FOLDERS = ["Flatfield", "Official", "Wikipedia"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ----------------------------
# LBP PARAMETERS
# ----------------------------
LBP_P = 8
LBP_R = 1

rows = []

# ----------------------------
# FEATURE EXTRACTION FUNCTION
# ----------------------------
def extract_features(img):

    # 1️⃣ Mean intensity
    mean_intensity = np.mean(img)

    # 2️⃣ Standard deviation
    std_intensity = np.std(img)

    # 3️⃣ FFT Feature (frequency magnitude)
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    fft_mag = np.mean(np.abs(fft_shift))

    # 4️⃣ LBP Texture Features
    lbp = local_binary_pattern(img, LBP_P, LBP_R, method="uniform")
    lbp_mean = np.mean(lbp)
    lbp_std = np.std(lbp)

    # 5️⃣ Simple Noise Map (baseline)
    blurred_small = cv2.GaussianBlur(img, (3, 3), 0)
    noise_basic = img.astype("float32") - blurred_small.astype("float32")
    noise_variance = np.var(noise_basic)

    # ==========================
    # 🔥 6️⃣ PRNU FEATURES (NEW)
    # ==========================

    # Stronger blur to remove image content
    blurred_prnu = cv2.GaussianBlur(img, (5, 5), 0)

    # Noise residual (PRNU approximation)
    prnu_noise = img.astype("float32") - blurred_prnu.astype("float32")

    prnu_variance = np.var(prnu_noise)
    prnu_energy = np.sum(prnu_noise ** 2)
    prnu_mean = np.mean(prnu_noise)

    return (
        mean_intensity,
        std_intensity,
        fft_mag,
        lbp_mean,
        lbp_std,
        noise_variance,
        prnu_variance,
        prnu_energy,
        prnu_mean
    )

# ----------------------------
# MAIN LOOP
# ----------------------------
for dataset in DATASET_FOLDERS:
    dataset_path = os.path.join(PROCESSED_PATH, dataset)

    if not os.path.exists(dataset_path):
        print(f"Dataset missing: {dataset_path}")
        continue

    for scanner in os.listdir(dataset_path):
        scanner_path = os.path.join(dataset_path, scanner)

        if not os.path.isdir(scanner_path):
            continue

        for file in os.listdir(scanner_path):

            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue

            img_path = os.path.join(scanner_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print("Could not read:", img_path)
                continue

            features = extract_features(img)

            rows.append([
                dataset,
                scanner,
                file,
                *features
            ])

            print("Extracted:", img_path)

# ----------------------------
# SAVE CSV
# ----------------------------
df = pd.DataFrame(rows, columns=[
    "dataset",
    "scanner_label",
    "image_name",
    "mean_intensity",
    "std_intensity",
    "fft_feature",
    "lbp_mean",
    "lbp_std",
    "noise_variance",
    "prnu_variance",
    "prnu_energy",
    "prnu_mean"
])

df.to_csv(OUTPUT_CSV, index=False)

print("\n🎉 Feature extraction complete with PRNU!")
print("Saved as:", OUTPUT_CSV)
