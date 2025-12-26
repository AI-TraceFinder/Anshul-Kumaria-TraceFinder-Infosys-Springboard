import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from scipy.stats import skew, kurtosis

# ================= PATHS =================
LABELS_CSV = "../Milestone_1/metadata/dataset_labels.csv"
OUT_FEATURES = "features/features.csv"

LBP_RADII = [1, 2, 3]
LBP_POINTS = 8

os.makedirs("features", exist_ok=True)

# -------- FFT RADIAL ENERGY --------
def fft_radial_energy(img, bins=5):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    max_r = np.max(r)
    energies = []
    for i in range(bins):
        mask = (r >= i * max_r / bins) & (r < (i + 1) * max_r / bins)
        energies.append(np.mean(mag[mask]))

    return energies

# -------- FEATURE EXTRACTION --------
def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"❌ Image not found: {img_path}")

    img = img.astype(np.float32) / 255.0

    noise_mean = np.mean(img)
    noise_std = np.std(img)
    noise_skew = skew(img.flatten())
    noise_kurt = kurtosis(img.flatten())
    noise_energy = np.sum(img ** 2) / img.size

    fft_feats = fft_radial_energy(img)

    lbp_feats = []
    for r in LBP_RADII:
        lbp = local_binary_pattern(
            img, LBP_POINTS * r, r, method="uniform"
        )
        hist, _ = np.histogram(
            lbp,
            bins=np.arange(0, LBP_POINTS * r + 3),
            density=True
        )
        lbp_feats.extend(hist)

    return (
        [noise_mean, noise_std, noise_skew, noise_kurt, noise_energy]
        + fft_feats
        + lbp_feats
    )

# -------- MAIN --------
def main():
    print("🚀 Milestone 2 – Feature Extraction Started")

    if not os.path.exists(LABELS_CSV):
        print("❌ dataset_labels.csv not found")
        return

    df = pd.read_csv(LABELS_CSV)
    print("Total images:", len(df))

    # Debug first image
    first_img = os.path.join("../Milestone_1", df.iloc[0]["processed"])
    print("First image path:", first_img)
    print("Exists:", os.path.exists(first_img))

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join("../Milestone_1", row["processed"])
        feats = extract_features(img_path)
        rows.append([row["scanner"]] + feats)

    columns = (
        ["scanner",
         "noise_mean", "noise_std", "noise_skew",
         "noise_kurt", "noise_energy"]
        + [f"fft_band_{i}" for i in range(5)]
        + [f"lbp_{i}" for i in range(len(rows[0]) - 11)]
    )

    feat_df = pd.DataFrame(rows, columns=columns)
    feat_df.to_csv(OUT_FEATURES, index=False)

    print("✅ Features saved to:", OUT_FEATURES)

# -------- RUN --------
if __name__ == "__main__":
    main()
