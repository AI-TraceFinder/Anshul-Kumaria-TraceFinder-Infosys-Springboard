"""
FEATURE EXTRACTION
------------------------------------
Adds:
✔ LBP (P=16, R=2)
✔ FFT with 5 radial bands
✔ Wavelet energy features (2 levels)
✔ Extended residual statistics (entropy, MAD, percentiles)

OUTPUTS:
 - features/tracefinder_v2_features.npy
 - features/tracefinder_v2_labels.npy
 - features/tracefinder_v2_features.csv
 - features/tracefinder_v2_label_map.json
"""

import os
import cv2
import numpy as np
import csv
import json
from skimage.feature import local_binary_pattern
from scipy.stats import skew, kurtosis, entropy
import pywt
from tqdm import tqdm
import argparse

# -------------------------
# LBP Configuration
# -------------------------
LBP_P = 16         # neighbors
LBP_R = 2          # radius
LBP_METHOD = "uniform"  # uniform LBP (P+2 bins)


def lbp_histogram(img_gray):
    """Compute normalized LBP histogram."""
    lbp = local_binary_pattern(img_gray, LBP_P, LBP_R, method=LBP_METHOD)
    n_bins = LBP_P + 2  # uniform bins
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


# ------------------------------------------------------------------
# FFT – Radial Frequency Energy Bands
# ------------------------------------------------------------------
def fft_radial_bands(img_gray, n_bands=5):
    f = np.fft.fft2(img_gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_mag = np.log1p(magnitude)

    h, w = img_gray.shape
    cy, cx = h // 2, w // 2

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    max_radius = dist.max()
    bins = np.linspace(0, max_radius, n_bands + 1)

    energies = []
    total_energy = np.sum(log_mag) + 1e-12

    for i in range(n_bands):
        mask = (dist >= bins[i]) & (dist < bins[i + 1])
        band_energy = np.sum(log_mag[mask])
        energies.append(band_energy / total_energy)

    return np.array(energies, dtype=np.float32)


# ------------------------------------------------------------------
# Wavelet Features
# ------------------------------------------------------------------
def wavelet_energies(img_gray):
    """Compute wavelet band energies using Daubechies (db2)."""
    coeffs = pywt.wavedec2(img_gray, "db2", level=2)

    energies = []

    # Level 1: (cH, cV, cD)
    cH1, cV1, cD1 = coeffs[1]
    energies += [np.sum(cH1**2), np.sum(cV1**2), np.sum(cD1**2)]

    # Level 2: (cH, cV, cD)
    cH2, cV2, cD2 = coeffs[2]
    energies += [np.sum(cH2**2), np.sum(cV2**2), np.sum(cD2**2)]

    total = sum(energies) + 1e-12
    energies = [e / total for e in energies]  # normalize

    return np.array(energies, dtype=np.float32)


# ------------------------------------------------------------------
# Extended Residual Statistics
# ------------------------------------------------------------------
def residual_stats(img_gray):
    arr = img_gray.astype(np.float32).ravel()
    arr_center = arr - np.mean(arr)

    mad = np.median(np.abs(arr_center - np.median(arr_center)))
    p5 = np.percentile(arr_center, 5)
    p95 = np.percentile(arr_center, 95)
    ent = entropy(np.histogram(arr_center, bins=32)[0] + 1e-12)

    return np.array([
        np.mean(arr_center),
        np.std(arr_center),
        skew(arr_center),
        kurtosis(arr_center),
        mad,
        p5,
        p95,
        ent
    ], dtype=np.float32)


# ------------------------------------------------------------------
# Collect Image Paths
# ------------------------------------------------------------------
def collect_paths(processed_root):
    paths = []
    root = os.path.join(processed_root, "residual")

    for scanner in sorted(os.listdir(root)):
        scanner_dir = os.path.join(root, scanner)
        if not os.path.isdir(scanner_dir):
            continue

        for dpi in sorted(os.listdir(scanner_dir)):
            dpi_dir = os.path.join(scanner_dir, dpi)
            if not os.path.isdir(dpi_dir):
                continue

            for fname in sorted(os.listdir(dpi_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    paths.append((os.path.join(dpi_dir, fname), scanner))
    return paths


# ------------------------------------------------------------------
# Main Feature Extraction
# ------------------------------------------------------------------
def main(processed_root, out_prefix):
    entries = collect_paths(processed_root)
    print(f"Images found: {len(entries)}")

    scanners = sorted({scanner for _, scanner in entries})
    label_map = {i: s for i, s in enumerate(scanners)}
    inv_label = {s: i for i, s in label_map.items()}

    features = []
    labels = []
    csv_rows = []

    for path, scanner_name in tqdm(entries, desc="Extracting upgraded features"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("\n⚠ Warning: Could not read:", path)
            continue

        feat_lbp = lbp_histogram(img)
        feat_fft = fft_radial_bands(img, n_bands=5)
        feat_wave = wavelet_energies(img)
        feat_stats = residual_stats(img)

        feat = np.concatenate([feat_lbp, feat_fft, feat_wave, feat_stats])
        features.append(feat)
        labels.append(inv_label[scanner_name])

        csv_rows.append([path, inv_label[scanner_name], scanner_name] + feat.tolist())

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    np.save(out_prefix + "_features.npy", features)
    np.save(out_prefix + "_labels.npy", labels)

    # CSV header
    header = (
        ["filepath", "label", "scanner_name"]
        + [f"lbp_{i}" for i in range(LBP_P + 2)]
        + [f"fft_band_{i}" for i in range(5)]
        + [f"wavelet_{i}" for i in range(6)]
        + ["mean", "std", "skew", "kurt", "mad", "p5", "p95", "entropy"]
    )

    with open(out_prefix + "_features.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)

    with open(out_prefix + "_label_map.json", "w", encoding="utf-8") as jf:
        json.dump(label_map, jf, indent=4)


    print("\n✔ Upgraded feature extraction complete!")
    print("Feature matrix shape:", features.shape)
    print("Labels shape:", labels.shape)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True, help="processed folder containing residual/")
    parser.add_argument("--out-prefix", default="features/tracefinder_v2", help="output prefix")
    args = parser.parse_args()

    main(args.processed_root, args.out_prefix)
