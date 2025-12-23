import os
import cv2
import numpy as np

from prnu_utils import extract_prnu_residual
from prnu_correlation import normalized_correlation

# ----------------------------
# PATHS
# ----------------------------
FINGERPRINT_PATH = "prnu_fingerprints"
TEST_PATH = r"C:\Users\M.varshith reddy\Downloads\MileStone-1\Processed\Official"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ----------------------------
# LOAD SCANNER-LEVEL FINGERPRINTS
# ----------------------------
fingerprints = {}

for file in os.listdir(FINGERPRINT_PATH):
    if file.endswith(".npy"):
        scanner_model = file.replace(".npy", "")
        fingerprints[scanner_model] = np.load(
            os.path.join(FINGERPRINT_PATH, file)
        )

print("Loaded fingerprints:", list(fingerprints.keys()))


# ----------------------------
# PRNU CORRELATION TEST
# ----------------------------
correct = 0
total = 0

for scanner_folder in os.listdir(TEST_PATH):
    scanner_path = os.path.join(TEST_PATH, scanner_folder)

    if not os.path.isdir(scanner_path):
        continue

    # 🔥 Extract scanner MODEL
    true_scanner = scanner_folder.split("-")[0]

    for img_file in os.listdir(scanner_path):
        if not img_file.lower().endswith(IMAGE_EXTENSIONS):
            continue

        img_path = os.path.join(scanner_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        prnu_test = extract_prnu_residual(img)

        scores = {}
        for model, fingerprint in fingerprints.items():
            scores[model] = normalized_correlation(prnu_test, fingerprint)

        predicted_scanner = max(scores, key=scores.get)

        if predicted_scanner == true_scanner:
            correct += 1

        total += 1

accuracy = correct / total if total > 0 else 0
print("\n🔥 FINAL PRNU CORRELATION ACCURACY:", accuracy)
