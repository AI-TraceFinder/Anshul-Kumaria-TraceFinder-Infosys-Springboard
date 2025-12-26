import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.filters import gaussian

# ================= CONFIG =================
RAW_ROOT = "raw"                     # Input dataset folder
OUT_IMG = "processed/images"         # Output processed images
OUT_NOISE = "processed/noise_maps"   # Output noise residuals
META_DIR = "metadata"                # Metadata CSV folder

TARGET_SIZE = (512, 512)             # Resize to 512x512
GRAYSCALE = True                     # Convert to grayscale
BATCH_SIZE = 20                      # Prevent crash
# ==========================================

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# Create folders if missing
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_NOISE, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)


def list_images(root):
    """Scan raw/ folder and list all images with scanner name."""
    items = []
    for folder in os.listdir(root):
        folder_path = os.path.join(root, folder)
        if os.path.isdir(folder_path):
            for dp, _, files in os.walk(folder_path):
                for f in files:
                    if f.lower().endswith(EXTS):
                        items.append((folder, os.path.join(dp, f)))
    return items


def preprocess_image(im):
    """Convert to grayscale, resize, normalize."""
    if GRAYSCALE:
        im = im.convert("L")
    im = im.resize(TARGET_SIZE)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr


def extract_noise(arr):
    """Residual image = input - denoised(input)."""
    den = gaussian(arr, sigma=1)
    return arr - den


def save_arr(arr, path):
    """Rescale 0–255 and save as PNG."""
    mn, mx = arr.min(), arr.max()
    arr2 = (arr - mn) / (mx - mn + 1e-8)
    Image.fromarray((arr2 * 255).astype("uint8")).save(path)


def main():
    print(">>> Started preprocessing... scanning raw folder...")

    images = list_images(RAW_ROOT)
    print(">>> Found", len(images), "images.")

    labels = []
    stats = []

    # Process in batches to avoid memory issues
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i+BATCH_SIZE]
        print(f"\n>>> Processing batch {i} → {i + len(batch) - 1}")

        for scanner, img_path in tqdm(batch):
            try:
                base = os.path.splitext(os.path.basename(img_path))[0]

                out_proc = f"{OUT_IMG}/{scanner}__{base}.png"
                out_noise = f"{OUT_NOISE}/{scanner}__{base}_noise.png"

                im = Image.open(img_path)

                # Preprocess and extract noise
                arr = preprocess_image(im)
                noise = extract_noise(arr)

                save_arr(arr, out_proc)
                save_arr(noise, out_noise)

                labels.append([scanner, img_path, out_proc, out_noise])
                stats.append([img_path, "ok"])

            except Exception as e:
                print("!! ERROR processing", img_path, ":", e)
                stats.append([img_path, str(e)])

    # Save metadata
    pd.DataFrame(labels, columns=["scanner", "original", "processed", "noise"]).to_csv(
        f"{META_DIR}/dataset_labels.csv", index=False
    )
    pd.DataFrame(stats, columns=["file", "status"]).to_csv(
        f"{META_DIR}/image_stats.csv", index=False
    )

    print("\n>>> Preprocessing completed successfully.")
    print(">>> Processed images saved in:", OUT_IMG)
    print(">>> Noise maps saved in:", OUT_NOISE)
    print(">>> CSV metadata saved in:", META_DIR)


if __name__ == "__main__":
    main()
