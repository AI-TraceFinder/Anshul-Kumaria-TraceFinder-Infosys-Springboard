# src/preprocess.py

import os
import cv2
import argparse
import numpy as np
from skimage.restoration import denoise_wavelet
from skimage import img_as_float32

def read_image_strict(path):
    """Read image robustly (handles Unicode/OneDrive long paths on Windows). Returns BGR uint8 or None."""
    try:
        arr = np.fromfile(path, dtype=np.uint8)
        if arr.size == 0:
            return None
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("[ERROR] read failed:", path, e)
        return None

def denoise_nlm_gray(img_gray, h=10):
    """OpenCV Non-local Means denoising for grayscale uint8 image."""
    return cv2.fastNlMeansDenoising(img_gray, None, h, 7, 21)

def denoise_wavelet_gray(img_gray_float):
    """
    Wavelet denoising for float image in [0,1].
    Uses channel_axis=None for grayscale images (skimage API).
    Returns float image in [0,1].
    """
    # Use channel_axis=None for single-channel images (replaces deprecated multichannel)
    den = denoise_wavelet(img_gray_float, channel_axis=None, rescale_sigma=True)
    return den

def normalize_to_uint8(img_float):
    """
    Normalize a float image (may contain negative values, e.g., residual)
    to uint8 [0..255] for saving/visualization. Preserves sign info by linear mapping.
    """
    mn = float(np.min(img_float))
    mx = float(np.max(img_float))
    if np.isclose(mx, mn):
        return np.zeros(img_float.shape, dtype=np.uint8)
    norm = (img_float - mn) / (mx - mn)
    return (np.clip(norm * 255.0, 0, 255)).astype(np.uint8)

def save_uint8(path, img):
    """Ensure directory exists and save uint8 image with cv2.imwrite."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.dtype != np.uint8:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_u8 = img
    cv2.imwrite(path, img_u8)

def preprocess_one(in_path, out_denoised_path, out_residual_path, size=(256,256), method="wavelet", nlm_h=10):
    """
    Process single file:
    - read, convert to gray
    - denoise at full resolution (wavelet or nlm)
    - residual = orig_float - denoised_float
    - resize denoised and residual (normalized) to `size`
    - save outputs (PNG)
    """
    img_color = read_image_strict(in_path)
    if img_color is None:
        print("[ERROR] Cannot read:", in_path)
        return False

    # Convert to grayscale (uint8)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Denoise on full resolution
    if method == "nlm":
        denoised_full = denoise_nlm_gray(img_gray, h=nlm_h)          # uint8
        denoised_full_float = denoised_full.astype(np.float32) / 255.0
    else:  # wavelet (default)
        img_f = img_as_float32(img_gray)                             # float [0,1]
        denoised_full_float = denoise_wavelet_gray(img_f)            # float [0,1]
        denoised_full = (np.clip(denoised_full_float * 255.0, 0, 255)).astype(np.uint8)

    # Original in float [0,1]
    orig_float = img_gray.astype(np.float32) / 255.0
    residual_full = orig_float - denoised_full_float                # signed float

    # Resize denoised (uint8) and normalized residual (uint8) to target size
    denoised_resized = cv2.resize(denoised_full, size, interpolation=cv2.INTER_AREA)
    residual_norm = normalize_to_uint8(residual_full)
    residual_resized = cv2.resize(residual_norm, size, interpolation=cv2.INTER_AREA)

    # Save results
    save_uint8(out_denoised_path, denoised_resized)
    if out_residual_path:
        save_uint8(out_residual_path, residual_resized)

    return True

def process_dataset(in_root, out_root, size=(256,256), method="wavelet", nlm_h=10, save_residual=True):
    total = 0
    failed = 0
    for scanner in sorted(os.listdir(in_root)):
        scanner_path = os.path.join(in_root, scanner)
        if not os.path.isdir(scanner_path):
            continue

        for dpi in sorted(os.listdir(scanner_path)):
            dpi_path = os.path.join(scanner_path, dpi)
            if not os.path.isdir(dpi_path):
                continue

            out_den_dir = os.path.join(out_root, "denoised", scanner, dpi)
            out_res_dir = os.path.join(out_root, "residual", scanner, dpi) if save_residual else None
            os.makedirs(out_den_dir, exist_ok=True)
            if save_residual:
                os.makedirs(out_res_dir, exist_ok=True)

            for fname in sorted(os.listdir(dpi_path)):
                if not fname.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                    continue
                in_path = os.path.join(dpi_path, fname)
                base = fname.rsplit(".", 1)[0]
                out_den = os.path.join(out_den_dir, base + ".png")
                out_res = os.path.join(out_res_dir, base + ".png") if save_residual else None

                ok = preprocess_one(in_path, out_den, out_res, size=size, method=method, nlm_h=nlm_h)
                total += 1
                if not ok:
                    failed += 1
                    print("[WARN] Failed:", in_path)
                if total % 100 == 0:
                    print(f"[INFO] Processed {total} images so far...")

    print(f"[SUMMARY] Done. Processed: {total}, Failed: {failed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="preprocess.py", description="Preprocess dataset: denoise + residual + resize")
    parser.add_argument("--dataset-root", required=True, help="Path to raw Dataset root")
    parser.add_argument("--out-root", required=True, help="Output root (will contain denoised/ and residual/)")
    parser.add_argument("--size", nargs=2, type=int, default=[256,256], help="Target size (width height) e.g. 256 256")
    parser.add_argument("--method", choices=["wavelet","nlm"], default="wavelet", help="Denoise method (wavelet or nlm)")
    parser.add_argument("--nlm-h", type=int, default=10, help="h parameter for NLM denoising (if method=nlm)")
    parser.add_argument("--no-residual", action="store_true", help="Do not save residual images (save only denoised)")
    args = parser.parse_args()

    size_tuple = (args.size[0], args.size[1])
    process_dataset(args.dataset_root, args.out_root, size=size_tuple, method=args.method, nlm_h=args.nlm_h, save_residual=not args.no_residual)
    print("[OK] Preprocessing (denoise + residual) finished.")

