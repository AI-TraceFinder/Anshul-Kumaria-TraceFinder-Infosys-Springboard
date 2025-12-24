# src/inventory.py
import os
import csv
from PIL import Image
import argparse

def build_inventory(dataset_root):
    rows = []
    for scanner in sorted(os.listdir(dataset_root)):
        scanner_path = os.path.join(dataset_root, scanner)
        if not os.path.isdir(scanner_path):
            continue

        for dpi_folder in sorted(os.listdir(scanner_path)):
            dpi_path = os.path.join(scanner_path, dpi_folder)
            if not os.path.isdir(dpi_path):
                continue

            for fname in sorted(os.listdir(dpi_path)):
                if not fname.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                    continue

                file_path = os.path.join(dpi_path, fname)
                try:
                    img = Image.open(file_path)
                    width, height = img.size
                    mode = img.mode  # e.g., "RGB" or "L"
                    fmt = img.format
                    filesize = os.path.getsize(file_path)
                    rows.append([file_path, scanner, dpi_folder, fname, fmt, width, height, mode, filesize])
                except Exception as e:
                    print(f"[ERROR] Cannot read: {file_path}  --> {e}")

    return rows

def save_csv(rows, out_file):
    header = ["filepath","scanner","dpi","filename","format","width","height","mode","filesize_bytes"]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[OK] Inventory written to: {out_file}  (rows: {len(rows)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset inventory CSV")
    parser.add_argument("--dataset-root", required=True, help="Path to Dataset folder")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    rows = build_inventory(args.dataset_root)
    save_csv(rows, args.out)
