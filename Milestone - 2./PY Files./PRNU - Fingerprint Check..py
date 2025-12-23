import numpy as np

# -----------------------------
# LOAD PRNU FINGERPRINT
# -----------------------------
FILE_PATH = FILE_PATH = r"C:\Users\M.varshith reddy\Downloads\Milestone-2\prnu_fingerprints\Canon120.npy"
  # ✅ FIXED PATH

fp = np.load(FILE_PATH)

# -----------------------------
# CHECK 1: ARRAY TYPE & SHAPE
# -----------------------------
print("Array type:", type(fp))
print("Array shape:", fp.shape)

# -----------------------------
# CHECK 2: DATA TYPE
# -----------------------------
print("Data type:", fp.dtype)

# -----------------------------
# CHECK 3: STATISTICAL PROPERTIES
# -----------------------------
print("Min value:", fp.min())
print("Max value:", fp.max())
print("Mean value:", fp.mean())
print("Standard deviation:", fp.std())

# -----------------------------
# CHECK 4: SAMPLE VALUES
# -----------------------------
print("\nSample values (5x5 block):")
print(fp[:5, :5])
