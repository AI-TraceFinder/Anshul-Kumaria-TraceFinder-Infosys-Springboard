import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# Create plots folder if not exists
# -------------------------------
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------------------
# Load extracted features
# -------------------------------
df = pd.read_csv("features.csv")

print("Feature dataset loaded successfully!")
print(df.head())
print("\nTotal samples:", len(df))

sns.set(style="whitegrid")

# ---------------------------------------
# 1. Mean Intensity per Scanner
# ---------------------------------------
plt.figure(figsize=(14, 6))
sns.boxplot(x="scanner_label", y="mean_intensity", data=df)
plt.xticks(rotation=45, ha="right")
plt.title("Mean Intensity Distribution per Scanner")
plt.tight_layout()

# Save plot
plt.savefig(os.path.join(PLOT_DIR, "mean_intensity_per_scanner.png"))
plt.show()
plt.close()

# ---------------------------------------
# 2. Standard Deviation per Scanner
# ---------------------------------------
plt.figure(figsize=(14, 6))
sns.boxplot(x="scanner_label", y="std_intensity", data=df)
plt.xticks(rotation=45, ha="right")
plt.title("Standard Deviation per Scanner")
plt.xlabel("Scanner")
plt.ylabel("Standard Deviation")
plt.tight_layout()

# Save plot
plt.savefig(os.path.join(PLOT_DIR, "std_intensity_per_scanner.png"))
plt.show()
plt.close()

# ---------------------------------------
# 3. Mean Intensity per Dataset
# ---------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x="dataset", y="mean_intensity", data=df)
plt.title("Mean Intensity Across Datasets")
plt.tight_layout()

# Save plot
plt.savefig(os.path.join(PLOT_DIR, "mean_intensity_per_dataset.png"))
plt.show()
plt.close()

# ---------------------------------------
# 4. Correlation Heatmap
# ---------------------------------------
plt.figure(figsize=(6, 4))
corr = df[["mean_intensity", "std_intensity"]].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Extracted Features")
plt.tight_layout()

# Save plot
plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"))
plt.show()
plt.close()

print("\nAll plots generated and saved successfully in 'plots/' folder!")
