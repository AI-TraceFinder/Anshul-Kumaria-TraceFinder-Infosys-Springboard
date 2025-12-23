import pandas as pd

df = pd.read_csv("features.csv")

print("\nColumns in CSV:", df.columns.tolist())

numeric_cols = ["mean_intensity", "std_intensity", "fft_feature", "lbp_mean", "lbp_std", "noise_variance"]

print("\nChecking numeric feature summary:\n")
print(df[numeric_cols].describe())

