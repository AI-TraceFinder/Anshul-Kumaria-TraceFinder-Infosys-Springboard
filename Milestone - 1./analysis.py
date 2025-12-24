# src/analysis.py
import pandas as pd

df = pd.read_csv("dataset_inventory.csv")

print("\nTOTAL IMAGES:", len(df))

print("\nIMAGES PER SCANNER:")
print(df['scanner'].value_counts().to_string())

print("\nIMAGES PER DPI:")
print(df['dpi'].value_counts().to_string())

print("\nIMAGE FORMATS:")
print(df['format'].value_counts().to_string())

print("\nIMAGE MODES (RGB / L):")
print(df['mode'].value_counts().to_string())

print("\nAVERAGE DIMENSIONS PER SCANNER:")
print(df.groupby('scanner')[['width','height']].mean().round(1).to_string())

print("\nFIRST 10 ROWS:")
print(df.head(10).to_string(index=False))
