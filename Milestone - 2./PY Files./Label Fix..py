import pandas as pd

df = pd.read_csv("features_with_prnu.csv")

df["scanner_model"] = df["scanner_label"].str.split("-").str[0]
df = df.drop(columns=["scanner_label"])

df.to_csv("features_prnu_final.csv", index=False)

print("✅ PRNU dataset ready")
print(df["scanner_model"].value_counts())
