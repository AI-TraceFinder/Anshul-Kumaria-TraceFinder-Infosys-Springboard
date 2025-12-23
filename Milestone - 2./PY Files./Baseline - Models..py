import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# ===============================
# STEP 1: LOAD FINAL DATASET
# ===============================

df = pd.read_csv("features_prnu_final.csv")

print("Columns in dataset:")
print(df.columns)


# ===============================
# STEP 2: SELECT FEATURES & LABEL
# ===============================

# Target column (scanner model)
y = df["scanner_model"].values

# Feature columns only (drop metadata + label)
X = df.drop(columns=[
    "dataset",
    "image_name",
    "scanner_model"
]).values

print("\nFeature matrix shape:", X.shape)
print("Label vector shape:", y.shape)


# ===============================
# STEP 3: ENCODE LABELS
# ===============================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# ===============================
# STEP 4: FEATURE SCALING
# ===============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ===============================
# STEP 5: TRAIN-TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)


# ===============================
# STEP 6: MODEL 1 - LOGISTIC REGRESSION
# ===============================

lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\n🔹 Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_pred))


# ===============================
# STEP 7: MODEL 2 - SVM (BEST BASELINE)
# ===============================

svm = SVC(kernel="rbf", C=10, gamma="scale")
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("🔹 SVM Accuracy:",
      accuracy_score(y_test, svm_pred))


# ===============================
# STEP 8: MODEL 3 - RANDOM FOREST
# ===============================

rf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("🔹 Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))


# ===============================
# STEP 9: CONFUSION MATRIX & REPORT
# ===============================

print("\n📊 CONFUSION MATRIX (Random Forest)")
print(confusion_matrix(y_test, rf_pred))

print("\n📄 CLASSIFICATION REPORT")
print(classification_report(y_test, rf_pred))
