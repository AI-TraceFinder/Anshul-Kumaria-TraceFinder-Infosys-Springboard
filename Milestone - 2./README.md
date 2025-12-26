# Milestone 2 – Feature Engineering & Baseline Modeling

## Objective
To extract meaningful features from preprocessed document images and train
baseline machine learning models for document source and tampering classification.

## Feature Engineering
- Noise statistics (mean, std, skewness, kurtosis)
- FFT-based radial energy features
- Local Binary Pattern (LBP) texture features

All features are stored in:
features/features.csv

## Baseline Models
- Random Forest Classifier
- Support Vector Machine (SVM)

## Results
- Random Forest Accuracy: 98.32%
- SVM Accuracy: 98.32%

Detailed results are available in:
results/baseline_results.txt

## How to Run
python feature_extraction.py  
python baseline_model.py
