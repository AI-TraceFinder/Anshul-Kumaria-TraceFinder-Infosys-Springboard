# Milestone 2: Feature Engineering & Baseline Modeling

## Overview

Milestone 2 focuses on transforming preprocessed residual images into meaningful numerical representations and developing baseline machine learning models for scanner identification. This milestone establishes a performance benchmark using traditional ML classifiers before transitioning to deep learning approaches.

## Objectives

- Extract discriminative features from scanner noise patterns
- Evaluate traditional ML classifiers
- Establish baseline performance metrics
- Analyze strengths and limitations of hand-crafted features

## Feature Engineering Pipeline

### Input Data
Processed residual images from Milestone 1:
```
processed/residual/Scanner/DPI/*.png
```

### Feature Extraction Techniques

#### 1. **Local Binary Patterns (LBP)** - Texture Features
- Captures micro-texture variations in scanner noise
- Configuration: Uniform LBP with 8 neighbors, radius 1
- Detects local neighborhood transitions

#### 2. **FFT Radial Energy Bands** - Frequency Domain Features
- Transforms residuals into frequency domain
- Detects periodic structures and banding artifacts
- **V2 Enhancement**: 5 radial frequency bands (improved from 3)

#### 3. **Statistical Noise Features**
Comprehensive noise characterization including:
- Mean, Standard Deviation
- Skewness, Kurtosis
- 5th & 95th Percentiles
- Median Absolute Deviation
- Wavelet Energy Coefficients (5 subbands)

## Generated Feature Dataset

### Output Files
```
tracefinder_v2_features.npy       # Numerical feature matrix
tracefinder_v2_labels.npy         # Class labels (0-6)
tracefinder_v2_features.csv       # Human-readable feature table
tracefinder_v2_label_map.json     # Label to scanner model mapping
```

## Baseline Machine Learning Models

### Algorithms Implemented
1. Logistic Regression
2. Support Vector Machine (RBF kernel)
3. Random Forest
4. Hyperparameter-tuned models (Randomized Search CV)
5. Stacking Ensemble (Random Forest + SVM + XGBoost)

### Training Configuration
- **Normalization**: StandardScaler
- **Split**: Stratified 80-20 train-test
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Confusion matrices

## Performance Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~72.8% |
| Random Forest | ~71.4% |
| SVM (RBF) | ~68.2% |
| **Stacking Ensemble (Optimized)** | **~76.4%** |

### Key Insights
- Stacking ensemble achieved best performance at **76.4% accuracy**
- Results reveal inherent limitations of hand-crafted features
- Confirms need for CNN-based feature learning (Milestone 3)

## Project Structure

### Scripts Developed
```
src/
├── inventory.py              # Dataset metadata extraction
├── analysis.py               # Statistical analysis
├── preprocess.py             # Residual image generation
├── extract_features.py       # Basic feature extraction
├── extract_features_v2.py    # Enhanced feature extraction (FINAL)
├── train_baseline.py         # Initial ML training
└── train_optimized.py        # Hyperparameter tuning + stacking
```

### Model Outputs
```
models/baseline_v2/
├── baseline_results_summary.csv
├── feature_importances_RandomForest.csv
├── final_summary.csv
├── classification_report_stacking.csv
├── confusion_matrix_stacking.csv
└── pipeline_stacking.joblib
```

## Key Achievements

1. Comprehensive scanner-noise feature engineering  
2. Multiple ML models trained and evaluated  
3. Strong baseline established (~76% accuracy)  
4. Feature importance analysis completed  
5. Identified limitations of manual feature engineering  
6. System prepared for deep learning (Milestone 3)

## Feature Importance Insights

Feature importance analysis revealed which noise patterns are most discriminative for scanner identification, providing valuable insights into the forensic signatures of different scanner models.

**Milestone 2** successfully bridges preprocessing and deep learning by creating a robust classical ML benchmark for forensic scanner identification.
