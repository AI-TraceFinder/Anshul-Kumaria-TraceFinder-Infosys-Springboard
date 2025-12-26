# Milestone 2 Documentation
## Feature Engineering and Baseline Modeling

### 1. Objective
To extract handcrafted features from preprocessed document images and
train baseline machine learning models for document classification and
tampering detection.

### 2. Feature Engineering
Feature engineering was performed to capture noise, frequency, and
texture characteristics of document images.

#### 2.1 Noise-Based Features
- Mean
- Standard deviation
- Skewness
- Kurtosis

These features help identify scanner-specific noise patterns.

#### 2.2 Frequency Domain Features (FFT)
Fast Fourier Transform was applied to extract radial frequency energy
distribution, which captures structural differences in documents.

#### 2.3 Texture Features (LBP)
Local Binary Pattern (LBP) was used to capture texture variations in
document images, which is effective for distinguishing document sources
and detecting tampering.

All extracted features are stored in:
features/features.csv

### 3. Baseline Modeling
The following baseline classifiers were trained using the extracted
features:

- Random Forest Classifier
- Support Vector Machine (SVM)

### 4. Results
Both baseline models achieved accuracy greater than 95%.

| Model | Accuracy |
|------|----------|
| Random Forest | 98.32% |
| SVM | 98.32% |

Detailed results are stored in:
results/baseline_results.txt

### 5. Conclusion
The combined use of noise, frequency, and texture features along with
baseline machine learning models produced strong performance, providing
a solid foundation for further improvements in future milestones.
