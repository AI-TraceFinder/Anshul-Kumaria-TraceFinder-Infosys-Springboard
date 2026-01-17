## Overview
In Week 5 and Week 6, Deep learning approaches were explored for scanner device identification using scanned document images. The objective was to evaluate whether Convolutional Neural Networks (CNNs) trained on raw images can outperform classical hand-crafted feature-based methods.

---

## Week 5: CNN Model Development and Training

### Objective
To build and train a CNN model on raw scanned document images and analyze its performance under limited data conditions. Additionally, image augmentation techniques were applied to improve generalization.

---
### 1. CNN Trained from Scratch

#### Approach
- A custom CNN architecture was designed and trained from scratch using grayscale scanned document images.
- Images were resized to a fixed resolution.
- The dataset contained approximately **1000+ images**, distributed across multiple scanner brands.
- Basic image augmentation (rotation and brightness variation) was applied.

#### Results
- **Training Accuracy:** ~55%
- **Validation Accuracy:** ~40%
- **Test Accuracy:** ~40%

#### Observation
The model showed **underfitting**, meaning it failed to learn sufficiently from the data. This behavior is expected because:
- CNNs require large datasets to learn robust features.
- Scanner identification relies on subtle noise and texture patterns.
- The dataset size was relatively small for training a CNN from scratch.


---

### 2. Transfer Learning using MobileNetV2

#### Objective
To overcome underfitting, a **pre-trained CNN model (MobileNetV2)** was used. Transfer learning allows the model to leverage features learned from large-scale datasets (ImageNet) and adapt them to the scanner identification task.

---

### Model Configuration
- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Input Size:** 224 × 224 × 3
- **Classifier Head:** Global Average Pooling + Dense layers
- **Fine-Tuning:** Only upper layers were trained
- **Optimizer:** Adam
- **Learning Rate:** Reduced during fine-tuning

---

### Data Augmentation
To improve generalization and prevent overfitting:
- Small rotations (±2–3 degrees)
- Slight brightness variation
- No aggressive transformations (to preserve scanner noise)

---

### Results (After Transfer Learning)
- **Training Accuracy:** ~82%
- **Validation Accuracy:** ~84%
- **Test Accuracy:** ~86+%

#### Observation
- Transfer learning significantly improved performance.
- The model generalized better to unseen data.
- MobileNetV2 captured scanner-specific texture and noise patterns more effectively than the CNN trained from scratch.


---

## Week 6: Model Evaluation and Explainability

### Objective
To evaluate the CNN model using appropriate performance metrics and apply explainability techniques to understand how the model identifies scanner-specific patterns.

---

### 1. Model Evaluation Metrics

The following metrics were used:
- **Accuracy:** Overall classification performance
- **F1-Score:** Balance between precision and recall
- **Confusion Matrix:** Class-wise prediction analysis

These metrics provided a comprehensive understanding of model behavior and misclassification patterns across scanner brands.

---

### 2. Explainability using Grad-CAM

#### Why Explainability?
CNNs are often considered black-box models. In forensic applications, it is essential to verify that the model focuses on **scanner-specific artifacts** rather than document content such as text.

---

### Grad-CAM (Gradient-weighted Class Activation Mapping)
Grad-CAM was applied to visualize which regions of the scanned image influenced the model’s prediction.

#### Observations
- The model primarily focused on **background regions**, **paper texture**, and **noise patterns**.
- Minimal attention was observed on textual content.
- This confirms that the CNN learned **scanner-specific characteristics**, not semantic information.

---

## Final Summary

| Model Approach | Accuracy |
|--------------|----------|
| CNN (from scratch) | ~40% |
| MobileNetV2 (transfer learning) | ~86+% |
| LBP + SVM (classical ML) | ~92+% |

- CNN trained from scratch underperformed due to limited data.
- Transfer learning significantly improved accuracy and generalization.
- Classical LBP + SVM remains the best-performing method for this dataset.
- CNN + Grad-CAM provides valuable insights and validation of scanner-specific learning.

---