# Scanner Brand Identification Using Hand-Crafted Features

This project aims to identify scanner brands from scanned document images by focusing on hardware-specific artifacts rather than the actual document content. The work described here summarizes the feature extraction and model training stages carried out during Weeks 3 and 4 of the project.

---

## Week 3: Hand-Crafted Feature Extraction

In this phase, traditional image processing techniques were used to extract features that reflect scanner-specific characteristics. The goal was to capture artifacts introduced by scanner hardware—such as texture variations, sensor noise, and frequency patterns—while minimizing the influence of text, layout, or document semantics.

### Extracted Features

#### 1. Local Binary Patterns (LBP)

Local Binary Patterns were used to capture fine-grained texture variations present in scanned images. These variations often arise due to differences in scanner sensors, optics, and internal processing pipelines.

- Uniform LBP configuration  
- Parameters: **P = 8**, **R = 1**  
- A normalized LBP histogram was computed for each image  

LBP is well suited for this task because it is sensitive to micro-texture differences while remaining relatively robust to illumination changes.

#### 2. Noise Pattern Analysis

To analyze scanner-specific noise, document content was suppressed using smoothing filters. The remaining noise residuals primarily represent sensor noise and illumination non-uniformities introduced during the scanning process.

Since these noise patterns are largely independent of document content, they provide useful cues for distinguishing between different scanner devices.

#### 3. Frequency-Domain Features (FFT)

Frequency-domain analysis was performed using the Fast Fourier Transform (FFT).

- Used to identify periodic artifacts introduced by scanner hardware  
- Captures frequency-based noise characteristics  

Such artifacts tend to be consistent for a given scanner model and can therefore act as reliable identifiers.

### Feature Selection Rationale

Scanner identification relies on subtle, hardware-induced artifacts rather than visible document content. Texture, noise, and frequency-based features were selected because they directly reflect scanner behavior and imaging characteristics.

---

## Week 4: Model Training and Evaluation

### Models Used

The following baseline machine learning models were trained using the extracted hand-crafted features:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  

The dataset was split into training and testing sets to evaluate model performance on unseen data.

### Evaluation Metrics

Model performance was evaluated using:

- Classification accuracy  
- Confusion matrix  

### Results and Model Selection

Among the evaluated models, the Support Vector Machine (SVM) achieved the best performance.

- **SVM (RBF kernel) accuracy: 91%**

### Why SVM Was Chosen

SVM was selected as the final classifier based on both performance and practical considerations:

- It effectively models non-linear decision boundaries, which are common in texture-based features such as LBP  
- It generalizes well on medium-sized datasets  
- It is relatively robust to noise, which is important for scanner identification  
- It is widely used in document image forensics research  

Based on these factors and its superior accuracy, SVM was chosen as the final model for scanner brand classification.

---

## Summary

This stage of the project shows that carefully designed hand-crafted features can successfully capture scanner-specific artifacts. When combined with an appropriate classifier such as SVM, these features enable accurate scanner brand identification from scanned document images.