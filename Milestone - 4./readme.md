## Scanner Identification Web Interface (Streamlit UI)

### Overview
To make the scanner identification system usable in practice, a simple web-based user interface was developed using **Streamlit**. This interface allows users to upload a scanned document image and instantly receive the predicted scanner device.

The UI is designed to closely follow the same preprocessing and feature extraction steps used during model training to ensure consistent and reliable predictions.

---

### User Workflow
1. The user uploads a scanned document image (`.jpg`, `.png`, or `.tif`).
2. The system preprocesses the image using the **same preprocessing pipeline used during training**.
3. **LBP features** are extracted from the processed image.
4. A trained **SVM model** predicts the scanner device.
5. The uploaded image and predicted scanner brand are displayed on the screen.

---

### Why Consistent Preprocessing Is Important
During testing, it was observed that even small differences in image preprocessing can change the prediction when using classical machine learning models such as **LBP + SVM**.  
To avoid this issue, the Streamlit UI does **not reimplement preprocessing steps manually**.

Instead:
- The uploaded image is temporarily saved to disk.
- The exact same `preprocessing()` function used during training is applied.
- This guarantees that the extracted features match the model’s expectations.

This approach ensures that predictions made through the UI match offline predictions.

---

### Handling TIFF Images
Scanned documents are often stored in TIFF format and may contain:
- High resolution (e.g., 300 DPI, 150 DPI)
- Higher bit depth (16-bit images)

The UI properly handles these cases by:
- Converting images to a compatible format
- Downscaling images using appropriate interpolation
- Ensuring correct visualization without affecting model accuracy

Although the displayed image may appear resized, the prediction quality remains unaffected.

---

### Model Integration
- The trained **SVM classifier** is loaded using `joblib`.
- The model expects **LBP feature vectors**, not raw image pixels.
- The UI strictly follows this requirement to maintain prediction accuracy.

---

### Output
After processing, the UI displays:
- The uploaded scanned image
- The predicted scanner device (e.g., Canon, HP, Epson)

This allows quick and intuitive verification of scanner source without manual analysis.

---