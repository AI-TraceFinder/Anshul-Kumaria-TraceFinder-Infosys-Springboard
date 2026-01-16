# AI TraceFinder – Milestone 4  
## Deployment & Final Report

### Project Title
AI TraceFinder: Scanner Image Verification System

### Milestone Objective
The objective of Milestone 4 is to deploy the trained deep learning model using a simple user interface, test the system on real scanned images, log predictions, and prepare final documentation with results and screenshots.

---

## Features Implemented
- Streamlit-based web UI for image upload
- Prediction of scanned image type (Official / Wiki)
- Confidence score display
- Prediction logging with downloadable CSV file
- Testing on multiple sample images
- Screenshot-based result validation

---

## Technologies Used
- Python 3.11
- TensorFlow / Keras
- Streamlit
- NumPy, Pandas
- PIL (Image Processing)
- VS Code (Local Deployment)
- Google Colab (Model Testing)

---

## Folder Structure
Milestone_4/
│
├── app.py
├── requirements.txt
├── tracefinder_model.h5
├── prediction_log.csv
├── AI_TraceFinder_M4(Testing).ipynb
├── test1.png
├── test2.png
├── test3.png
└── result_screenshots/

---

## How to Run the Application (Local)
```bash
pip install -r requirements.txt
python -m streamlit run app.py

The application will open in the browser at:
http://localhost:8501

##Output
Uploaded image preview
Predicted image class
Confidence score
Downloadable prediction log

##Notes
The model file (tracefinder_model.h5) is used locally due to GitHub file size limitations.
Screenshots of results and UI are available in the result_screenshots folder.
