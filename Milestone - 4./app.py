import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

# Load model
model = tf.keras.models.load_model("tracefinder_model.h5")
IMG_SIZE = (224, 224)

st.title("AI TraceFinder – Scanner Image Verification")
st.write("Upload a scanned document image to classify it.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

def predict(image):
    # Convert PIL image to RGB (VERY IMPORTANT)
    image = image.convert("RGB")

    # Resize image
    image = image.resize(IMG_SIZE)

    # Convert to numpy array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "Wiki Image"
        confidence = float(prediction)
    else:
        label = "Official Image"
        confidence = float(1 - prediction)

    return label, confidence


if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict(image)

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}")

    # Log prediction
    log = {
        "Time": datetime.now(),
        "Image Name": uploaded_file.name,
        "Prediction": label,
        "Confidence": confidence
    }

    df = pd.DataFrame([log])
    try:
        old = pd.read_csv("prediction_log.csv")
        df = pd.concat([old, df])
    except:
        pass

    df.to_csv("prediction_log.csv", index=False)

    st.download_button(
        "Download Prediction Log",
        df.to_csv(index=False),
        "prediction_log.csv",
        "text/csv"
    )
