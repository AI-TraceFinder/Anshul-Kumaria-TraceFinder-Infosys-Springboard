import streamlit as st
import joblib
from PIL import Image
import numpy as np
import os
from skimage.feature import local_binary_pattern
import cv2


def lbp(img, p=8, r=1):
    img = img.squeeze()
    img_uint8 = (img * 255).astype(np.uint8)

    lbp_img = local_binary_pattern(img_uint8, p, r, method='uniform')

    hist, _ = np.histogram(lbp_img.ravel(), bins=np.arange(0, p + 3), range=(0, p + 2))

    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)

    return hist


def preprocessing_size(img, size=(224, 224)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize(img):
    return img / 255.0


def load_tiff(path):
    return np.array(Image.open(path))


st.set_page_config(page_title="AItracker", layout="wide")

st.title(" Scanned Device Identifier")
st.write("Upload a scanned document image to identify the scanner device.")
st.markdown("---")

model = joblib.load("svm_scanner.pkl")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader(" Input Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png", "tif"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        
        img_arr = np.array(pil_img)
        
        st.image(pil_img, caption=f"Original Image: {uploaded_file.name}", use_container_width=True)
        
        with st.expander(" Image Details"):
            st.write(f"Dimensions: {img_arr.shape[1]} x {img_arr.shape[0]} pixels")
            st.write(f"Channels: {img_arr.shape[2] if len(img_arr.shape) == 3 else 1}")
            st.write(f"File size:  {uploaded_file.size / 1024:.1f} KB")

with col2:
    st.subheader(" Processing & Results")
    
    if uploaded_file is not None:
        with st.status("Processing image ", expanded=True) as status:
            img = img_arr
            
            st.write("1. Converting to grayscale.")
            if len(img.shape) == 3:
                img = gray_scale(img)
            
            st.write("2. Normalizing pixel values.")
            if img.dtype != np.uint8:
                img = cv2.normalize(
                    img, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
            
            st.write("3. Resizing image..")
            display_img = preprocessing_size(img, (224, 224))
            
            st.write("4. Extracting features..")
            model_img = normalize(display_img)
            model_img = model_img.reshape(224, 224, 1)
            features = lbp(model_img).reshape(1, -1)
            
            st.write("5. Making prediction..")
            prediction = model.predict(features)[0]
            
            status.update(label="Processing complete!", state="complete", expanded=False)
        
        st.image(display_img, caption="Processed Image (224×224, Grayscale)", use_container_width=True)
        
        st.markdown("---")
        st.subheader(" Prediction Result")
    
        if "Canon" in prediction or "HP" in prediction or "Epson" in prediction:
            result_color = "#4CAF50"  
        else:
            result_color = "#2196F3" 
        
        st.markdown(
            f"""
            <div style="background-color:{result_color}10; padding:20px; border-radius:10px; border-left:5px solid {result_color};">
                <h3 style="color:{result_color}; margin-top:0;">Scanned with:</h3>
                <h1 style="color:{result_color}; text-align:center;">{prediction.upper()}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
    else:
        st.info("👈 Upload an image to see results here")
        st.image(
            "https://cdn-icons-png.flaticon.com/512/3767/3767084.png",
            caption="Waiting for image upload...",
            width=200
        )