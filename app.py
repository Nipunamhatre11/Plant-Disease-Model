import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to detect potential diseases using our deep learning model.")

# 1. Advanced Downloader for Large Google Drive Files
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    
    # Check if Google Drive threw a large-file warning cookie
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    # If there is a warning token, request again with the confirmation token included
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        
    # Write the actual binary chunks of the model to disk
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_my_model():
    model_path = 'final_balanced_plant_model.keras'
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading your trained model from Google Drive... This takes about 30-45 seconds."):
            file_id = '1lZGXdbs_ioN_djsvWhpT2uo-6DT6BC6M'
            download_file_from_google_drive(file_id, model_path)
            
    return tf.keras.models.load_model(model_path)

try:
    model = load_my_model()
    
    # 2. File uploader UI
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Leaf Image', use_container_width=True)
        
        st.write("🔄 Processing and predicting...")
        
        # 3. Preprocess the image to match your model's input size
        img = image.resize((256, 256)) 
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        img_array = img_array / 255.0  # Normalize if your model expects it
        
        # 4. Predict
        predictions = model.predict(img_array)
        st.success("Model evaluation complete! (Add your class labels to see the exact disease name)")
except Exception as e:
    st.error(f"Could not load the model file. Error: {e}")
