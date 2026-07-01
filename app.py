import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request

st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to detect potential diseases using our deep learning model.")

# 1. Download and load your trained model from Google Drive
@st.cache_resource
def load_my_model():
    model_path = 'final_balanced_plant_model.keras'
    
    # If the model isn't downloaded to the server yet, grab it from Drive
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive... This might take a minute."):
            # Using your exact Google Drive file ID
            file_id = '1lZGXdbs_ioN_djsvWhpT2uo-6DT6BC6M' 
            url = f'https://drive.google.com/uc?export=download&id={file_id}'
            urllib.request.urlretrieve(url, model_path)
            
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
