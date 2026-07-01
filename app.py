import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to detect potential diseases using our deep learning model.")

# List of all 38 classes trained by the model
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# 1. Load your trained model from Hugging Face Hub
@st.cache_resource
def load_my_model():
    model_path = hf_hub_download(
        repo_id="nipunamhatre11/plant-disease-model",
        filename="final_balanced_plant_model.keras"
    )
    return tf.keras.models.load_model(model_path)

try:
    model = load_my_model()

    # 2. File uploader UI
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Leaf Image', use_container_width=True)

        with st.spinner("🔄 Processing and analyzing image..."):
            # 3. Preprocess the image to match your model's input size
            img = image.resize((300, 300))  # matches training IMG_SIZE
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # 4. Predict
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_index] * 100
            
            # 5. Map and Clean Class Label
            raw_label = class_labels[predicted_index]
            # Formats "Tomato___Septoria_leaf_spot" to "Tomato - Septoria leaf spot"
            clean_label = raw_label.replace("___", " - ").replace("_", " ")

        # 6. Display Results
        st.success("🎉 Analysis complete!")
        
        # Display nicely in a metric or large text style
        st.markdown(f"### **Prediction:** `{clean_label}`")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

except Exception as e:
    st.error(f"Could not load the model file. Error: {e}")
