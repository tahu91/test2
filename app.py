import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Cek apakah TensorFlow terinstall
try:
    import tensorflow as tf
except ImportError:
    st.error("Error: TensorFlow not installed. Please add `tensorflow` to requirements.txt.")
    st.stop()

# Load model dengan error handling
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('best_cnn_model.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# UI
st.title("🐱 vs 🐶 Classifier")
uploaded_file = st.file_uploader("Upload image...", type=["jpg", "png"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file).resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        result = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
        
        st.image(img, caption="Uploaded Image", width=200)
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
