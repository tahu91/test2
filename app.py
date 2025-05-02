import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('cnn_model.h5')
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def preprocess_image(img):
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

st.title("Image Classification with CNN")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    processed = preprocess_image(image)
    prediction = model.predict(processed)
    label = class_names[np.argmax(prediction)]
    st.write(f"**Prediction:** {label}")
