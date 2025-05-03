# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide"
)

# Load and cache the model
@st.cache_resource
def load_model():
    # This should match your model architecture from the notebook
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Load your trained weights (you'll need to save them first)
    # model.load_weights('path_to_your_weights.h5')
    
    return model

model = load_model()

# Preprocessing function (matches your notebook)
def preprocess_image(image):
    image = tf.image.resize(image, (160, 160))
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Streamlit app
def main():
    st.title("🐱 vs 🐶 Image Classifier")
    st.write("Upload an image of a cat or dog, and the model will predict which one it is!")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess and predict
        img_array = np.array(image)
        img_array = preprocess_image(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        prediction = model.predict(img_array)
        
        # Display results
        st.subheader("Prediction Results")
        if prediction[0][0] > 0.5:
            st.success(f"🐶 It's a dog! (Confidence: {prediction[0][0]*100:.2f}%)")
        else:
            st.success(f"🐱 It's a cat! (Confidence: {(1-prediction[0][0])*100:.2f}%)")
            
        # Show confidence bar
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
        st.progress(float(confidence))
        
        # Add some explanation
        st.markdown("""
        ### How It Works
        - The model uses a convolutional neural network (CNN) trained on thousands of cat and dog images
        - It analyzes visual patterns in the image to make its prediction
        - Confidence score shows how certain the model is about its prediction
        """)

if __name__ == "__main__":
    main()
