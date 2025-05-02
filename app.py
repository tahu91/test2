import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import sys

# Configuration
MODEL_PATH = 'best_cnn_model.h5'
TARGET_SIZE = (150, 150)
CLASS_NAMES = {0: 'Cat 🐱', 1: 'Dog 🐶'}

# Check TensorFlow installation
try:
    tf_version = tf.__version__
    st.sidebar.success(f"TensorFlow {tf_version} detected")
except ImportError:
    st.error("""
    TensorFlow not installed. Please add to requirements.txt:
    ```
    tensorflow>=2.10.0
    ```
    """)
    st.stop()

# Enhanced model loading with multiple fallbacks
@st.cache_resource
def load_safe_model():
    try:
        # Attempt standard load
        model = load_model(MODEL_PATH)
        st.sidebar.success("Model loaded successfully")
        return model
    except Exception as e:
        st.sidebar.error(f"Primary load failed: {str(e)}")
        
        try:
            # Fallback 1: Load with compile=False
            model = load_model(MODEL_PATH, compile=False)
            st.sidebar.warning("Model loaded without compilation")
            return model
        except:
            # Fallback 2: Use MobileNet as emergency backup
            st.sidebar.warning("Using MobileNetV2 as fallback")
            from tensorflow.keras.applications import MobileNetV2
            return MobileNetV2(weights='imagenet')

# Image preprocessing pipeline
def preprocess_image(image):
    try:
        # Convert to RGB if not already
        image = image.convert('RGB')
        
        # Resize and normalize
        image = image.resize(TARGET_SIZE)
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

# Main app
def main():
    st.title("🐱 vs 🐶 Classifier")
    st.markdown("""
    Upload an image of a cat or dog for classification.
    The model expects 150x150 RGB images.
    """)
    
    # Load model
    model = load_safe_model()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        try:
            # Display original image
            img = Image.open(uploaded_file)
            st.image(img, caption="Original Image", use_column_width=True)
            
            # Preprocess and predict
            with st.spinner('Analyzing image...'):
                processed_img = preprocess_image(img)
                if processed_img is not None:
                    prediction = model.predict(processed_img)
                    confidence = float(prediction[0][0] if prediction.shape[1] > 1 else prediction[0])
                    
                    # Determine result
                    class_idx = 1 if confidence > 0.5 else 0
                    confidence_pct = max(confidence, 1-confidence) * 100
                    
                    # Display results
                    st.success(f"""
                    **Prediction:** {CLASS_NAMES[class_idx]}  
                    **Confidence:** {confidence_pct:.1f}%
                    """)
                    
                    # Debug info (expandable)
                    with st.expander("Technical Details"):
                        st.json({
                            "Raw Output": float(confidence),
                            "Image Shape": processed_img.shape,
                            "Model Input Shape": model.input_shape
                        })
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
