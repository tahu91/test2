import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Konfigurasi awal
MODEL_PATH = 'best_cnn_model.h5'
CLASS_NAMES = {0: 'Cat 🐱', 1: 'Dog 🐶'}

# Deteksi TensorFlow
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

# Load model dan tentukan ukuran target gambar
@st.cache_resource
def load_safe_model():
    global TARGET_SIZE, is_mobilenet

    try:
        model = load_model(MODEL_PATH)
        TARGET_SIZE = (150, 150)  # untuk model custom CNN
        is_mobilenet = False
        st.sidebar.success("Model loaded successfully")
        return model
    except Exception as e:
        st.sidebar.error(f"Primary load failed: {str(e)}")
        try:
            model = load_model(MODEL_PATH, compile=False)
            TARGET_SIZE = (150, 150)
            is_mobilenet = False
            st.sidebar.warning("Model loaded without compilation")
            return model
        except:
            st.sidebar.warning("Using MobileNetV2 as fallback")
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            TARGET_SIZE = (224, 224)  # ukuran untuk MobileNetV2
            is_mobilenet = True
            return MobileNetV2(weights='imagenet')

# Preprocessing image
def preprocess_image(image):
    try:
        image = image.convert('RGB')
        image = image.resize(TARGET_SIZE)
        img_array = np.array(image)

        if is_mobilenet:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            img_array = preprocess_input(img_array)
        else:
            img_array = img_array / 255.0

        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

# Fungsi utama Streamlit
def main():
    st.title("🐱 vs 🐶 Classifier")
    st.markdown("""
    Upload an image of a cat or dog for classification.  
    The model expects 150x150 or 224x224 RGB images depending on model.
    """)

    model = load_safe_model()

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Original Image", use_column_width=True)

            with st.spinner('Analyzing image...'):
                processed_img = preprocess_image(img)

                if processed_img is not None:
                    prediction = model.predict(processed_img)

                    if is_mobilenet:
                        # Decode predictions (1000-class ImageNet)
                        from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
                        decoded = decode_predictions(prediction, top=1)[0][0]
                        label = decoded[1]
                        confidence = decoded[2] * 100
                        st.success(f"**Prediction:** {label} ({confidence:.1f}%)")
                        with st.expander("Technical Details"):
                            st.json({
                                "Decoded": decoded,
                                "Image Shape": processed_img.shape,
                                "Model Input Shape": model.input_shape
                            })
                    else:
                        confidence = float(prediction[0][0] if prediction.shape[1] > 1 else prediction[0])
                        class_idx = 1 if confidence > 0.5 else 0
                        confidence_pct = max(confidence, 1-confidence) * 100
                        st.success(f"""
                        **Prediction:** {CLASS_NAMES[class_idx]}  
                        **Confidence:** {confidence_pct:.1f}%
                        """)
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
