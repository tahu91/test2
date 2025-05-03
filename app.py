# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# --- Fungsi Preprocessing (dari utils.py) ---
def preprocess_image(image, img_size=(160, 160)):
    """
    Preprocessing gambar yang konsisten dengan pelatihan model:
    1. Resize ke 160x160
    2. Normalisasi pixel [0, 1]
    """
    # Convert PIL Image to TensorFlow tensor if needed
    if isinstance(image, np.ndarray):
        image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()

# --- Augmentasi (opsional) ---
def apply_augmentations(image):
    """Augmentasi real-time jika diperlukan"""
    if st.sidebar.checkbox("Gunakan Augmentasi"):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
    return image

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('cats_vs_dogs_mobilenetv2_final.h5')
        st.sidebar.success("Model berhasil dimuat!")
        return model
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model: {str(e)}")
        return None

# --- Main App ---
def main():
    st.title("🐱 vs 🐶 Image Classifier")
    st.markdown("""
    Upload gambar kucing atau anjing, dan model akan memprediksi jenisnya!
    """)

    # Sidebar
    with st.sidebar:
        st.header("Pengaturan")
        show_confidence = st.checkbox("Tampilkan Visualisasi Confidence", True)
        # debug_mode = st.checkbox("Mode Debug", False)

    # Upload gambar
    uploaded_file = st.file_uploader(
        "Pilih gambar...", 
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )

    if uploaded_file is not None:
        try:
            # Load dan tampilkan gambar
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Gambar Asli", use_column_width=True)

            # Preprocessing
            img_array = np.array(image)
            if debug_mode:
                st.write("Shape sebelum preprocessing:", img_array.shape)
                st.write("Tipe data:", img_array.dtype)

            processed_img = preprocess_image(img_array)
            processed_img = apply_augmentations(processed_img)

            if debug_mode:
                st.write("Shape setelah preprocessing:", processed_img.shape)
                st.write("Nilai pixel (contoh):", processed_img[0,0,:])

            # Prediksi
            model = load_model()
            if model is not None:
                input_tensor = np.expand_dims(processed_img, axis=0)
                prediction = model.predict(input_tensor)[0][0]

                # Tampilkan hasil
                with col2:
                    st.subheader("Hasil Prediksi")
                    
                    if prediction > 0.5:
                        st.success(f"🐶 Anjing (Confidence: {prediction*100:.1f}%)")
                        class_label = "Anjing"
                    else:
                        st.success(f"🐱 Kucing (Confidence: {(1-prediction)*100:.1f}%)")
                        class_label = "Kucing"

                    if show_confidence:
                        # Visualisasi confidence
                        fig, ax = plt.subplots(figsize=(6, 2))
                        ax.barh(['Kucing', 'Anjing'], 
                               [(1-prediction)*100, prediction*100], 
                               color=['#ff9999', '#66b3ff'])
                        ax.set_xlim(0, 100)
                        ax.set_title('Confidence Score')
                        st.pyplot(fig)

                        # Tampilkan gambar yang sudah diproses untuk 
                        if debug_mode:
                            st.image(processed_img, caption="Gambar setelah Preprocessing", clamp=True)

        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")
            if debug_mode:
                st.exception(e)

if __name__ == "__main__":
    main()
