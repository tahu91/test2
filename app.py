# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Cat vs Dog Classifier - No Size Limit",
    page_icon="🐾",
    layout="centered"
)

# --- Fungsi Preprocessing ---
def preprocess_image(image, img_size=(160, 160)):
    """
    Preprocessing gambar untuk model:
    1. Konversi ke tensor jika belum
    2. Resize ke target size
    3. Normalisasi pixel [0, 1]
    """
    # Handle berbagai tipe input
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError("Format gambar tidak didukung")
    
    # Konversi ke tensor dan preprocessing
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    
    return image.numpy()  # Kembalikan numpy array

# --- Fungsi Tampilan Gambar yang Aman ---
def safe_display_image(image, caption, use_column_width=True):
    """Menampilkan gambar dari berbagai format input"""
    try:
        # Jika input adalah PIL Image
        if isinstance(image, Image.Image):
            st.image(image, caption=caption, use_column_width=use_column_width)
            return
        
        # Jika input adalah tensor
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        
        # Handle numpy array
        if isinstance(image, np.ndarray):
            # Normalisasi jika perlu
            if image.dtype == np.float32:
                image = np.clip(image, 0, 1)
                image = (image * 255).astype(np.uint8)
            
            st.image(image, caption=caption, use_column_width=use_column_width)
    except Exception as e:
        st.error(f"Gagal menampilkan gambar: {str(e)}")

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = 'cats_vs_dogs_mobilenetv2_final.h5'
    try:
        if not os.path.exists(model_path):
            st.error(f"File model tidak ditemukan di: {model_path}")
            st.info("Pastikan file model ada di direktori yang sama dengan script")
            return None
            
        model = tf.keras.models.load_model(model_path)
        st.sidebar.success("Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

# --- Main App ---
def main():
    st.title("🐱 vs 🐶 Image Classifier - Unlimited Size")
    st.markdown("""
    Upload gambar kucing atau anjing dalam ukuran berapapun!
    """)

    # Sidebar
    with st.sidebar:
        st.header("Pengaturan")
        show_confidence = st.checkbox("Tampilkan Visualisasi Confidence", True)
        debug_mode = st.checkbox("Mode Debug", False)

    # Upload gambar tanpa limit size
    uploaded_file = st.file_uploader(
        "Pilih gambar...", 
        type=["jpg", "jpeg", "png"],
        help="Tidak ada batasan ukuran file"
    )

    if uploaded_file is not None:
        try:
            # Load gambar
            pil_image = Image.open(uploaded_file).convert("RGB")
            
            # Tampilkan gambar asli
            col1, col2 = st.columns(2)
            with col1:
                safe_display_image(pil_image, "Gambar Asli")

            # Preprocessing
            img_array = np.array(pil_image)
            if debug_mode:
                st.write("Shape sebelum preprocessing:", img_array.shape)
                st.write("Tipe data:", img_array.dtype)

            processed_img = preprocess_image(img_array)

            if debug_mode:
                st.write("Shape setelah preprocessing:", processed_img.shape)
                st.write("Nilai pixel (contoh):", processed_img[0,0,:])

            # Prediksi
            model = load_model()
            if model is not None:
                input_tensor = np.expand_dims(processed_img, axis=0)
                prediction = model.predict(input_tensor, verbose=0)[0][0]

                # Tampilkan hasil
                with col2:
                    st.subheader("Hasil Prediksi")
                    
                    if prediction > 0.5:
                        st.success(f"🐶 Anjing (Confidence: {prediction*100:.1f}%)")
                    else:
                        st.success(f"🐱 Kucing (Confidence: {(1-prediction)*100:.1f}%)")

                    if show_confidence:
                        # Visualisasi confidence
                        fig, ax = plt.subplots(figsize=(6, 2))
                        ax.barh(['Kucing', 'Anjing'], 
                               [(1-prediction)*100, prediction*100], 
                               color=['#ff9999', '#66b3ff'])
                        ax.set_xlim(0, 100)
                        ax.set_title('Confidence Score')
                        st.pyplot(fig)

                    if debug_mode:
                        safe_display_image(processed_img, "Gambar setelah Preprocessing")

        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")
            if debug_mode:
                st.exception(e)

if __name__ == "__main__":
    main()
