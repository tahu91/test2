import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('best_cnn_model.h5')

# Judul Aplikasi
st.title("🚀 Aplikasi Prediksi Gambar Kucing vs Anjing")

# Upload Gambar
uploaded_file = st.file_uploader(
    "Upload gambar kucing atau anjing...", 
    type=["jpg", "jpeg", "png"]
)

# Fungsi Prediksi
def predict(image):
    img = Image.open(image).resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Anjing 🐶" if prediction > 0.5 else "Kucing 🐱"

# Tampilkan Hasil
if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Gambar yang Diupload", width=200)
    with col2:
        result = predict(uploaded_file)
        st.success(f"Prediksi: {result}")

# Panduan Penggunaan
st.markdown("""
### Cara Menggunakan:
1. Upload gambar kucing atau anjing (format JPG/PNG).
2. Aplikasi akan menampilkan prediksi.
3. Untuk dataset custom, ganti model di `best_cnn_model.h5`.
""")
