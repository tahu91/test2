import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# Load model yang telah dilatih (misalnya, cnn_model.h5)
model = tf.keras.models.load_model('cnn_model.h5')

# Daftar kelas CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Fungsi untuk memproses gambar sebelum diberikan ke model
def preprocess_image(img):
    # Pastikan gambar diubah menjadi RGB jika memiliki 4 saluran
    img = img.convert('RGB')  # Convert image to RGB (removes alpha channel if exists)
    img = img.resize((32, 32))  # Resize gambar ke ukuran yang diinginkan oleh model
    img_array = np.array(img) / 255.0  # Normalize image
    return np.expand_dims(img_array, axis=0)

# Judul aplikasi
st.title("Image Classification with CNN")
st.write("Upload an image to predict its class using a trained CNN model.")

# Input gambar dari pengguna
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Baca gambar yang diupload
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocessing gambar
    processed_image = preprocess_image(image)

    # Prediksi kelas gambar
    prediction = model.predict(processed_image)
    label = class_names[np.argmax(prediction)]

    # Tampilkan hasil prediksi
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {100 * np.max(prediction):.2f}%")
