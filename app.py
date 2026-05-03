import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("Ağız Kanseri Teşhis Sistemi")
st.write("Kameranızı açın ve analizi başlatın.")

# Modeli Yükle
@st.cache_resource
def load_my_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_my_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kamera Girdisi
img_file = st.camera_input("Fotoğraf Çek veya Kamerayı Başlat")

if img_file is not None:
    # Görüntüyü Hazırla
    img = Image.open(img_file)
    img = img.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Tahmin yap
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    # Sonucu Göster
    p_index = np.argmax(prediction[0])
    confidence = prediction[0][p_index] * 100

    if p_index == 1:
        st.error(f"RİSK TESPİT EDİLDİ! (Güven: %{confidence:.2f})")
    else:
        st.success(f"SAĞLIKLI GÖRÜNÜYOR. (Güven: %{confidence:.2f})")