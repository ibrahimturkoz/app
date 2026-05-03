import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Sayfa Ayarları ve Logo
st.set_page_config(page_title="Ağız Kanseri Tespiti", page_icon="💜", layout="centered")

# Tasarımı Güzelleştiren Özel CSS (Mor Tema)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #FDFBFF;
    }
    
    .stApp {
        background: linear-gradient(180deg, #E6E0FF 0%, #FFFFFF 100%);
    }

    .main-title {
        color: #4A308E;
        font-weight: 700;
        text-align: center;
        font-size: 28px;
        margin-bottom: 5px;
    }

    .sub-title {
        color: #7B61FF;
        text-align: center;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Kart ve Buton Tasarımları */
    div.stButton > button {
        background-color: white;
        color: #4A308E;
        border: 2px solid #E6E0FF;
        border-radius: 20px;
        height: 80px;
        width: 100%;
        font-weight: 600;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
        transition: 0.3s;
    }
    
    div.stButton > button:hover {
        border-color: #7B61FF;
        background-color: #F8F7FF;
    }

    /* Uyarı Kutuları Tasarımı */
    .result-box {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Başlık ve Logo Alanı
st.markdown('<p class="main-title">AĞIZ KANSERİ TESPİTİ</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ağız içi görüntülerinizi analiz ederek risk taşıyan bölgeleri tespit edin.</p>', unsafe_allow_html=True)

# Modeli Yükle
@st.cache_resource
def load_my_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_my_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Giriş Seçenekleri (Kart yapısı)
col1, col2 = st.columns(2)
with col1:
    img_file = st.camera_input("📷 Kamera")
with col2:
    uploaded_file = st.file_uploader("🖼️ Galeri", type=['jpg', 'png', 'jpeg'])

source = img_file if img_file is not None else uploaded_file

if source is not None:
    img = Image.open(source).convert('RGB')
    st.image(img, use_column_width=True)
    
    # Model Hazırlığı
    target_size = (input_details[0]['shape'][1], input_details[0]['shape'][2])
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    p_index = np.argmax(prediction[0])
    confidence = prediction[0][p_index] * 100

    # Tasarımdaki gibi Sonuç Ekranları
    if p_index == 1: # Risk Durumu
        st.error(f"🚨 **KANSER TESPİTİ YAPILDI** \n\n Riskli bir lezyon tespit edildi (%{confidence:.1f}). Lütfen bir uzmana danışınız.")
    else: # Sağlıklı Durumu
        st.success(f"✅ **SAĞLIKLI TESPİTİ YAPILDI** \n\n Ağız taramanızda anormal bir lezyon tespit edilmedi (%{confidence:.1f}).")
