import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# Sayfa Ayarları
st.set_page_config(page_title="Ağız Kanseri Analizi", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #F8F7FF; }
    .main-title { color: #4A308E; font-size: 30px; font-weight: bold; text-align: center; }
    .status-card { padding: 20px; border-radius: 15px; text-align: center; font-weight: bold; margin-top: 20px; border: 2px solid #EEE; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">🦷 TANI SİSTEMİ ANALİZİ</p>', unsafe_allow_html=True)

# MODEL YÜKLEME (Cache'i temizleyerek)
def load_interp():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interp = load_interp()
input_det = interp.get_input_details()
output_det = interp.get_output_details()

# GİRİŞ ALANLARI
up_file = st.file_uploader("Analiz edilecek fotoğrafı buraya sürükleyin", type=['jpg','png','jpeg'], key="file_uploader")

if up_file:
    # Görüntüyü oku ve ekrana bas
    img = Image.open(up_file).convert('RGB')
    st.image(img, caption="Yüklenen Görsel", use_container_width=True)
    
    with st.spinner('Analiz yapılıyor, lütfen bekleyin...'):
        # Görüntü hazırlama
        size = (input_det[0]['shape'][1], input_det[0]['shape'][2])
        img_res = img.resize(size)
        img_arr = np.array(img_res).astype(np.float32)
        
        # NORMALİZASYON: Eğer modelin Teachable Machine ise /255 yeterlidir. 
        img_arr = img_arr / 255.0 
        img_arr = np.expand_dims(img_arr, axis=0)

        # MODELİ ÇALIŞTIR
        interp.set_tensor(input_det[0]['index'], img_arr)
        interp.invoke()
        preds = interp.get_tensor(output_det[0]['index'])[0]
        
        # HATA AYIKLAMA (Burası çok önemli!)
        # Modelin gerçek ham çıktılarını görelim
        st.write(f"🔍 **Model Ham Analiz Verisi:** Sınıf 0: `{preds[0]:.4f}`, Sınıf 1: `{preds[1]:.4f}`")

        idx = np.argmax(preds)
        conf = preds[idx] * 100

        # SONUÇ MANTIĞI
        # Eğer hep sağlıklı çıkıyorsa, modelin kanser ve sağlıklı indeksleri ters olabilir.
        # Genelde 0=Sağlıklı, 1=Kanserdir. Ama tam tersi de olabilir.
        
        if idx == 1: # Bu kısmı 0 yaparak da test edebilirsin
            st.markdown(f'<div style="background-color: #FFEBEE; color: #C62828;" class="status-card">🚨 RİSK TESPİT EDİLDİ (%{conf:.2f})<br>Model yüksek olasılıkla lezyon saptadı.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color: #E8F5E9; color: #2E7D32;" class="status-card">✅ SAĞLIKLI GÖRÜNÜYOR (%{conf:.2f})<br>Anormal bir doku saptanmadı.</div>', unsafe_allow_html=True)

    st.button("Yeniden Analiz Et")
