import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# 1. SAYFA AYARLARI VE MOBİL UYUMLU CSS
st.set_page_config(page_title="Tanı Sistemi", page_icon="💜", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@400;500;700&display=swap');
    
    /* Genel Arka Plan */
    .stApp {
        background: linear-gradient(180deg, #F0E6FF 0%, #FFFFFF 100%);
        font-family: 'Ubuntu', sans-serif;
    }

    /* Başlık Alanı */
    .header-container {
        text-align: center;
        padding: 20px 0;
        background-color: #4A308E;
        margin: -60px -20px 30px -20px;
        border-radius: 0 0 30px 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-title { color: white; font-weight: 700; font-size: 24px; margin: 0; }
    .sub-title { color: #D1C4E9; font-size: 13px; margin-top: 5px; }

    /* Kart Yapısı */
    .status-card {
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        font-weight: 600;
        margin: 20px 0;
        border: none;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    
    /* Kamera ve Dosya Yükleyiciyi Güzelleştirme */
    .stCameraInput > div { border-radius: 20px !important; overflow: hidden; border: 3px solid #7B61FF !important; }
    .stFileUploader section { border-radius: 20px !important; border: 2px dashed #7B61FF !important; background: white !important; }
    
    /* Butonları Gizle/Düzenle */
    button[kind="secondary"] { border-radius: 15px !important; background-color: white !important; color: #4A308E !important; border: 1px solid #4A308E !important; }
    </style>
    """, unsafe_allow_html=True)

# Başlık Paneli
st.markdown("""
    <div class="header-container">
        <p class="main-title">AĞIZ KANSERİ ANALİZİ</p>
        <p class="sub-title">Yapay Zeka Destekli Erken Tanı Sistemi</p>
    </div>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME
@st.cache_resource
def load_model():
    model_path = "model_unquant.tflite"
    if not os.path.exists(model_path):
        st.error("Model dosyası bulunamadı!")
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Model Hatası: {e}")
        return None

interp = load_model()

if interp:
    input_det = interp.get_input_details()
    output_det = interp.get_output_details()

    # Giriş Metotları
    tab1, tab2 = st.tabs(["📸 Kamera Kullan", "🖼️ Galeri"])
    
    with tab1:
        img_file = st.camera_input("Ağız içini net bir şekilde çekin")
    with tab2:
        up_file = st.file_uploader("Veya bir fotoğraf yükleyin", type=['jpg','png','jpeg'])

    source = img_file if img_file else up_file

    if source:
        img = Image.open(source).convert('RGB')
        st.image(img, use_container_width=True, caption="Yüklenen Görsel")
        
        with st.spinner('Görüntü analiz ediliyor...'):
            # Preprocessing
            size = (input_det[0]['shape'][1], input_det[0]['shape'][2])
            img_res = img.resize(size)
            img_arr = np.array(img_res).astype(np.float32) / 127.5 - 1
            img_arr = np.expand_dims(img_arr, axis=0)

            # Inference
            interp.set_tensor(input_det[0]['index'], img_arr)
            interp.invoke()
            preds = interp.get_tensor(output_det[0]['index'])[0]
            
            idx = np.argmax(preds)
            conf = preds[idx] * 100

            # 3. SONUÇ EKRANI (TAM TASARIM)
            if idx == 2: # Arka Plan
                st.markdown(f'<div style="background-color: #FFF3E0; color: #E65100;" class="status-card">⚠️ ANALİZ YAPILAMADI<br><span style="font-weight:400; font-size:14px;">Görüntü net değil veya ağız içi algılanamadı. Lütfen daha yakından çekin.</span></div>', unsafe_allow_html=True)
            
            elif idx == 1: # Kanser
                st.markdown(f'<div style="background-color: #FFEBEE; color: #C62828;" class="status-card">🚨 RİSK SAPTANDI (%{conf:.1f})<br><span style="font-weight:400; font-size:14px;">Şüpheli doku bulundu. En kısa sürede bir hekime danışmanız önerilir.</span></div>', unsafe_allow_html=True)
            
            else: # Sağlıklı
                st.markdown(f'<div style="background-color: #E8F5E9; color: #2E7D32;" class="status-card">✅ SAĞLIKLI GÖRÜNÜYOR (%{conf:.1f})<br><span style="font-weight:400; font-size:14px;">Belirgin bir risk saptanmadı. Düzenli kontrollere devam edin.</span></div>', unsafe_allow_html=True)

            # Detaylı Veriler (Kullanıcı isterse bakar)
            with st.expander("Gelişmiş Veri Analizi"):
                st.progress(float(preds[0]))
                st.write(f"Sağlıklı: %{preds[0]*100:.1f}")
                st.progress(float(preds[1]))
                st.write(f"Kanser: %{preds[1]*100:.1f}")
