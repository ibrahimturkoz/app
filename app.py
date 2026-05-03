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
    .stApp { background: linear-gradient(180deg, #F0E6FF 0%, #FFFFFF 100%); font-family: 'Ubuntu', sans-serif; }
    .header-container { text-align: center; padding: 20px 0; background-color: #4A308E; margin: -60px -20px 30px -20px; border-radius: 0 0 30px 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    .main-title { color: white; font-weight: 700; font-size: 24px; margin: 0; }
    .sub-title { color: #D1C4E9; font-size: 13px; margin-top: 5px; }
    .status-card { padding: 25px; border-radius: 20px; text-align: center; font-weight: 600; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.08); }
    .stCameraInput > div { border-radius: 20px !important; border: 3px solid #7B61FF !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="header-container"><p class="main-title">AĞIZ KANSERİ ANALİZİ</p><p class="sub-title">Yapay Zeka Destekli Erken Tanı Sistemi</p></div>', unsafe_allow_html=True)

# 2. MODEL YÜKLEME
@st.cache_resource
def load_model():
    model_path = "model_unquant.tflite"
    if not os.path.exists(model_path):
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except:
        return None

interp = load_model()

if interp:
    input_det = interp.get_input_details()
    output_det = interp.get_output_details()

    img_file = st.camera_input("Ağız içini net bir şekilde çekin")
    up_file = st.file_uploader("Veya fotoğraf yükleyin", type=['jpg','png','jpeg'])
    source = img_file if img_file else up_file

    if source:
        img = Image.open(source).convert('RGB')
        # Loglardaki uyarıyı giderdik: use_container_width yerine width='stretch'
        st.image(img, width='stretch', caption="Yüklenen Görsel")
        
        with st.spinner('Analiz ediliyor...'):
            size = (input_det[0]['shape'][1], input_det[0]['shape'][2])
            img_res = img.resize(size)
            img_arr = np.array(img_res).astype(np.float32) / 127.5 - 1
            img_arr = np.expand_dims(img_arr, axis=0)

            interp.set_tensor(input_det[0]['index'], img_arr)
            interp.invoke()
            preds = interp.get_tensor(output_det[0]['index'])[0]
            
            # INDEX ERROR ÇÖZÜMÜ: Çıktı sayısını kontrol ediyoruz
            num_classes = len(preds)

            if num_classes == 1:
                # EĞER MODEL TEK ÇIKTILIK İSE (Sigmoid)
                score = preds[0]
                if score > 0.5:
                    idx, conf = 1, score * 100
                else:
                    idx, conf = 0, (1 - score) * 100
            else:
                # EĞER MODEL ÇOK ÇIKTILIK İSE (Softmax)
                idx = np.argmax(preds)
                conf = preds[idx] * 100

            # 3. SONUÇ EKRANI
            if idx == 2 and num_classes > 2: # Arka Plan (Eğer 3 sınıf varsa)
                st.markdown(f'<div style="background-color: #FFF3E0; color: #E65100;" class="status-card">⚠️ ANALİZ YAPILAMADI<br><span style="font-weight:400; font-size:14px;">Görüntü net değil. Lütfen daha yakından çekin.</span></div>', unsafe_allow_html=True)
            elif idx == 1: # Kanser
                st.markdown(f'<div style="background-color: #FFEBEE; color: #C62828;" class="status-card">🚨 RİSK SAPTANDI (%{conf:.1f})<br><span style="font-weight:400; font-size:14px;">Şüpheli doku bulundu. Lütfen bir hekime danışın.</span></div>', unsafe_allow_html=True)
            else: # Sağlıklı
                st.markdown(f'<div style="background-color: #E8F5E9; color: #2E7D32;" class="status-card">✅ SAĞLIKLI GÖRÜNÜYOR (%{conf:.1f})<br><span style="font-weight:400; font-size:14px;">Belirgin bir risk saptanmadı.</span></div>', unsafe_allow_html=True)
else:
    st.error("Model dosyası yüklenemedi. Lütfen GitHub'da 'model_unquant.tflite' dosyasının olduğundan emin olun.")
