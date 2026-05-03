import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. SAYFA AYARLARI VE TASARIM
st.set_page_config(page_title="Ağız Kanseri Tespiti", page_icon="💜", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; background-color: #F8F7FF; }
    
    .stApp { background: linear-gradient(180deg, #E6E0FF 0%, #FFFFFF 100%); }

    .main-title { color: #4A308E; font-weight: 700; text-align: center; font-size: 32px; margin-top: -40px; }
    .sub-title { color: #7B61FF; text-align: center; font-size: 15px; margin-bottom: 25px; }

    .status-card {
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        font-weight: 600;
        margin-top: 20px;
        box-shadow: 0px 10px 25px rgba(123, 97, 255, 0.1);
        border: 1px solid rgba(255,255,255,0.5);
    }
    
    /* Kamera ve Yükleme Kutuları */
    .stCameraInput > div, .stFileUploader > div { border-radius: 20px !important; border: 2px dashed #7B61FF !important; }
    </style>
    """, unsafe_allow_html=True)

# Başlıklar
st.markdown('<p class="main-title">AĞIZ KANSERİ TESPİTİ</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Yapay zeka ile hızlı ve güvenilir analiz.</p>', unsafe_allow_html=True)

# 2. MODEL YÜKLEME
@st.cache_resource
def load_model():
    # İsmi senin belirttiğin gibi "model_unquant.tflite" yaptık
    try:
        interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Model dosyası bulunamadı! GitHub'daki ismin 'model_unquant.tflite' olduğundan emin ol. Hata: {e}")
        return None

interp = load_model()

if interp:
    input_det = interp.get_input_details()
    output_det = interp.get_output_details()

    # 3. GİRİŞ SEÇENEKLERİ
    col1, col2 = st.columns(2)
    with col1:
        img_file = st.camera_input("📸 Fotoğraf Çek")
    with col2:
        up_file = st.file_uploader("🖼️ Galeriden Yükle", type=['jpg','png','jpeg'])

    source = img_file if img_file else up_file

    if source:
        # Görüntü İşleme
        img = Image.open(source).convert('RGB')
        st.image(img, use_container_width=True, caption="Analiz Edilen Görüntü")
        
        with st.spinner('Analiz ediliyor...'):
            # Teachable Machine standart hazırlığı
            size = (input_det[0]['shape'][1], input_det[0]['shape'][2])
            img_res = img.resize(size)
            img_arr = np.array(img_res).astype(np.float32) / 127.5 - 1 # Normalizasyon (-1 to 1)
            img_arr = np.expand_dims(img_arr, axis=0)

            # Tahmin Çalıştır
            interp.set_tensor(input_det[0]['index'], img_arr)
            interp.invoke()
            preds = interp.get_tensor(output_det[0]['index'])[0]
            
            # Sonuç İndeksleri (TM Sıralaması: 0: Sağlıklı, 1: Kanser, 2: Arka Plan)
            idx = np.argmax(preds)
            conf = preds[idx] * 100

            # 4. SONUÇ EKRANI
            if idx == 2: # Arka Plan / Geçersiz
                st.markdown(f'<div style="background-color: #FFF3E0; color: #E65100;" class="status-card">❓ AĞIZ ALGILANAMADI<br><span style="font-weight:400;">Lütfen kamerayı ağız içine daha yakın ve net tutun. (%{conf:.1f})</span></div>', unsafe_allow_html=True)
            
            elif idx == 1: # Kanser
                st.markdown(f'<div style="background-color: #FFEBEE; color: #C62828;" class="status-card">🚨 RİSK TESPİT EDİLDİ<br><span style="font-weight:400;">Anormal doku saptandı. Lütfen bir uzmana danışın. (%{conf:.1f})</span></div>', unsafe_allow_html=True)
            
            else: # Sağlıklı
                st.markdown(f'<div style="background-color: #E8F5E9; color: #2E7D32;" class="status-card">✅ SAĞLIKLI GÖRÜNÜYOR<br><span style="font-weight:400;">Şu an için endişe verici bir durum saptanmadı. (%{conf:.1f})</span></div>', unsafe_allow_html=True)

            # Detaylı Veri (Geliştirici Notu)
            with st.expander("Analiz Detaylarını Gör"):
                st.write(f"**Sağlıklı:** %{preds[0]*100:.1f}")
                st.write(f"**Kanserli:** %{preds[1]*100:.1f}")
                st.write(f"**Arka Plan:** %{preds[2]*100:.1f}")
