import streamlit as st
import spacy
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Prediction Demo", layout="wide")

st.title("Live Prediction & Blocking Demo")
st.markdown("Uji coba model secara langsung! Sistem akan mengklasifikasikan teks berita dan menentukan apakah konten tersebut harus **diblokir** atau **diizinkan** berdasarkan mode produktivitas.")


# 1. PENGECEKAN MODEL DI SESSION STATE
if 'trained_model' not in st.session_state or 'tfidf_vectorizer' not in st.session_state:
    st.warning("Model belum dilatih! Silakan kembali ke halaman **Train Model** untuk melatih algoritma terlebih dahulu.")
    st.stop()

model = st.session_state['trained_model']
tfidf = st.session_state['tfidf_vectorizer']
model_name = st.session_state['model_name']

st.success(f"Sistem berjalan menggunakan model: **{model_name}**")
st.write("---")


# 2. LOAD SPACY UNTUK CLEANING INPUT BARU
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    except OSError:
        st.error("Model spaCy belum diunduh. Jalankan: `python -m spacy download en_core_web_sm`")
        st.stop()

nlp = load_spacy()

def clean_single_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(tokens)


# 3. ANTARMUKA PENGGUNA (INPUT)
col_input, col_rules = st.columns([2, 1])

with col_rules:
    st.header("Productivity Rules")
    st.markdown("""
    Dalam mode **Fokus Kerja/Belajar**, sistem akan menerapkan aturan berikut:
    
    **Diizinkan (Allowed):**
    - `business` (Bisnis & Ekonomi)
    - `tech` (Teknologi)
    - `politics` (Politik/Berita Utama)
    
    **Diblokir (Blocked):**
    - `sport` (Olahraga)
    - `entertainment` (Hiburan & Gosip)
    """)

with col_input:
    st.header("Masukkan Teks")
    user_input = st.text_area(
        "Salin dan tempel judul atau isi artikel berita di sini (Gunakan Bahasa Inggris):",
        height=200,
        placeholder="Contoh: Apple launches new iPhone with advanced AI features..."
    )
    
    predict_btn = st.button("Analisis & Prediksi Konten", type="primary", use_container_width=True)

st.write("---")


# 4. PROSES PREDIKSI & VISUALISASI
if predict_btn:
    if not user_input.strip():
        st.error("Teks tidak boleh kosong. Silakan masukkan teks berita terlebih dahulu.")
    else:
        with st.spinner("Memproses teks dan melakukan klasifikasi..."):
            # 1. Preprocessing Input Text
            cleaned_input = clean_single_text(user_input)
            
            if not cleaned_input.strip():
                st.warning("Teks tidak mengandung kata bermakna setelah dibersihkan (mungkin hanya berisi angka/tanda baca/stopword).")
            else:
                # 2. Transformasi ke TF-IDF
                input_tfidf = tfidf.transform([cleaned_input])
                
                # 3. Prediksi Kategori
                prediction = model.predict(input_tfidf)[0]
                
                # 4. Probabilitas / Confidence Score
                probabilities = model.predict_proba(input_tfidf)[0]
                classes = model.classes_
                
                # Membuat DataFrame Probabilitas untuk Chart
                prob_df = pd.DataFrame({
                    'Kategori': classes,
                    'Probabilitas': probabilities * 100
                }).sort_values(by='Probabilitas', ascending=True)

        # --- MENAMPILKAN HASIL PREDIKSI KATEGORI & PROBABILITAS ---
        st.header("Hasil Analisis Sistem")
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.markdown("#### Distribusi Probabilitas Model")
            # Horizontal Bar Chart untuk Probabilitas
            fig_prob = go.Figure(go.Bar(
                x=prob_df['Probabilitas'],
                y=prob_df['Kategori'],
                orientation='h',
                text=prob_df['Probabilitas'].apply(lambda x: f"{x:.2f}%"),
                textposition='auto',
                marker_color='rgb(55, 83, 109)'
            ))
            fig_prob.update_layout(
                xaxis_title="Confidence Level (%)",
                yaxis_title="Kategori",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
        with col_res2:
            st.markdown("#### Keputusan Sistem (Productivity Shield)")
            
            # --- SIMULASI BLOCKING SYSTEM DENGAN CLEAN MINIMALIST UI ---
            blocked_categories = ['sport', 'entertainment']
            
            if prediction in blocked_categories:
                # Custom HTML/CSS untuk kotak Blocked (Clean Minimalist Red)
                blocked_html = f"""
                <div style="background-color: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-left: 5px solid #ef4444; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
                    <h3 style="margin-top: 0; color: #ef4444; display: flex; align-items: center; gap: 8px;">
                        Akses Diblokir
                    </h3>
                    <p style="font-size: 15px; margin-bottom: 12px; opacity: 0.9;">
                        <strong>Kategori Terdeteksi:</strong> <span style="background-color: rgba(239, 68, 68, 0.2); color: #ef4444; padding: 4px 8px; border-radius: 4px; font-size: 13px; text-transform: uppercase; font-weight: bold;">{prediction}</span>
                    </p>
                    <p style="margin-bottom: 0; font-size: 14px; opacity: 0.85; line-height: 1.5;">
                        Sistem mendeteksi bahwa konten ini mengandung unsur hiburan atau olahraga yang dapat mengganggu produktivitas Anda.<br><br>
                        <strong>Silakan kembali fokus pada pekerjaan Anda.</strong>
                    </p>
                </div>
                """
                st.markdown(blocked_html, unsafe_allow_html=True)
            else:
                # Custom HTML/CSS untuk kotak Allowed (Clean Minimalist Green)
                allowed_html = f"""
                <div style="background-color: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.2); border-left: 5px solid #22c55e; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
                    <h3 style="margin-top: 0; color: #22c55e; display: flex; align-items: center; gap: 8px;">
                        Akses Diizinkan
                    </h3>
                    <p style="font-size: 15px; margin-bottom: 12px; opacity: 0.9;">
                        <strong>Kategori Terdeteksi:</strong> <span style="background-color: rgba(34, 197, 94, 0.2); color: #22c55e; padding: 4px 8px; border-radius: 4px; font-size: 13px; text-transform: uppercase; font-weight: bold;">{prediction}</span>
                    </p>
                    <p style="margin-bottom: 0; font-size: 14px; opacity: 0.85; line-height: 1.5;">
                        Konten ini tergolong dalam kategori berita informatif/produktif. Anda diizinkan untuk membaca artikel ini.
                    </p>
                </div>
                """
                st.markdown(allowed_html, unsafe_allow_html=True)
                
            with st.expander("Lihat Teks yang Telah Dibersihkan (Internal System View)"):
                st.write(cleaned_input)