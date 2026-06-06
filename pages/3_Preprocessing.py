import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import subprocess
import sys

st.set_page_config(page_title="Preprocessing & TF-IDF", layout="wide")

st.title("Text Preprocessing & Feature Extraction")
st.markdown("Tahap ini memproses teks mentah menggunakan **spaCy** (Lemmatization, Stopword Removal) dan mengubahnya menjadi vektor numerik menggunakan **TF-IDF**.")

# 1. LOAD DATA & MODEL
@st.cache_data
def load_data():
    return pd.read_csv("dataset/bbc-news-data.csv", sep='\t')

try:
    df = load_data()
except FileNotFoundError:
    st.error("File dataset tidak ditemukan. Pastikan 'bbc-news-data.csv' ada di folder 'dataset/'.")
    st.stop()

# --- SCRIPT INSTALASI PAKSA UNTUK STREAMLIT CLOUD ---
@st.cache_resource
def load_spacy():
    try:
        # Coba import normal
        import en_core_web_sm
        return en_core_web_sm.load(disable=['parser', 'ner'])
    except ModuleNotFoundError:
        # Jika gagal, paksa server Streamlit install via terminal Python internal
        st.warning("Sedang menginstal model spaCy di server Cloud. Proses ini hanya terjadi satu kali, mohon tunggu sebentar...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"])
        
        # Setelah terinstal, import ulang
        import en_core_web_sm
        return en_core_web_sm.load(disable=['parser', 'ner'])

nlp = load_spacy()
# ---------------------------------------------------

# 2. FEATURE SELECTION
st.header("Feature Selection")
st.markdown("Pilih bagian teks mana yang ingin digunakan sebagai fitur utama untuk melatih model klasifikasi:")

feature_choice = st.radio(
    "Pilih Fitur Teks:",
    options=["Title Only", "Content Only", "Title + Content"],
    horizontal=True
)

if feature_choice == "Title Only":
    df['text_to_process'] = df['title']
elif feature_choice == "Content Only":
    df['text_to_process'] = df['content']
else:
    df['text_to_process'] = df['title'] + " " + df['content']

# 3. TEXT CLEANING (SPACY)
st.header("Text Cleaning (spaCy)")

# Fungsi cleaning dengan cache agar tidak memproses ulang jika input tidak berubah
@st.cache_data(show_spinner=False)
def clean_text_data(texts):
    cleaned_docs = []
    # Menggunakan nlp.pipe untuk batch processing (lebih efisien)
    for doc in nlp.pipe(texts, batch_size=100):
        # Ambil lemma, jadikan huruf kecil, buang stopword, tanda baca, dan angka
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        cleaned_docs.append(" ".join(tokens))
    return cleaned_docs

# Sediakan preview agar user tahu apa yang terjadi sebelum memproses 2225 baris
preview_size = st.slider("Jumlah data yang di-preview:", 5, 20, 5)
preview_df = df.head(preview_size).copy()

with st.spinner("Membersihkan sampel teks..."):
    preview_df['cleaned_text'] = clean_text_data(preview_df['text_to_process'])

st.markdown("#### Preview: Raw vs Cleaned Text")
st.dataframe(preview_df[['category', 'text_to_process', 'cleaned_text']], use_container_width=True)

st.info("**Catatan:** Membersihkan seluruh dataset (2000+ baris) dengan spaCy akan memakan waktu. Tentukan ukuran Data Uji (Test Size) di bawah, lalu klik tombol untuk mengeksekusi *pipeline* secara penuh.")

# --- PENAMBAHAN SLIDER TRAIN-TEST SPLIT ---
st.header("Train - Test Split Configuration")
test_size_ratio = st.slider(
    "Tentukan persentase Data Uji (Test Size):", 
    min_value=0.1, 
    max_value=0.5, 
    value=0.2, 
    step=0.05,
    help="Contoh: 0.2 berarti 80% data untuk Train dan 20% data untuk Test."
)

st.write(f"Rasio pembagian saat ini: **{int((1 - test_size_ratio) * 100)}% Train** / **{int(test_size_ratio * 100)}% Test**")

# 4. FULL PIPELINE EXECUTION
if st.button("Preprocess", type="primary"):
    with st.spinner("Langkah 1: Membersihkan seluruh teks menggunakan spaCy..."):
        df['cleaned_text'] = clean_text_data(df['text_to_process'].tolist())
        
    with st.spinner("Langkah 2: Membagi data (Train/Test) dan mengekstrak fitur TF-IDF..."):
        st.header("Hasil Train - Test Split & TF-IDF")
        
        X = df['cleaned_text']
        y = df['category']
        
        # --- MENGGUNAKAN NILAI DARI SLIDER ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42, stratify=y)
        
        col1, col2 = st.columns(2)
        col1.success(f"**Data Latih (Train):** {len(X_train)} baris")
        col2.success(f"**Data Uji (Test):** {len(X_test)} baris")
        
        # --- TF-IDF VECTORIZATION ---
        # Membatasi 5000 fitur (kata) terpenting agar model tidak terlalu berat dan menghindari curse of dimensionality
        tfidf = TfidfVectorizer(max_features=5000) 
        
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        st.write(f"Dimensi Matriks TF-IDF (Train): `{X_train_tfidf.shape}`")
        
        # --- VISUALISASI TOP WORDS TF-IDF ---
        st.markdown("#### 📊 Top 20 Kata Berdasarkan Bobot TF-IDF (Data Train)")
        
        # Mengambil rata-rata bobot TF-IDF untuk setiap kata
        avg_tfidf_scores = np.mean(X_train_tfidf.toarray(), axis=0)
        feature_names = tfidf.get_feature_names_out()
        
        # Menggabungkan menjadi DataFrame untuk Plotly
        tfidf_df = pd.DataFrame({'Word': feature_names, 'TF-IDF Score': avg_tfidf_scores})
        tfidf_df = tfidf_df.sort_values(by='TF-IDF Score', ascending=False).head(20)
        
        fig_tfidf = px.bar(
            tfidf_df, 
            x='TF-IDF Score', 
            y='Word', 
            orientation='h',
            title="Top 20 Kata Paling Signifikan Secara Keseluruhan",
            color='TF-IDF Score',
            color_continuous_scale='Viridis'
        )
        fig_tfidf.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_tfidf, use_container_width=True)

        # --- SIMPAN KE SESSION STATE ---
        # Data ini akan dipanggil oleh halaman Model Training
        st.session_state['X_train_tfidf'] = X_train_tfidf
        st.session_state['X_test_tfidf'] = X_test_tfidf
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['tfidf_vectorizer'] = tfidf
        st.session_state['data_processed'] = True
        
        st.success("Seluruh *pipeline* Preprocessing selesai dan data telah disimpan ke dalam memori! Silakan lanjut ke halaman **Train Model**.")

elif 'data_processed' in st.session_state and st.session_state['data_processed']:
    st.success("Data sudah diproses sebelumnya dan tersimpan di sistem. Anda bisa langsung menuju halaman **Train Model** atau klik tombol di atas untuk memproses ulang dengan persentase split atau fitur yang berbeda.")