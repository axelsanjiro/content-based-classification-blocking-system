import streamlit as st

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Automated Content Based Classification and Blocking System")
st.markdown("#### *for Enhancing Digital Productivity*")
st.write("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Project Overview")
    st.markdown("""
    Aplikasi ini adalah bentuk simulasi (dashboard pipeline) dari sebuah *mobile app* yang dirancang untuk memblokir konten tidak produktif saat pengguna sedang fokus bekerja atau belajar. 
    
    Karena fokus utama dari *Machine Learning* adalah pemahaman terhadap **proses pipeline**, dashboard ini dibuat khusus untuk mendemonstrasikan bagaimana model *Content-Based Classification* dibangun dari awal hingga tahap deployment.
    """)

    st.header("Objectives")
    st.markdown("""
    - **Data Exploration:** Memahami distribusi kategori konten dari sumber berita nyata.
    - **Text Preprocessing:** Melakukan pembersihan teks menggunakan **spaCy** (Stopword removal, Lemmatization).
    - **Feature Extraction (TF-IDF):** Mengubah teks menjadi representasi vektor numerik menggunakan **Term Frequency-Inverse Document Frequency (TF-IDF)** untuk menangkap urgensi kata dalam dokumen.
    - **Model Training:** Membandingkan performa *Classical Machine Learning* (**Logistic Regression** vs **Naive Bayes**).
    - **Feature Selection:** Memberikan opsi klasifikasi berdasarkan `Title`, `Content`, atau gabungan keduanya.
    - **Deployment & Evaluation:** Menguji model secara *real-time* melalui antarmuka web interaktif.
    """)

with col2:
    st.header("Dataset Specification")
    st.markdown("""
    **Sumber Data:** [BBC News Archive (Kaggle)](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive)
    
    **Detail:**
    - Terdiri dari ribuan artikel berita BBC.
    - **Features yang digunakan:** `title` (Judul Berita) dan `content` (Isi Berita).
    - **Target Label:** `category` (kategori berita seperti bisnis, olahraga, teknologi, hiburan, dll).
    """)

st.write("---")
st.header("Pipeline Methodology")
st.markdown("""
1. **Exploratory Data Analysis (EDA):** Analisis univariat, bivariat, dan visualisasi distribusi data menggunakan Plotly.
2. **Preprocessing (spaCy):** Menghapus noise dari teks, tokenisasi, dan normalisasi kata (lemmatization).
3. **Feature Extraction (TF-IDF):** Menggunakan `TfidfVectorizer` untuk memberikan bobot pada kata-kata penting yang unik pada suatu kategori, sehingga model dapat membedakan konteks teks dengan lebih akurat sebelum dilatih.
4. **Data Splitting:** Membagi dataset menjadi data latih (*Train*) dan data uji (*Test*).
5. **Modeling:** Melatih model klasifikasi (Logistic Regression & Naive Bayes) dan melakukan penyesuaian *Hyperparameter*.
6. **Prediction:** Evaluasi model dan pengujian langsung oleh pengguna.
""")

st.write("---")
st.header("Team Members (Kelompok 5)")
st.markdown("""
- **Axel Sanjiro Yang** (Computer Science, BINUS University)
- **Alexandria Natasya Beslar** (Computer Science, BINUS University)
- **Sean Spencer** (Computer Science, BINUS University)
""")