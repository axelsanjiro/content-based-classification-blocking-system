import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

st.set_page_config(page_title="Train Model", layout="wide")

st.title("Model Training & Evaluation")
st.markdown("Pilih algoritma *Machine Learning*, sesuaikan *hyperparameter*-nya, dan evaluasi performanya menggunakan data uji (*Test Data*).")

# 1. PENGECEKAN DATA DI SESSION STATE
# Pastikan user sudah melakukan preprocessing sebelumnya
if 'data_processed' not in st.session_state or not st.session_state['data_processed']:
    st.warning("Data belum diproses! Silakan kembali ke halaman **Preprocessing** untuk menyiapkan data terlebih dahulu.")
    st.stop() # Menghentikan eksekusi kode di bawahnya jika data belum ada

# Mengambil data dari memory (session state)
X_train_tfidf = st.session_state['X_train_tfidf']
X_test_tfidf = st.session_state['X_test_tfidf']
y_train = st.session_state['y_train']
y_test = st.session_state['y_test']

st.success("Data TF-IDF berhasil dimuat dari memori! Siap untuk pelatihan model.")
st.write("---")

# 2. KONFIGURASI MODEL & HYPERPARAMETER
st.header("Model Configuration")

col_model, col_params = st.columns([1, 2])

with col_model:
    # MENGGANTI st.radio MENJADI st.selectbox DI SINI
    model_choice = st.selectbox(
        "Pilih Algoritma Model:",
        options=["Logistic Regression", "Naive Bayes (Multinomial)"]
    )

with col_params:
    st.markdown(f"#### Konfigurasi Hyperparameter: **{model_choice}**")
    
    if model_choice == "Logistic Regression":
        st.markdown("Logistic Regression sangat baik untuk klasifikasi teks linier. Sesuaikan parameter regularisasi untuk mencegah *overfitting*.")
        # Hyperparameter untuk Logistic Regression
        c_param = st.slider(
            "Parameter `C` (Inverse of regularization strength):", 
            min_value=0.01, max_value=10.0, value=1.0, step=0.1,
            help="Nilai C yang kecil berarti regularisasi lebih kuat (mencegah overfitting). Nilai besar berarti model lebih kompleks."
        )
        max_iter_param = st.slider(
            "Maksimal Iterasi (`max_iter`):", 
            min_value=100, max_value=1000, value=200, step=100,
            help="Jumlah iterasi maksimal bagi *solver* untuk mencapai konvergensi."
        )
        
    elif model_choice == "Naive Bayes (Multinomial)":
        st.markdown("Multinomial Naive Bayes adalah standar klasik yang sangat cepat dan efektif untuk klasifikasi teks / frekuensi kata.")
        # Hyperparameter untuk Naive Bayes
        alpha_param = st.slider(
            "Parameter `alpha` (Laplace Smoothing):", 
            min_value=0.01, max_value=5.0, value=1.0, step=0.1,
            help="Mencegah probabilitas bernilai nol untuk kata yang tidak pernah muncul di data latih."
        )

st.write("---")

# 3. EKSEKUSI TRAINING & EVALUASI
if st.button("Train & Evaluate Model", type="primary"):
    with st.spinner(f"Melatih model {model_choice}..."):
        
        # Inisialisasi model berdasarkan pilihan user
        if model_choice == "Logistic Regression":
            model = LogisticRegression(C=c_param, max_iter=max_iter_param, random_state=42)
        else:
            model = MultinomialNB(alpha=alpha_param)
            
        # Proses Training
        model.fit(X_train_tfidf, y_train)
        
        # Proses Prediksi pada Data Uji (Test Data)
        y_pred = model.predict(X_test_tfidf)
        
        # Menyimpan model yang sudah dilatih ke session state untuk halaman selanjutnya
        st.session_state['trained_model'] = model
        st.session_state['model_name'] = model_choice
        
    # --- MENAMPILKAN HASIL EVALUASI ---
    st.header("Evaluation Results")
    st.success(f"Model **{model_choice}** berhasil dilatih dan dievaluasi!")
    
    # 1. Main Metrics (Accuracy, Precision, Recall, F1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
    col_met1.metric("Accuracy", f"{acc:.4f}")
    col_met2.metric("Precision (Weighted)", f"{prec:.4f}")
    col_met3.metric("Recall (Weighted)", f"{rec:.4f}")
    col_met4.metric("F1-Score (Weighted)", f"{f1:.4f}")
    
    # 2. Classification Report (Detail per kelas)
    st.subheader("Classification Report")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    # 3. Confusion Matrix (dengan Plotly)
    st.subheader("Confusion Matrix")
    st.markdown("Matriks ini menunjukkan di mana model berhasil menebak benar dan di mana model sering keliru (*misclassification*).")
    
    # Mendapatkan label kelas yang unik agar sumbu matriksnya jelas
    classes = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    fig_cm = px.imshow(
        cm, 
        text_auto=True, 
        color_continuous_scale='Blues',
        labels=dict(x="Predicted Label", y="True / Actual Label"),
        x=classes, 
        y=classes,
        title=f"Confusion Matrix - {model_choice}"
    )
    fig_cm.update_layout(xaxis_title="Predicted Category", yaxis_title="Actual Category")
    st.plotly_chart(fig_cm, use_container_width=True)
    
    st.info("**Tips:** Model yang sudah dilatih ini sekarang tersimpan secara otomatis. Anda dapat langsung menuju halaman **Prediction Demo** untuk mengujinya dengan teks buatan Anda sendiri!")

elif 'trained_model' in st.session_state:
    st.info(f"Model **{st.session_state['model_name']}** sudah dilatih sebelumnya. Anda dapat melatih ulang dengan pengaturan berbeda, atau langsung menuju halaman **Prediction Demo**.")