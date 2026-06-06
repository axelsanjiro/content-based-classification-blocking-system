import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EDA", layout="wide")

st.title("📊 Exploratory Data Analysis (EDA)")
st.markdown("Eksplorasi dataset BBC News untuk memahami distribusi dan karakteristik teks sebelum masuk ke tahap *Preprocessing*.")

# Load Data dengan st.cache_data agar tidak load ulang berkali-kali
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/bbc-news-data.csv", sep='\t')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'bbc-news-data.csv' tidak ditemukan. Pastikan file sudah berada di dalam folder 'dataset/'.")
    st.stop()

# Ekstraksi fitur tambahan untuk keperluan EDA
df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))
df['content_word_count'] = df['content'].apply(lambda x: len(str(x).split()))

st.write("---")


# 1. PREVIEW RAW DATA
st.subheader("1. Preview Raw Data")
st.markdown("Berikut adalah cuplikan data mentah dari dataset. Terdapat fitur `category`, `filename`, `title`, dan `content`.")
st.dataframe(df.head(10), use_container_width=True)

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.info(f"**Total Baris (Dokumen):** {df.shape[0]}")
with col_info2:
    st.info(f"**Total Kolom:** {df.shape[1]}")

st.write("---")


# 2. ANALISIS KATEGORI (Bar & Donut Chart)
st.subheader("2. Analisis Distribusi Kategori")
st.markdown("Mengecek apakah dataset kita *balanced* (seimbang) atau *imbalanced* di tiap kategorinya. Keseimbangan data sangat memengaruhi performa model klasifikasi.")

category_counts = df['category'].value_counts().reset_index()
category_counts.columns = ['Kategori', 'Jumlah']

col_cat1, col_cat2 = st.columns(2)

with col_cat1:
    fig_bar = px.bar(
        category_counts, 
        x='Kategori', 
        y='Jumlah', 
        color='Kategori',
        text='Jumlah',
        title="Jumlah Artikel per Kategori",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_cat2:
    fig_pie = px.pie(
        category_counts, 
        names='Kategori', 
        values='Jumlah', 
        hole=0.4,
        title="Persentase Proporsi Kategori",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

st.write("---")


# 3. DISTRIBUSI PANJANG TEKS (Histogram)
st.subheader("3. Distribusi Jumlah Kata (Histogram)")
st.markdown("Mengetahui seberapa panjang rata-rata kata dalam `title` maupun `content` secara keseluruhan.")

col_hist1, col_hist2 = st.columns(2)

with col_hist1:
    fig_title_len = px.histogram(
        df, x='title_word_count', nbins=30, 
        title="Distribusi Jumlah Kata pada Judul (Title)",
        color_discrete_sequence=['#636EFA']
    )
    fig_title_len.update_layout(xaxis_title="Jumlah Kata", yaxis_title="Frekuensi")
    st.plotly_chart(fig_title_len, use_container_width=True)

with col_hist2:
    fig_content_len = px.histogram(
        df, x='content_word_count', nbins=50, 
        title="Distribusi Jumlah Kata pada Isi (Content)",
        color_discrete_sequence=['#EF553B']
    )
    fig_content_len.update_layout(xaxis_title="Jumlah Kata", yaxis_title="Frekuensi")
    st.plotly_chart(fig_content_len, use_container_width=True)

st.write("---")


# 4. OUTLIER & SEBARAN PER KATEGORI (Boxplot)
st.subheader("4. Analisis Sebaran Panjang Teks per Kategori (Boxplot)")
st.markdown("Boxplot membantu kita melihat anomali/outlier panjang teks di kategori tertentu.")

col_box1, col_box2 = st.columns(2)

with col_box1:
    fig_box_title = px.box(
        df, x='category', y='title_word_count', color='category',
        title="Sebaran Panjang Judul (Title)",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_box_title.update_layout(xaxis_title="Kategori", yaxis_title="Jumlah Kata", showlegend=False)
    st.plotly_chart(fig_box_title, use_container_width=True)

with col_box2:
    fig_box_content = px.box(
        df, x='category', y='content_word_count', color='category',
        title="Sebaran Panjang Isi (Content)",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_box_content.update_layout(xaxis_title="Kategori", yaxis_title="Jumlah Kata", showlegend=False)
    st.plotly_chart(fig_box_content, use_container_width=True)

st.write("---")


# 5. KORELASI (Scatter Plot)
st.subheader("5. Korelasi Antara Panjang Judul dan Panjang Isi")
st.markdown("Apakah judul yang panjang selalu diikuti dengan isi berita yang panjang pula? Mari kita lihat sebarannya berdasarkan kategori.")

fig_scatter = px.scatter(
    df, 
    x='title_word_count', 
    y='content_word_count', 
    color='category',
    opacity=0.7,
    title="Korelasi Panjang Judul vs Panjang Isi Berita",
    hover_data=['title'] # Menampilkan judul asli saat data di-hover
)
fig_scatter.update_layout(xaxis_title="Jumlah Kata (Title)", yaxis_title="Jumlah Kata (Content)")
st.plotly_chart(fig_scatter, use_container_width=True)