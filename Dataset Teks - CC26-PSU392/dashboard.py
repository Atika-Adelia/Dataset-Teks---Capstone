import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import ssl

st.set_page_config(page_title="Emovision", page_icon="logo.png", layout="wide")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# stopword
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Mapping Stopword
list_stopwords = set(stopwords.words('indonesian'))
list_stopwords.update(['gua', 'gue', 'lu','gw','lo', 'aja', 'nya', 'kalo', 'sama', 'buat',
                       'aku', 'di', 'ke', 'dari', 'ini', 'itu', 'sih', 'ya', 'kayak', 'deh', 'orang'])

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Final Dataset Teks 4.csv") 
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")
        return pd.DataFrame()

df_raw = load_data()

if not df_raw.empty:
    df_raw['label_emosi'] = df_raw['label_emosi'].astype(str).str.strip().str.lower()

# Sidebar
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.title("Emovision")
st.sidebar.write("See Your Emotion, Understand Yourself")
st.sidebar.markdown("---")

# Menu Navigasi
menu = st.sidebar.radio("Navigasi Menu:", ["Dashboard Dataset Teks", "Data di Indonesia"] )

# Dashboard insight
if menu == "Dashboard Dataset Teks":
    st.title("Dashboard Dataset Teks Emovision")
    st.write("Eksplorasi dataset teks proyek EmoVision berdasarkan 7 kelas emosi manusia.")
    st.write("Tim Capstone CC26-PSU392")
    st.markdown("---")
    
    if not df_raw.empty:
        df_distribusi = df_raw['label_emosi'].value_counts().reset_index()
        df_distribusi.columns = ['Emosi', 'Jumlah']
        
        st.markdown("### 1. Distribusi Dataset Kelas Emosi")
        
        col1, col2 = st.columns([1, 2]) 
        
        with col1:
            st.write("Tabel Persebaran Data:")
            st.dataframe(df_distribusi, use_container_width=True, hide_index=True)
            
        with col2:
            st.write("Grafik Batang (Bar Chart):")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(df_distribusi['Emosi'], df_distribusi['Jumlah'], color='steelblue')
            ax.set_xlabel('Label Emosi')
            ax.set_ylabel('Jumlah Data')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            
        st.markdown("---")
        
        st.markdown("### 2. Kata Kunci Paling Dominan pada 7 Kelas Emosi")
        pilihan_emosi = st.selectbox("Pilih Kelas Emosi:", df_distribusi['Emosi'].tolist())
        
        if pilihan_emosi:
            teks_filter = df_raw[df_raw['label_emosi'] == pilihan_emosi]['teks'].dropna().astype(str)
            gabungan_teks = " ".join(teks_filter)
            
            wc = WordCloud(
                width=800, 
                height=400, 
                background_color='white', 
                colormap='Set2', 
                max_words=100,
                stopwords=list_stopwords
            )
            wc.generate(gabungan_teks)
            
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off') 
            ax_wc.set_title(f"WordCloud Emosi: {pilihan_emosi.upper()}", fontsize=14, fontweight='bold', pad=10)
            
            st.pyplot(fig_wc)
    else:
        st.warning("Data belum berhasil dimuat. Silakan cek file CSV dataset kamu.")

# dashboard analisis data SKI 2023
elif menu == "Data di Indonesia":
    st.title("Eksplorasi Data Kesehatan Mental RI")
    st.write("Eksplorasi data prevalensi depresi berdasarkan berbagai faktor sosio-demografi dari **Survei Kesehatan Indonesia (SKI) 2023** oleh Kementerian Kesehatan RI.")
    st.markdown("---")
    
    try:
        df_ski_full = pd.read_csv("Data_SKI_2023.csv")
        
        daftar_kategori = df_ski_full['Kategori'].unique()
        pilihan_kategori = st.selectbox("Pilih Kategori Demografi untuk Dianalisis:", daftar_kategori)
        
        df_filter = df_ski_full[df_ski_full['Kategori'] == pilihan_kategori].copy()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write(f"**Tabel Prevalensi: {pilihan_kategori}**")
            st.dataframe(df_filter[['Karakteristik', 'Persentase_Depresi']], use_container_width=True, hide_index=True)
            st.caption("Sumber: Laporan SKI 2023, Kemenkes RI")
            
        with col2:
            st.write(f"**Grafik Prevalensi Berdasarkan {pilihan_kategori}**")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            bars = ax.barh(df_filter['Karakteristik'], df_filter['Persentase_Depresi'], color='lightgray')
            
            max_val = df_filter['Persentase_Depresi'].max()
            for bar in bars:
                if bar.get_width() == max_val:
                    bar.set_color('steelblue')

            ax.bar_label(bars, fmt='%.1f%%', padding=5, fontweight='bold')
            ax.set_xlabel('Persentase Depresi (%)', fontweight='bold')
            
            ax.invert_yaxis()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
            
        st.markdown("---")
        
        st.info("""
        **Insight & Latar Belakang Proyek:** 
        
        Grafik pada data **Kelompok Umur** di atas memvalidasi bahwa kelompok demografi termuda (**15-24 tahun**) sangat rentan terhadap
        tekanan psikologis dan depresi. Hal tersebut disebabkan oleh beberapa faktor, seperti beban belajar dan kerja
        sehingga dapat menimbulkan suatu stress dan depresi pada diri seorang tersebut. Salah satu penyebab depresi tersebut
        bisa disebabkannya ketidakstabilan emosi dari waktu ke waktu.
        
        Oleh karena itu, dikembangkannya Emovision untuk mendeteksi emosi secara otomatis dari waktu ke waktu melalui journaling
        sehingga emosi dari waktu ke waktu dapat ditracking secara real time
        """)
        
    except FileNotFoundError:
        st.error("⚠️ File 'Data_SKI_2023.csv' belum ditemukan. Pastikan file tersebut sudah berada di folder yang sama dengan dashboard.py.")
