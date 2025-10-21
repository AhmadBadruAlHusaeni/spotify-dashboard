# app.py (V2) — Spotify Popularity Dashboard (Upgraded UI)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import random

# ---------------------------
# Page config & style
# ---------------------------
st.set_page_config(
    page_title="Spotify Popularity Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# seaborn theme & palettes
sns.set_style("whitegrid")
PALETTE_1 = "magma"
PALETTE_2 = "coolwarm"
PALETTE_3 = "viridis"

# ---------------------------
# Caching loaders
# ---------------------------
@st.cache_data
def load_data(path="data/spotify_cleaned.csv"):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path="src/models/popularity_model.pkl"):
    return load(path)

# ---------------------------
# Load resources
# ---------------------------
df = load_data()
model = load_model()

# recompute top features (consistent with model training)
corr_matrix = df.corr()
top_features = corr_matrix['track_popularity'].abs().sort_values(ascending=False).index[1:6].tolist()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("🎵 Spotify Popularity Dashboard")
st.sidebar.markdown("""
**Project**: Prediksi & Analisis Faktor Popularitas Lagu di Spotify  
**Model**: Linear Regression (baseline)  
**Dataset**: 30k Spotify songs (preprocessed)  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Tips:** Gunakan tab di atas untuk berpindah antara view. Untuk interaksi lebih lanjut (prediksi manual), minta upgrade V3.")

# ---------------------------
# Header
# ---------------------------
st.title("🎧 Prediksi & Analisis Faktor Popularitas Lagu di Spotify")
st.markdown(
    "Dashboard ini menampilkan distribusi popularitas, insight genre, korelasi fitur, "
    "dan contoh prediksi menggunakan model Linear Regression. Warna dan layout telah diperbarui untuk presentasi."
)

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs([
    "🏠 Overview",
    "📈 Popularitas",
    "🎼 Genre Insight",
    "🔗 Korelasi",
    "🤖 Prediksi Contoh"
])

# ---------------------------
# Tab: Overview
# ---------------------------
with tabs[0]:
    st.header("🏠 Overview Project")
    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col1:
        st.subheader("Dataset Snapshot")
        st.write(f"Jumlah baris: **{df.shape[0]:,}**")
        st.write(f"Jumlah kolom: **{df.shape[1]}**")
        st.write("Contoh beberapa kolom penting:")
        st.dataframe(df[top_features + ['track_popularity']].head(6), use_container_width=True)
    with col2:
        st.subheader("Target Summary")
        st.metric("Mean Popularity", f"{df['track_popularity'].mean():.2f}")
        st.metric("Median Popularity", f"{df['track_popularity'].median():.2f}")
        st.metric("Max Popularity", f"{df['track_popularity'].max():.0f}")
    with col3:
        st.subheader("Top Features (by abs correlation)")
        for i, f in enumerate(top_features, 1):
            val = corr_matrix['track_popularity'][f]
            st.write(f"{i}. **{f}** — korelasi: {val:.3f}")

# ---------------------------
# Tab: Popularitas
# ---------------------------
with tabs[1]:
    st.header("📈 Distribusi Popularitas")
    st.markdown("Visualisasi distribusi `track_popularity` untuk melihat sebaran skor lagu dalam dataset.")
    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(df['track_popularity'], kde=True, ax=ax, stat="density", color=sns.color_palette(PALETTE_1, 1)[0])
    ax.set_xlabel("Track Popularity")
    ax.set_ylabel("Density")
    ax.set_title("Distribusi Popularitas Lagu")
    st.pyplot(fig, use_container_width=True)

    st.markdown("**Boxplot Popularitas** (mendeteksi skew & outlier yang tersisa)")
    fig2, ax2 = plt.subplots(figsize=(10,2))
    sns.boxplot(x=df['track_popularity'], ax=ax2, palette=[sns.color_palette(PALETTE_2, 1)[0]])
    st.pyplot(fig2, use_container_width=True)

# ---------------------------
# Tab: Genre Insight
# ---------------------------
with tabs[2]:
    st.header("🎼 Genre & Subgenre Insight")
    st.markdown("Top genre dan subgenre berdasarkan jumlah lagu — berguna untuk melihat distribusi konten dataset.")
    colg1, colg2 = st.columns(2)

    with colg1:
        top_genres = df['playlist_genre'].value_counts().head(10)
        figg1, axg1 = plt.subplots(figsize=(8,4))
        sns.barplot(x=top_genres.values, y=top_genres.index, palette=PALETTE_3, ax=axg1)
        axg1.set_title("Top 10 Genre (Jumlah Lagu)")
        axg1.set_xlabel("Jumlah Lagu")
        axg1.set_ylabel("Genre")
        st.pyplot(figg1, use_container_width=True)

    with colg2:
        top_sub = df['playlist_subgenre'].value_counts().head(10)
        figg2, axg2 = plt.subplots(figsize=(8,4))
        sns.barplot(x=top_sub.values, y=top_sub.index, palette=PALETTE_1, ax=axg2)
        axg2.set_title("Top 10 Subgenre (Jumlah Lagu)")
        axg2.set_xlabel("Jumlah Lagu")
        axg2.set_ylabel("Subgenre")
        st.pyplot(figg2, use_container_width=True)

    st.markdown("**Catatan:** Jika ingin analisis popularitas per genre (rata-rata), kita bisa tambah visual ini di versi berikutnya.")

# ---------------------------
# Tab: Korelasi
# ---------------------------
with tabs[3]:
    st.header("🔗 Korelasi Fitur dengan Popularitas")
    st.markdown("Heatmap korelasi antar top features + target. Scatterplot menunjukkan hubungan fitur vs popularitas.")

    # Heatmap (top features + target)
    cols_to_plot = top_features + ['track_popularity']
    figc, axc = plt.subplots(figsize=(8,6))
    sns.heatmap(df[cols_to_plot].corr(), annot=True, cmap="coolwarm", ax=axc, vmin=-1, vmax=1)
    axc.set_title("Heatmap Korelasi (top features)")
    st.pyplot(figc, use_container_width=True)

    # Scatter plots for each top feature
    st.markdown("#### Scatterplot: fitur vs popularity (dengan regression line)")
    for f in top_features:
        figsc, axsc = plt.subplots(figsize=(6,3))
        sns.regplot(x=df[f], y=df['track_popularity'], scatter_kws={'alpha':0.4}, line_kws={'color':'black'}, ax=axsc)
        axsc.set_xlabel(f)
        axsc.set_ylabel("Popularity")
        axsc.set_title(f"{f} vs Popularity")
        st.pyplot(figsc, use_container_width=True)

# ---------------------------
# Tab: Prediksi Contoh
# ---------------------------
with tabs[4]:
    st.header("🤖 Prediksi Contoh (Random Sample)")
    st.markdown("Model Linear Regression dipakai untuk memprediksi popularitas contoh lagu dari dataset. Ini berguna untuk mengecek kualitas prediksi (Predicted vs Actual).")

    # choose a random sample
    sample_idx = random.randint(0, len(df)-1)
    sample_row = df.iloc[sample_idx]
    sample_features = sample_row[top_features].values.reshape(1, -1)
    predicted = model.predict(sample_features)[0]
    actual = sample_row['track_popularity']

    colp1, colp2 = st.columns([1,1])
    with colp1:
        st.subheader("🎵 Data Contoh (features)")
        st.table(pd.DataFrame([sample_row[top_features].round(4)], index=["value"]).T)
    with colp2:
        st.subheader("📈 Hasil Prediksi")
        st.metric(label="Predicted Popularity", value=f"{predicted:.4f}")
        st.metric(label="Actual Popularity", value=f"{actual:.4f}")
        st.markdown("**Interpretasi singkat:** Angka popularitas dinormalisasi pada preprocessing. Bandingkan tren dan selisih Predicted vs Actual sebagai indikator performa.")

    st.markdown("---")
    st.markdown("Jika ingin melihat distribusi sisa error (residual) atau plot Predicted vs Actual untuk banyak sampel, minta fitur tambahan V3.")

# ---------------------------
# Footer / Notes
# ---------------------------
st.markdown("---")
st.caption("Dashboard V2 — tampilan diperbarui. Untuk memodifikasi warna/palettes atau menambahkan interaksi (prediksi manual), minta upgrade selanjutnya.")
