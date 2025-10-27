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

# Load original dataset (untuk nama lagu & artis, genre, subgenre sebelum encoding)
@st.cache_data
def load_original_data(path="data/spotify_songs.csv"):
    return pd.read_csv(path)

df_original = load_original_data()


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
    "🔍 Cari & Prediksi Lagu"
])

# ---------------------------
# Tab: Overview (UPDATED)
# ---------------------------
with tabs[0]:
    st.header("🏠 Overview Project")

    col1, col2, col3 = st.columns([1.5, 1, 1])
    
    # Dataset Snapshot
    with col1:
        st.subheader("Dataset Snapshot")
        st.write(f"Jumlah baris: **{df.shape[0]:,}**")
        st.write(f"Jumlah kolom: **{df.shape[1]}**")
        st.write("Contoh beberapa kolom penting:")
        st.dataframe(df[top_features + ['track_popularity']].head(6), use_container_width=True)

    # Target Summary
    with col2:
        st.subheader("Target Summary")
        mean_pop = df['track_popularity'].mean()
        median_pop = df['track_popularity'].median()
        max_pop = df['track_popularity'].max()
        st.metric("Mean Popularity", f"{mean_pop:.2f}")
        st.metric("Median Popularity", f"{median_pop:.2f}")
        st.metric("Max Popularity", f"{max_pop:.0f}")

    # Top Features
    with col3:
        st.subheader("Top Features (by abs correlation)")
        for i, f in enumerate(top_features, 1):
            val = corr_matrix['track_popularity'][f]
            st.write(f"{i}. **{f}** — korelasi: {val:.3f}")

    # ===========================
    # Tambahan Baru: Top 5 & Bottom 5 Lagu
    # ===========================
    st.subheader("🔥 Top 5 Lagu Paling Populer")
    df_top5 = df_original.nlargest(5, "track_popularity")[["track_name", "track_artist", "track_popularity"]].reset_index(drop=True)
    df_top5.index += 1
    st.dataframe(df_top5, use_container_width=True)

    st.subheader("❄ Bottom 5 Lagu Kurang Populer")
    df_bottom5 = df_original.nsmallest(5, "track_popularity")[["track_name", "track_artist", "track_popularity"]].reset_index(drop=True)
    df_bottom5.index += 1
    st.dataframe(df_bottom5, use_container_width=True)

    # Insight otomatis
    top_song = df_top5.iloc[0]["track_name"]
    top_artist = df_top5.iloc[0]["track_artist"]
    bottom_song = df_bottom5.iloc[0]["track_name"]
    bottom_artist = df_bottom5.iloc[0]["track_artist"]

    st.info(
        f"🎯 Lagu dengan popularitas tertinggi adalah **'{top_song}'** oleh **{top_artist}**, "
        f"sedangkan lagu dengan popularitas terendah adalah **'{bottom_song}'** oleh **{bottom_artist}**."
    )


# ---------------------------
# Tab: Popularitas (UPDATED)
# ---------------------------
with tabs[1]:
    st.header("📈 Distribusi Popularitas Lagu")
    st.markdown("Tab ini menunjukkan bagaimana popularitas lagu tersebar dalam dataset, lengkap dengan garis rata-rata (mean) dan median untuk membantu interpretasi.")

    mean_pop = df['track_popularity'].mean()
    median_pop = df['track_popularity'].median()

    # Histogram dengan garis Mean & Median
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['track_popularity'], kde=True, stat="density", color=sns.color_palette("plasma", 1)[0], ax=ax)
    ax.axvline(mean_pop, color='red', linestyle='--', linewidth=2, label=f"Mean: {mean_pop:.2f}")
    ax.axvline(median_pop, color='green', linestyle='-', linewidth=2, label=f"Median: {median_pop:.2f}")
    ax.set_xlabel("Track Popularity")
    ax.set_ylabel("Density")
    ax.set_title("Distribusi Popularitas Lagu")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    # Boxplot (tetap digunakan untuk deteksi outlier)
    st.markdown("### 📦 Boxplot Popularitas (untuk melihat outlier)")
    fig2, ax2 = plt.subplots(figsize=(10, 2))
    sns.boxplot(x=df['track_popularity'], ax=ax2, palette=[sns.color_palette("plasma", 1)[0]])
    st.pyplot(fig2, use_container_width=True)

    # Insight otomatis
    st.markdown("### 🧠 Insight:")
    if mean_pop > median_pop:
        skew_desc = "Distribusi sedikit condong ke kanan (right-skewed), artinya ada lagu-lagu dengan popularitas tinggi yang menarik rata-rata ke atas."
    elif mean_pop < median_pop:
        skew_desc = "Distribusi sedikit condong ke kiri (left-skewed), artinya sebagian besar lagu memiliki popularitas moderat."
    else:
        skew_desc = "Distribusi cukup simetris antara mean dan median."

    st.info(
        f"✅ Rata-rata popularitas: **{mean_pop:.2f}**, median: **{median_pop:.2f}**.\n"
        f"📌 {skew_desc}\n"
        f"📊 Mayoritas lagu berada pada rentang popularitas sekitar "
        f"**{df['track_popularity'].quantile(0.25):.0f} hingga {df['track_popularity'].quantile(0.75):.0f}**."
    )


# ---------------------------
# Tab: Genre Insight (UPDATED)
# ---------------------------
with tabs[2]:
    st.header("🎼 Genre & Popularitas Insight")
    st.markdown("""
    Tab ini menunjukkan distribusi jumlah lagu berdasarkan genre dan subgenre, 
    serta analisis genre mana yang cenderung lebih populer berdasarkan rata-rata popularitas.
    """)

    colg1, colg2 = st.columns(2)

    # ============================
    # Top 10 Genre (Jumlah Lagu)
    # ============================
    with colg1:
        st.subheader("📊 Top 10 Genre (Jumlah Lagu)")
        top_genres = df_original['playlist_genre'].value_counts().head(10)
        top_genres_labeled = top_genres.index + " (" + top_genres.values.astype(str) + " lagu)"
        figg1, axg1 = plt.subplots(figsize=(8, 4))
        sns.barplot(x=top_genres.values, y=top_genres_labeled, palette="plasma", ax=axg1)
        axg1.set_title("Top 10 Genre berdasarkan Jumlah Lagu")
        axg1.set_xlabel("Jumlah Lagu")
        axg1.set_ylabel("Genre")
        st.pyplot(figg1, use_container_width=True)

    # ============================
    # Top 10 Subgenre (Jumlah Lagu)
    # ============================
    with colg2:
        st.subheader("🎧 Top 10 Subgenre (Jumlah Lagu)")
        top_subgenres = df_original['playlist_subgenre'].value_counts().head(10)
        top_subgenres_labeled = top_subgenres.index + " (" + top_subgenres.values.astype(str) + " lagu)"
        figg2, axg2 = plt.subplots(figsize=(8, 4))
        sns.barplot(x=top_subgenres.values, y=top_subgenres_labeled, palette="plasma", ax=axg2)
        axg2.set_title("Top 10 Subgenre berdasarkan Jumlah Lagu")
        axg2.set_xlabel("Jumlah Lagu")
        axg2.set_ylabel("Subgenre")
        st.pyplot(figg2, use_container_width=True)

    # ============================
    # Rata-rata Popularitas per Genre
    # ============================
    st.subheader("🌟 Genre dengan Rata-rata Popularitas Tertinggi")
    genre_popularity = df_original.groupby("playlist_genre")["track_popularity"].mean().sort_values(ascending=False).head(7)
    figg3, axg3 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=genre_popularity.values, y=genre_popularity.index, palette="plasma", ax=axg3)
    axg3.set_title("Top 7 Genre berdasarkan Rata-rata Popularitas")
    axg3.set_xlabel("Rata-rata Popularitas")
    axg3.set_ylabel("Genre")
    st.pyplot(figg3, use_container_width=True)

    # Insight otomatis
    top_genre = genre_popularity.index[0]
    top_pop_score = genre_popularity.values[0]

    st.info(
        f"🎯 Genre dengan rata-rata popularitas tertinggi adalah **'{top_genre}'** "
        f"dengan rata-rata skor sekitar **{top_pop_score:.2f}**. "
        "Hal ini menunjukkan bahwa lagu dalam genre tersebut cenderung lebih disukai pendengar Spotify."
    )


# ---------------------------
# Tab: Korelasi (UPDATED)
# ---------------------------
with tabs[3]:
    st.header("🔗 Korelasi Fitur dengan Popularitas")
    st.markdown("""
    Korelasi membantu kita memahami seberapa kuat hubungan antara fitur audio dengan popularitas lagu.
    Nilai korelasi berada pada rentang -1 hingga 1:
    - **Mendekati 1** → hubungan positif kuat (nilai fitur naik → popularitas naik)
    - **Mendekati -1** → hubungan negatif kuat (nilai fitur naik → popularitas turun)
    - **Mendekati 0** → hampir tidak ada hubungan.
    """)

    # Heatmap (top features + target)
    cols_to_plot = top_features + ['track_popularity']
    figc, axc = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[cols_to_plot].corr(), annot=True, cmap="coolwarm", ax=axc, vmin=-1, vmax=1)
    axc.set_title("Heatmap Korelasi (Top Features vs Popularitas)")
    st.pyplot(figc, use_container_width=True)

    # Insight otomatis
    corr_pop = df.corr()['track_popularity'].drop('track_popularity')
    top2_corr = corr_pop.sort_values(ascending=False).head(2)
    lowest_corr = corr_pop.sort_values(ascending=True).head(1)
    
    st.subheader("📌 Insight Korelasi Utama")
    st.info(
        f"✅ Fitur dengan korelasi paling tinggi terhadap popularitas: **{top2_corr.index[0]} ({top2_corr.values[0]:.3f})** dan "
        f"**{top2_corr.index[1]} ({top2_corr.values[1]:.3f})**.\n"
        f"⚠ Fitur dengan korelasi paling rendah: **{lowest_corr.index[0]} ({lowest_corr.values[0]:.3f})**, "
        f"menunjukkan hampir tidak ada pengaruh langsung terhadap popularitas."
    )

    # Scatter plots with interpretation
    st.subheader("📍 Scatterplot: Hubungan Fitur vs Popularitas")
    st.markdown("Setiap grafik berikut menunjukkan hubungan antara fitur dan popularitas dengan garis regresi untuk melihat kecenderungan hubungan.")

    for f in top_features:
        figsc, axsc = plt.subplots(figsize=(6, 3))
        sns.regplot(x=df[f], y=df['track_popularity'], scatter_kws={'alpha': 0.4}, line_kws={'color': 'black'}, ax=axsc)
        axsc.set_xlabel(f)
        axsc.set_ylabel("Popularity")
        axsc.set_title(f"{f} vs Popularity")
        st.pyplot(figsc, use_container_width=True)
        
        # Caption otomatis
        corr_val = corr_pop[f]
        if corr_val > 0.1:
            trend_desc = "hubungan positif (semakin tinggi fitur, popularitas sedikit meningkat)"
        elif corr_val < -0.1:
            trend_desc = "hubungan negatif (semakin tinggi fitur, popularitas sedikit menurun)"
        else:
            trend_desc = "hubungan sangat lemah atau hampir tidak ada"
        st.caption(f"📍 Interpretasi: Korelasi {corr_val:.3f} menunjukkan {trend_desc}.")

    # Kesimpulan akhir
    st.markdown("---")
    st.success(
        "💡 Kesimpulan: Korelasi antar fitur dengan popularitas cenderung lemah, "
        "yang menjelaskan mengapa model linier menghasilkan skor R² yang rendah. "
        "Popularitas lagu kemungkinan juga dipengaruhi faktor eksternal seperti viralitas, artis terkenal, dan tren sosial."
    )


## ---------------------------
# Tab: 🔍 Cari & Prediksi Lagu (gabungan random + manual)
# ---------------------------
with tabs[4]:
    st.header("🔍 Cari & Prediksi Lagu")
    st.markdown("""
    Tab ini memungkinkan kamu untuk **memilih lagu tertentu berdasarkan judul atau artis**, 
    atau menampilkan **prediksi acak** dari model Linear Regression.
    """)

    # Pilihan mode
    mode = st.radio(
        "Pilih mode prediksi:",
        ["🎲 Prediksi Random", "📝 Cari Lagu Manual"],
        index=0,
        horizontal=True
    )

    st.markdown("---")

    # ---------------------------
    # MODE 1: Prediksi Random
    # ---------------------------
    if mode == "🎲 Prediksi Random":
        st.subheader("🎲 Prediksi Lagu Acak")
        if st.button("🔄 Ambil Lagu Acak Baru"):
            st.session_state['sample_idx_v3'] = random.randint(0, len(df) - 1)

        if 'sample_idx_v3' not in st.session_state:
            st.session_state['sample_idx_v3'] = random.randint(0, len(df) - 1)

        idx = st.session_state['sample_idx_v3']
        sample = df.iloc[idx]
        sample_orig = df_original.iloc[idx]

        st.markdown(f"""
        **🎵 Judul:** {sample_orig['track_name']}  
        **🎤 Artis:** {sample_orig['track_artist']}  
        **🎧 Genre:** {sample_orig['playlist_genre']} | **Subgenre:** {sample_orig['playlist_subgenre']}  
        **📅 Tanggal Rilis:** {sample_orig['track_album_release_date']}  
        **⏱️ Durasi:** {round(sample_orig['duration_ms']/60000, 2)} menit  
        """)

        st.markdown("#### 📊 Nilai Fitur (Top 5)")
        st.table(pd.DataFrame([sample[top_features].round(4)], index=["value"]).T)

        X = sample[top_features].values.reshape(1, -1)
        y_pred = model.predict(X)[0]
        y_true = sample['track_popularity']
        diff = abs(y_pred - y_true)

        st.markdown("#### 📈 Hasil Prediksi")
        st.metric("Predicted Popularity", f"{y_pred:.4f}")
        st.metric("Actual Popularity", f"{y_true:.4f}")
        st.metric("Selisih", f"{diff:.4f}")

        # Interpretasi popularitas
        pop_threshold = 0.46
        status = "🟢 Populer" if y_pred >= pop_threshold else "🔵 Kurang Populer"
        st.markdown(f"### 🎯 Interpretasi")
        st.markdown(f"Lagu **{sample_orig['track_name']}** oleh **{sample_orig['track_artist']}** diprediksi sebagai: **{status}**")
        st.caption(f"_Kriteria: skor popularitas ≥ {pop_threshold} dianggap populer berdasarkan distribusi dataset._")
        st.caption("_Nilai popularitas telah dinormalisasi (0–1). Semakin kecil selisih, semakin akurat model._")

    # ---------------------------
    # MODE 2: Cari Lagu Manual
    # ---------------------------
    else:
        st.subheader("📝 Cari Lagu Berdasarkan Judul atau Artis")
        query = st.text_input("Masukkan nama lagu atau artis:")

        if query:
            matches = df_original[df_original['track_name'].str.contains(query, case=False, na=False) |
                                  df_original['track_artist'].str.contains(query, case=False, na=False)]

            if len(matches) == 0:
                st.warning("❌ Lagu atau artis tidak ditemukan. Coba ketik sebagian nama lain.")
            else:
                st.success(f"✅ Ditemukan {len(matches)} hasil. Pilih salah satu untuk diprediksi:")
                selected = st.selectbox("Pilih lagu:", matches['track_name'] + " — " + matches['track_artist'])

                if selected:
                    row_idx = matches[matches['track_name'] + " — " + matches['track_artist'] == selected].index[0]
                    sample = df.iloc[row_idx]
                    sample_orig = df_original.iloc[row_idx]

                    st.markdown(f"""
                    **🎵 Judul:** {sample_orig['track_name']}  
                    **🎤 Artis:** {sample_orig['track_artist']}  
                    **🎧 Genre:** {sample_orig['playlist_genre']} | **Subgenre:** {sample_orig['playlist_subgenre']}  
                    **📅 Tanggal Rilis:** {sample_orig['track_album_release_date']}  
                    **⏱️ Durasi:** {round(sample_orig['duration_ms']/60000, 2)} menit  
                    """)

                    st.markdown("#### 📊 Nilai Fitur (Top 5)")
                    st.table(pd.DataFrame([sample[top_features].round(4)], index=["value"]).T)

                    X = sample[top_features].values.reshape(1, -1)
                    y_pred = model.predict(X)[0]
                    y_true = sample['track_popularity']
                    diff = abs(y_pred - y_true)

                    st.markdown("#### 📈 Hasil Prediksi")
                    st.metric("Predicted Popularity", f"{y_pred:.4f}")
                    st.metric("Actual Popularity", f"{y_true:.4f}")
                    st.metric("Selisih", f"{diff:.4f}")

                    # Interpretasi popularitas
                    pop_threshold = 0.46
                    status = "🟢 Populer" if y_pred >= pop_threshold else "🔵 Kurang Populer"
                    st.markdown(f"### 🎯 Interpretasi")
                    st.markdown(f"Lagu **{sample_orig['track_name']}** oleh **{sample_orig['track_artist']}** diprediksi sebagai: **{status}**")
                    st.caption(f"_Kriteria: skor popularitas ≥ {pop_threshold} dianggap populer berdasarkan distribusi dataset._")
                    st.caption("_Nilai popularitas telah dinormalisasi (0–1). Semakin kecil selisih, semakin akurat model._")



