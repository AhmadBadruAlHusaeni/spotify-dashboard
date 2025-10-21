# ============================================
# Exploratory Data Analysis (EDA)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. Load Data
# ============================================
df = pd.read_csv("data/spotify_cleaned.csv")

print("✅ Data loaded successfully!")
print(df.head())
print("\n🔍 Info Dataset:")
print(df.info())

# ============================================
# 2. Distribusi Popularitas Lagu
# ============================================
plt.figure(figsize=(8, 5))
sns.histplot(df['track_popularity'], kde=True)
plt.title("Distribusi Popularitas Lagu")
plt.xlabel("Popularitas")
plt.ylabel("Frekuensi")
plt.show()

# ============================================
# 3. Heatmap Korelasi Antar Fitur Numerik
# ============================================
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap="viridis")
plt.title("Heatmap Korelasi Antar Fitur")
plt.show()

# ============================================
# 4. Ambil 5 Fitur Teratas yang Paling Berkorelasi dengan Popularitas
# ============================================
target = "track_popularity"
corr_with_target = corr_matrix[target].abs().sort_values(ascending=False)
top_features = corr_with_target.index[1:6]  # Skip target itself
print("\n📊  Top 5 fitur dengan korelasi tertinggi terhadap popularitas:")
print(corr_with_target.head(6))

# ============================================
# 5. Scatterplot Setiap Fitur dengan Target (Top 5)
# ============================================
for feature in top_features:
    plt.figure(figsize=(6, 4))
    sns.regplot(x=df[feature], y=df[target], scatter_kws={'alpha':0.5})
    plt.title(f"{feature} vs {target}")
    plt.xlabel(feature)
    plt.ylabel("Popularitas")
    plt.show()

# ============================================
# 6. Genre Terpopuler (Berdasarkan Jumlah Lagu)
# ============================================
plt.figure(figsize=(12,6))
genre_counts = df['playlist_genre'].value_counts().head(10)
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title("Top 10 Genre Terpopuler (Jumlah Lagu)")
plt.xlabel("Genre")
plt.ylabel("Jumlah Lagu")
plt.xticks(rotation=45)
plt.show()

# ============================================
# 7. Subgenre Terpopuler
# ============================================
plt.figure(figsize=(12,6))
subgenre_counts = df['playlist_subgenre'].value_counts().head(10)
sns.barplot(x=subgenre_counts.index, y=subgenre_counts.values)
plt.title("Top 10 Subgenre Terpopuler (Jumlah Lagu)")
plt.xlabel("Subgenre")
plt.ylabel("Jumlah Lagu")
plt.xticks(rotation=45)
plt.show()

print("\n✅ EDA Selesai! Semua grafik telah ditampilkan.")
