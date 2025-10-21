# ============================================
# 1. Import Library
# ============================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ============================================
# 2. Load Dataset
# ============================================
csv_path = "data/spotify_songs.csv"  # Relative path sesuai struktur folder
df = pd.read_csv(csv_path)

# ============================================
# 3. Cek Informasi Awal Dataset
# ============================================
print("\n=== Informasi Dataset ===")
df.info()

print("\n=== 5 Data Teratas ===")
print(df.head())

print("\n=== Statistik Deskriptif ===")
print(df.describe())

# ============================================
# 4. Cek Missing Values & Duplikasi
# ============================================
print("\n=== Jumlah Missing Values per Kolom ===")
print(df.isnull().sum())

print("\n=== Jumlah Data Duplikat ===")
print(df.duplicated().sum())

# Hapus duplikat jika ada
df = df.drop_duplicates()

# ============================================
# 5. Tangani Missing Values
# ============================================
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

cat_cols = df.select_dtypes(exclude=np.number).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ============================================
# 6. Deteksi Outlier (IQR Method)
# ============================================
def remove_outliers_iqr(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

df = remove_outliers_iqr(df, num_cols)

# ============================================
# 7. Normalisasi Data Numerik
# ============================================
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ============================================
# 8. Encoding Kolom Kategorikal
# ============================================
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ============================================
# 9. Cek Hasil Akhir
# ============================================
print("\n=== Data Setelah Cleaning ===")
df.info()
print(df.head())

# ============================================
# 10. Simpan Dataset Bersih
# ============================================
output_path = "data/spotify_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Dataset bersih telah disimpan di: {output_path}")
