# ============================================
#  Model Training - Linear Regression (Top 5 Features)
# ============================================

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump
import numpy as np

# ============================================
# 1. Load Dataset
# ============================================
df = pd.read_csv("data/spotify_cleaned.csv")

# ============================================
# 2. Tentukan Target
# ============================================
target = "track_popularity"

# Hitung korelasi dan ambil 5 fitur paling berkorelasi (selain target)
corr_matrix = df.corr()
top_features = corr_matrix[target].abs().sort_values(ascending=False).index[1:6]  # Skip target itself

print("\n✅ Top 5 fitur yang digunakan untuk model:")
print(list(top_features))

# ============================================
# 3. Pisahkan Fitur & Target
# ============================================
X = df[top_features]
y = df[target]

# ============================================
# 4. Split Train-Test
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 5. Train Model Linear Regression
# ============================================
model = LinearRegression()
model.fit(X_train, y_train)

# ============================================
# 6. Evaluasi Model
# ============================================
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n📊 Hasil Evaluasi Model Linear Regression:")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")

# ============================================
# 7. Simpan Model ke Folder src/models/
# ============================================
model_path = "src/models/popularity_model.pkl"
dump(model, model_path)

print(f"\n✅ Model berhasil disimpan di: {model_path}")
