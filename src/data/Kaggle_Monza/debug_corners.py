# debug_corners.py - Trova parametri perfetti
import pandas as pd
import numpy as np

df = pd.read_parquet("parquet//kaggle_monza_gt3.parquet")

print("=== DIAGNOSTICA ===")
print("Steer stats:", df["steer"].describe())
print("Steer > 0.01:", (df["steer"].abs() > 0.01).sum())
print("Steer > 0.05:", (df["steer"].abs() > 0.05).sum())

# Segmenta con parametri ULTRA-PERMISIVI
steer_thresh = df["steer"].abs().quantile(0.95)  # top 5%
print(f"\nsteer_thresh 95%: {steer_thresh:.4f}")

df["steer_active"] = df["steer"].abs() > steer_thresh
print("Steer active:", df["steer_active"].sum())

# Solo STEER (no speed_drop per ora)
df["is_curve"] = df["steer_active"]
df["curve_id"] = (df["is_curve"] != df["is_curve"].shift()).cumsum().where(df["is_curve"], -1)

print(f"Curve con solo steer: {df['curve_id'].nunique() - 1}")

# Salva per LSTM
df.to_parquet("parquet//kaggle_for_lstm.parquet", index=False)
print("✅ Salvato kaggle_for_lstm.parquet con curve da steer puro")
