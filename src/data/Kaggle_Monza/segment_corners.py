# segment_corners_adaptive.py
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter


def detect_corners_adaptive(df: pd.DataFrame):
    df = df.copy()

    # ADATTIVO ai tuoi dati
    steer_abs = df["steer"].abs()
    steer_thresh = steer_abs.quantile(0.9)  # top 10%
    speed_std = df["speed_kmh"].rolling(30).std()
    speed_drop_thresh = speed_std.quantile(0.75)

    print(f"steer range: {steer_abs.min():.3f} - {steer_abs.max():.3f} → thresh={steer_thresh:.3f}")
    print(f"speed_drop thresh: {speed_drop_thresh:.1f}")

    df["steer_smooth"] = savgol_filter(df["steer"].fillna(0), 7, 3)
    df["speed_ma"] = df["speed_kmh"].rolling(15, center=True).mean().bfill().ffill()
    df["speed_drop"] = df["speed_ma"].shift(-10) - df["speed_ma"]

    df["steer_active"] = np.abs(df["steer_smooth"]) > steer_thresh
    df["speed_drop_active"] = df["speed_drop"] > speed_drop_thresh
    df["is_curve_raw"] = df["steer_active"] & df["speed_drop_active"]

    df["curve_group"] = (df["is_curve_raw"] != df["is_curve_raw"].shift()).cumsum()
    group_lengths = df[df["is_curve_raw"]].groupby("curve_group").size()
    valid_groups = group_lengths[group_lengths >= 15].index  # min 15 step

    df["is_curve"] = df["curve_group"].isin(valid_groups)
    df["curve_id"] = np.where(df["is_curve"], df["curve_group"], -1).astype(int)

    # Fasi più permissive
    df["phase"] = "straight"
    df.loc[df["brake"] > 0.2, "phase"] = "braking"  # ← 0.2 invece di 0.3
    df.loc[df["throttle"] > 0.7, "phase"] = "accel"  # ← aggiunta

    # Cleanup
    df.drop(["steer_smooth", "speed_ma", "speed_drop", "steer_active",
             "speed_drop_active", "is_curve_raw", "curve_group"], axis=1, inplace=True)

    print(f"✅ Curve rilevate: {df['curve_id'].nunique() - 1}")
    print("Fasi:", df["phase"].value_counts().to_dict())

    return df


if __name__ == "__main__":
    df = pd.read_parquet("parquet//kaggle_monza_gt3.parquet")
    df_seg = detect_corners_adaptive(df)

    print("\nSample curve:")
    print(df_seg[df_seg["is_curve"]][["step_idx", "speed_kmh", "steer", "phase"]].head(10))

    df_seg.to_parquet("parquet//kaggle_monza_gt3_segmented.parquet", index=False)
    print("✅ SALVATO")
