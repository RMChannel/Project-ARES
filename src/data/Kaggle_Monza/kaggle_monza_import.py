# import_kaggle_standalone.py - VERSIONE STANDALONE (no dipendenze esterne)
import pandas as pd
import numpy as np
from pathlib import Path


def import_kaggle_monza_exact(path: str | Path, session_id: str = "kaggle_monza_gt3"):
    """
    Adapter standalone per CSV Kaggle con colonne esatte fornite.
    """
    path = Path(path)
    df_raw = pd.read_csv(path)

    print(f"Import {len(df_raw)} righe da {path}")
    print("Colonne:", df_raw.columns.tolist())

    df = pd.DataFrame()

    # Metadati
    df["session_id"] = session_id
    df["source"] = "kaggle_gt3"
    df["track"] = "monza"
    df["car"] = "gt3_generic"

    df["step_idx"] = df_raw["binIndex"].astype(int)

    df["t_abs"] = pd.NA
    df["t_lap"] = df_raw["lap_time"].astype(float)
    df["lap"] = df_raw["lapNum"].astype(int)

    # Posizione
    df["x_world"] = df_raw["world_position_X"].astype(float)
    df["y_world"] = df_raw["world_position_Y"].astype(float)
    df["z_world"] = df_raw["world_position_Z"].astype(float)

    # Velocità
    df["vx"] = df_raw["velocity_X"].astype(float)
    df["vy"] = df_raw["velocity_Y"].astype(float)
    df["vz"] = df_raw["velocity_Z"].astype(float)
    df["speed_kmh"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2 + df["vz"] ** 2) * 3.6

    # Orientamento
    df["yaw"] = pd.NA
    df["yaw_rate"] = pd.NA
    df["pitch"] = pd.NA
    df["roll"] = pd.NA

    # Controlli (clip in range)
    df["throttle"] = df_raw["throttle"].astype(float).clip(0, 1)
    df["brake"] = df_raw["brake"].astype(float).clip(0, 1)
    df["steer"] = df_raw["steering"].astype(float)  # assumi già [-1,1]
    df["clutch"] = pd.NA

    # Engine
    df["rpm"] = df_raw["rpm"].astype(float)
    df["gear"] = df_raw["gear"].astype(int)

    # Gomme → NaN
    for col in ["tyre_slip_fl", "tyre_slip_fr", "tyre_slip_rl", "tyre_slip_rr",
                "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr"]:
        df[col] = pd.NA

    # G-force
    df["acc_g_x"] = pd.NA
    df["acc_g_y"] = df_raw["gforce_Y"].astype(float)
    df["acc_g_z"] = pd.NA

    # Flag
    df["is_valid_lap"] = df_raw["validBin"].astype(bool)
    df["is_curve"] = False
    df["curve_id"] = -1
    df["sector"] = -1

    # Normalizza steer se necessario
    steer_max = df["steer"].abs().max()
    if steer_max > 2.0:  # se in gradi
        df["steer"] = df["steer"] / steer_max

    # Feature derivate
    df["speed_norm"] = (df["speed_kmh"] / 300.0).clip(0, 1)
    df["acc_long"] = pd.NA
    df["acc_lat"] = df["acc_g_y"] * 9.81 if "acc_g_y" in df else pd.NA

    # Norm_pos da binIndex
    df["norm_pos"] = df["step_idx"] / df_raw["binIndex"].max()

    print("Dataset pronto:", df.shape)
    print("Speed range:", df["speed_kmh"].min(), "→", df["speed_kmh"].max())
    print("Steer range:", df["steer"].min(), "→", df["steer"].max())

    return df


if __name__ == "__main__":
    # Test diretto
    df = import_kaggle_monza_exact("csv//Monzafc-5lap.csv")

    df.to_parquet("parquet//kaggle_monza_gt3.parquet", index=False)
    print("Salvato → /parquet/kaggle_monza_gt3.parquet")
