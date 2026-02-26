import pandas as pd
import numpy as np

CORE_COLUMNS = [
    "session_id", "source", "track", "car",
    "t_abs", "t_lap", "lap", "step_idx",
    "x_world", "y_world", "z_world",
    "speed_kmh", "vx", "vy", "vz",
    "yaw", "yaw_rate", "pitch", "roll",
    "throttle", "brake", "steer", "clutch",
    "rpm", "gear",
    "tyre_slip_fl", "tyre_slip_fr", "tyre_slip_rl", "tyre_slip_rr",
    "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr",
    "acc_g_x", "acc_g_y", "acc_g_z",
    "is_valid_lap", "is_curve", "curve_id", "sector",
]

def init_core_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CORE_COLUMNS)

def normalize_controls(df: pd.DataFrame, max_steer_deg: float = None) -> pd.DataFrame:
    df["throttle"] = df["throttle"].astype(float).clip(0, 1)
    df["brake"]    = df["brake"].astype(float).clip(0, 1)
    # se arriva steer in gradi da AC, normalizza in [-1,1]
    if max_steer_deg is not None and "steer" in df.columns:
        df["steer"] = (df["steer"] / max_steer_deg).clip(-1, 1)
    return df

def add_derived_features(df: pd.DataFrame, vmax: float = 300.0) -> pd.DataFrame:
    df = df.copy()
    df["speed_norm"] = (df["speed_kmh"] / vmax).clip(0, 1)
    df["acc_long"] = df["acc_g_x"] * 9.81
    df["acc_lat"]  = df["acc_g_y"] * 9.81
    return df
