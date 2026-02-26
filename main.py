# main.py - LIVE AI COACH COMPLETO (fix dim + dati reali)
from pyaccsharedmemory import accSharedMemory
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
from collections import deque


# ========== LSTM MODEL DEFINITION ==========
class ImitationLSTM(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 3), nn.Tanh()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        act = self.fc(out[:, -1, :])
        steer = act[:, 0]
        throttle = (act[:, 1] + 1) / 2
        brake = (act[:, 2] + 1) / 2
        return steer, throttle, brake


# ========== CARICA MODEL ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Usando device: {device}")
model = ImitationLSTM(input_dim=7)
model.load_state_dict(torch.load("src/data/Kaggle_Monza/models/lstm_imitation_trained.pt", map_location=device))
model.to(device)
model.eval()

# ========== BUFFER E PARAMETRI ==========
SEQ_LEN = 20
buffer = deque(maxlen=SEQ_LEN)
REFRESH_RATE = 0.05  # 20Hz
MAX_STEER_DEG = 540.0


def normalize_features(speed, steer, brake, throttle, norm_pos, vx, vy):
    """7 feature esatte per il modello."""
    speed_norm = np.clip(speed / 300.0, 0, 1)
    steer_norm = np.clip(steer / MAX_STEER_DEG, -1.0, 1.0) if abs(steer) > 2 else float(steer)
    return np.array([speed_norm, steer_norm, brake, throttle, norm_pos, vx, vy], dtype=np.float32)


def get_ai_suggestion(buffer):
    if len(buffer) < SEQ_LEN:
        return None, None, None

    seq = np.array(list(buffer))
    with torch.no_grad():
        seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        steer_ai, thr_ai, brk_ai = model(seq_t)
        return steer_ai.cpu().numpy()[0], thr_ai.cpu().numpy()[0], brk_ai.cpu().numpy()[0]


def connect_shared_memory():
    try:
        asm = accSharedMemory()
        print("[+] Connessione Shared Memory OK")
        return asm
    except Exception as e:
        print(f"[!] Errore: {e}")
        return None


def main():
    asm = None
    norm_pos = 0.0

    while asm is None:
        asm = connect_shared_memory()
        if asm is None:
            time.sleep(1)

    print("[?] In attesa ACC in pista...")

    try:
        while True:
            try:
                sm = asm.read_shared_memory()
            except:
                print("\n[!] Reconnect...")
                if asm:
                    asm.close()
                time.sleep(1)
                asm = connect_shared_memory()
                continue

            if sm is None:
                print("[-] ACC non rilevato...                   ", end="\r")
                time.sleep(1)
                continue

            graphics = sm.Graphics
            physics = sm.Physics

            if graphics is None or physics is None:
                time.sleep(0.2)
                continue

            try:
                status = int(graphics.status)
            except:
                status = getattr(graphics.status, 'value', 0)

            if status < 2:
                print("[-] Non in sessione LIVE...               ", end="\r")
                time.sleep(0.5)
                continue

            # ========== DATI REALI ==========
            speed = getattr(physics, "speed_kmh", 0.0)
            rpm = getattr(physics, "rpm", 0)
            gear_raw = getattr(physics, "gear", 1)

            # CONTROLLI REALI
            throttle = getattr(physics, "gas", 0.0)
            brake = getattr(physics, "brake", 0.0)
            steer_raw = getattr(physics, "steering", 0.0)  # nome esatto da pyaccsharedmemory

            # Velocità locali (se disponibili)
            vx = getattr(physics, "localVelocityX", speed)
            vy = getattr(physics, "localVelocityY", 0.0)

            # Norm pos
            norm_pos = getattr(graphics, "normalizedCarPosition", norm_pos)

            if rpm < 500:
                print("[-] Auto non attiva...                    ", end="\r")
                time.sleep(0.2)
                continue

            gear_display = gear_raw - 1 if gear_raw > 1 else ("R" if gear_raw <= 0 else "N")

            # ========== AI BUFFER ==========
            feats = normalize_features(speed, steer_raw, brake, throttle, norm_pos, vx, vy)
            buffer.append(feats)

            # ========== SUGGERIMENTO AI ==========
            steer_ai, thr_ai, brk_ai = get_ai_suggestion(buffer)

            if steer_ai is not None:
                # Delta rispetto a quello che stai facendo
                delta_steer = steer_ai - (steer_raw / MAX_STEER_DEG if abs(steer_raw) > 2 else steer_raw)
                delta_thr = thr_ai - throttle
                delta_brk = brk_ai - brake

                ai_msg = (f"AI:S{steer_ai:+.2f}Δ{delta_steer:+.2f} "
                          f"T{thr_ai:.1f}Δ{delta_thr:+.1f} "
                          f"B{brk_ai:.1f}Δ{delta_brk:+.1f} | "
                          f"{speed:4.0f}kmh {rpm:4.0f}rpm G{gear_display}")
            else:
                ai_msg = f"AI:warm-up({len(buffer)}/{SEQ_LEN}) | {speed:4.0f}kmh {rpm:4.0f}rpm G{gear_display}"

            print(ai_msg, end="\r")
            time.sleep(REFRESH_RATE)

    except KeyboardInterrupt:
        print("\n[!] Ferma coach")
    finally:
        if asm:
            asm.close()
        print("\n[-] Chiuso")


if __name__ == "__main__":
    main()
