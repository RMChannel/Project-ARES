"""
Sistema di test per modelli addestrati su Assetto Corsa - AGGIORNATO PER IL NUOVO PILOTNET
Permette di testare i modelli generati direttamente su AC.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import sys
from pathlib import Path

# Import delle utility
import read_ai
from utils.driver import (
    AssettoCorsaData,
    get_car_position,
    send_reset_to_ac
)

# Simulatore controller con vgamepad
import vgamepad as vg

# --- DEFINIZIONE RETE (Copiata dall'addestramento per renderlo standalone) ---
class PilotNet(nn.Module):
    def __init__(self, input_dim=7):
        super(PilotNet, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Actor: decide Accelerazione e Sterzo
        self.actor = nn.Sequential(
            nn.Linear(128, 2),
            nn.Tanh()
        )
        # Critic: valuta quanto è buona la situazione attuale
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.common(x)
        return self.actor(x), self.critic(x)
# -----------------------------------------------------------------------------

class ControllerSimulator:
    """Simula input di controller Xbox per Assetto Corsa usando vgamepad."""

    def __init__(self):
        print("[Controller] Inizializzazione controller virtuale (vgamepad)...")
        try:
            self.gamepad = vg.VX360Gamepad()
            print("[Controller] Controller virtuale Xbox 360 creato con successo")

            self.release_all()
            time.sleep(0.1)
            print("[Controller] Test completato ✓")

        except Exception as e:
            print(f"[Controller] ERRORE CRITICO: impossibile creare gamepad virtuale: {e}")
            raise

        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.last_steer = 0.0

    def apply_controls(self, throttle: float, brake: float, steer: float):
        throttle = np.clip(throttle, 0.0, 0.5) # Max 50% di gas come da tuo script originale
        brake = np.clip(brake, 0.0, 1.0)

        # vgamepad vuole un float puro tra -1.0 e 1.0
        steer = np.clip(steer, -1.0, 1.0)

        self.gamepad.right_trigger(value=int(throttle * 255))
        self.gamepad.left_trigger(value=int(brake * 255))
        self.gamepad.left_joystick(x_value=int(steer * 32767), y_value=0)
        self.gamepad.update()

        self.last_throttle = throttle
        self.last_brake = brake
        self.last_steer = steer

    def release_all(self):
        self.gamepad.right_trigger(value=0)
        self.gamepad.left_trigger(value=0)
        self.gamepad.left_joystick(x_value=0, y_value=0)
        self.gamepad.update()


class ModelTester:
    def __init__(self, model_path: str, fast_lane_path: str, device: str = "cuda"):
        print(f"[Test] Inizializzazione tester...")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.ai_coordinates = read_ai.get_data(fast_lane_path)

        self.current_target_idx = 0
        self.search_window = 50
        self.ac_data = AssettoCorsaData()
        self.controller = ControllerSimulator()
        self.printed_at_5s = False

        self.stats = {
            'frames': 0, 'distance_traveled': 0.0, 'max_speed': 0.0,
            'avg_speed': 0.0, 'checkpoints_hit': 0, 'tyres_out_count': 0,
        }

    def _load_model(self, model_path: str) -> PilotNet:
        model = PilotNet(input_dim=7).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model

    def _find_nearest_target(self, car_x: float, car_z: float) -> int:
        n_points = len(self.ai_coordinates)
        min_dist = float('inf')
        nearest_idx = self.current_target_idx

        for offset in range(-self.search_window, self.search_window + 1):
            idx = (self.current_target_idx + offset) % n_points
            target = self.ai_coordinates[idx]
            dist = np.sqrt((car_x - target.x)**2 + (car_z - target.z)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx

        advance = nearest_idx - self.current_target_idx
        if advance > (n_points // 2): advance -= n_points
        elif advance < -(n_points // 2): advance += n_points

        if advance > 0:
            self.current_target_idx = nearest_idx
            self.stats['checkpoints_hit'] += advance

        return nearest_idx

    def _get_ideal_heading(self, idx: int) -> float:
        n = len(self.ai_coordinates)
        prev = self.ai_coordinates[(idx - 1) % n]
        nxt = self.ai_coordinates[(idx + 1) % n]
        dx = nxt.x - prev.x
        dz = nxt.z - prev.z
        return float(np.arctan2(dz, dx))

    def get_observation(self, x: float, z: float) -> tuple[torch.Tensor, dict]:
        nearest_idx = self._find_nearest_target(x, z)
        target = self.ai_coordinates[nearest_idx]

        speed = self.ac_data.speed
        heading = self.ac_data.heading
        lat_vel = self.ac_data.localVelocityX
        accel_g = self.ac_data.accGZ

        dist_to_center = np.sqrt((x - target.x)**2 + (z - target.z)**2)

        ideal_heading_raw = self._get_ideal_heading(nearest_idx)

        # Correzione del sistema di riferimento
        ideal_heading_corrected = ideal_heading_raw - (np.pi / 2)

        # Calcolo dell'errore (angolo verso il centro)
        angle_to_center = (ideal_heading_corrected - heading + np.pi) % (2 * np.pi) - np.pi

        n = len(self.ai_coordinates)
        future_heading = self._get_ideal_heading((nearest_idx + 50) % n)
        next_curve_dir = np.sign((future_heading - ideal_heading_raw + np.pi) % (2 * np.pi) - np.pi)

        # L'ordine deve coincidere ESATTAMENTE con l'addestramento
        obs = np.array([
            speed * 0.01,
            accel_g,
            heading * 0.1,
            lat_vel * 0.1,
            dist_to_center / 50.0,
            angle_to_center / np.pi,
            next_curve_dir
        ], dtype=np.float32)

        debug_info = {
            'ideal_heading_rad': ideal_heading_corrected,
            'car_heading_rad': heading,
            'angle_error_rad': angle_to_center
        }

        return torch.from_numpy(obs).to(self.device), debug_info

    def run(self, duration_seconds: float = 60.0, update_rate: float = 60.0, debug: bool = False):
        print(f"\n[Test] Avvio test per {duration_seconds}s @ {update_rate}Hz")
        time.sleep(3)
        self.ac_data.start()
        time.sleep(0.5)

        dt = 1.0 / update_rate
        start_time = time.time()
        boost_frames = int(update_rate * 2.0)

        try:
            with torch.no_grad():
                while (time.time() - start_time) < duration_seconds:
                    frame_start = time.time()
                    elapsed_total = frame_start - start_time

                    self.ac_data.update()
                    x, z = get_car_position()

                    obs, debug_info = self.get_observation(x, z)

                    # Spacchettiamo l'output della nuova rete
                    action_means, critic_val = self.model(obs.unsqueeze(0))
                    action = action_means.squeeze(0).cpu().numpy()

                    gas_brake_raw = float(action[0])
                    steer_raw = float(action[1])

                    if self.stats['frames'] < boost_frames:
                        throttle, brake = 1.0, 0.0
                    else:
                        if gas_brake_raw > 0:
                            throttle = np.clip(gas_brake_raw, 0.0, 1.0)
                            brake = 0.0
                        else:
                            throttle = 0.0  # ORA SE L'IA FRENA, IL GAS È A ZERO
                            brake = np.clip(abs(gas_brake_raw), 0.0, 1.0)

                    self.controller.apply_controls(throttle, brake, steer_raw)

                    # ---------------------------------------------------------
                    # DEBUG DEGLI ANGOLI
                    # ---------------------------------------------------------
                    if self.stats['frames'] % int(update_rate) == 0 and self.stats['frames'] > 0:
                        err_deg = np.degrees(debug_info['angle_error_rad'])
                        print(f"[ERROR Degree] {err_deg:.2f}° | [⚙️ OUTPUT]: {action.tolist()}")

                    # ---------------------------------------------------------

                    if not self.printed_at_5s and elapsed_total >= 5.0:
                        print("\n" + "="*60)
                        print(f"🕒 SNAPSHOT DATI DOPO 5 SECONDI")
                        print("="*60)
                        print(f"[🧠 INPUT]: {[round(v, 4) for v in obs.cpu().numpy().tolist()]}")
                        print(f"[⚙️ OUTPUT]: {[round(v, 4) for v in action.tolist()]}")
                        print(f"[🎮 CONTROLLI]: Gas {throttle:.2f} | Brake {brake:.2f} | Steer {steer_raw:.2f}")
                        print("="*60 + "\n")
                        self.printed_at_5s = True

                    self._update_stats()

                    elapsed = time.time() - frame_start
                    if elapsed < dt:
                        time.sleep(dt - elapsed)

                    self.stats['frames'] += 1

        except KeyboardInterrupt:
            print("\n[Test] Interruzione utente")
        finally:
            self.controller.release_all()
            self.ac_data.stop()
            self._print_final_stats()

    def _update_stats(self):
        self.stats['max_speed'] = max(self.stats['max_speed'], self.ac_data.speed)
        self.stats['avg_speed'] += self.ac_data.speed

    def _print_final_stats(self):
        frames = max(1, self.stats['frames'])
        print("\n" + "="*40 + "\nSTATISTICHE FINALI\n" + "="*40)
        print(f"Frames: {frames} | Avg Speed: {(self.stats['avg_speed'] / frames):.1f} km/h")


def main():
    parser = argparse.ArgumentParser()
    # Assicurati che il percorso del modello sia corretto
    parser.add_argument('--model', default='pilot_model.pth')
    parser.add_argument('--fast-lane', default='../files_ai/fast_lane.ai')
    parser.add_argument('--duration', type=float, default=60.0)
    parser.add_argument('--rate', type=float, default=60.0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    tester = ModelTester(args.model, args.fast_lane, args.device)
    tester.run(args.duration, args.rate, False)

if __name__ == "__main__":
    main()