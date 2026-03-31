import time
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from pyaccsharedmemory import accSharedMemory
import vgamepad as vg
import pyautogui

from utils.driver import AssettoCorsaData, get_car_position
import read_ai as fast_lane_api  # Il tuo lettore di tracciati

# ========== COSTANTI ==========

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEER_DEG = 540.0
TARGET_SPEED_KMH = 150.0
REFRESH_RATE = 0.05  # 20 Hz
MODEL_PATH = "model_creation/pilot_model.pth"
TRACK_PATH = "files_ai/fast_lane.ai"  # <--- INSERISCI IL PERCORSO CORRETTO

MODE = "training"
LR = 5e-6
GAMMA = 0.99
OUT_OF_BOUNDS_DIST = 40.0


# ========== FUNZIONI SUPPORTO PISTA ==========
def load_real_track(file_path):
    print(f"[*] Caricamento tracciato da {file_path}...")
    lista_coordinate = fast_lane_api.get_data(file_path)

    # Coordinate pure
    points = [[c.x, c.z] for c in lista_coordinate]
    track_tensor = torch.tensor(points, dtype=torch.float32, device=DEVICE)

    next_p = torch.roll(track_tensor, -1, dims=0)
    diff = next_p - track_tensor
    track_headings = torch.atan2(diff[:, 1], diff[:, 0])

    print(f"[+] Tracciato caricato: {len(track_tensor)} waypoints.")
    return track_tensor, track_headings


# ========== RETE NEURALE ==========
class PilotNet(nn.Module):
    def __init__(self, input_dim=11):
        super(PilotNet, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 2),
            nn.Tanh()
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.common(x)
        return self.actor(x), self.critic(x)


# ========== AMBIENTE ASSETTO CORSA ==========
class AssettoCorsaEnv:
    def __init__(self):
        self.acc_sm = self._connect_shared_memory()
        self.asm = AssettoCorsaData()
        self.asm.start()

        self.track, self.track_headings = load_real_track(TRACK_PATH)
        self.n_track = len(self.track)

        self._prev_speed = 0.0
        self.gamepad = vg.VX360Gamepad()
        self.debug_info = {}  # <--- DIZIONARIO PER SALVARE I DATI DI DEBUG
        print("[+] Ambiente inizializzato con Visione Pista Attiva")

    def _connect_shared_memory(self):
        while True:
            try:
                asm = accSharedMemory()
                print("[+] Connessione Shared Memory OK")
                return asm
            except:
                print("[-] In attesa di Assetto Corsa...", end="\r")
                time.sleep(1)

    def _get_state(self):
        self.asm.update()
        sm = self.acc_sm.read_shared_memory()

        if sm is None or sm.Graphics is None or sm.Physics is None:
            return torch.zeros(11, dtype=torch.float32, device=DEVICE)

        physics = sm.Physics
        graphics = sm.Graphics

        # Dati Telemetria
        speed_kmh = getattr(self.asm, "speed", 0.0)
        vx = getattr(self.asm, "localVelocityX", 0.0)

        speed_change = speed_kmh - self._prev_speed
        accel_g = speed_change / (REFRESH_RATE * 9.81) if REFRESH_RATE > 0 else 0.0
        self._prev_speed = speed_kmh

        # ==========================================
        # IL NUOVO RADAR: BASATO SULLE COORDINATE SPLINE E BUSSOLA
        cx, cz = get_car_position()
        car_pos = torch.tensor([cx, cz], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        dists = torch.cdist(car_pos, self.track)
        dist_to_center_t, nearest_idx_t = torch.min(dists, dim=1)
        
        dist_to_center = dist_to_center_t.item()
        nearest_idx = nearest_idx_t.item()

        global_vx = physics.velocity.x
        global_vz = physics.velocity.z

        # Bussola coerente
        if speed_kmh > 2.0:
            car_heading = math.atan2(global_vz, global_vx)
        else:
            car_heading = self.track_headings[nearest_idx].item()

        ideal_h = self.track_headings[nearest_idx].item()
        heading_error = (ideal_h - car_heading + math.pi) % (2 * math.pi) - math.pi

        # Look-ahead a diverse distanze (come in training)
        look_aheads = [20, 50, 100, 200]
        curvatures = []
        for la in look_aheads:
            f_idx = (nearest_idx + la) % self.n_track
            f_h = self.track_headings[f_idx].item()
            curv = (f_h - ideal_h + math.pi) % (2 * math.pi) - math.pi
            curvatures.append(curv / math.pi)
        # ==========================================

        # SALVATAGGIO DEI DATI GREZZI PER IL DEBUG ESTREMO
        self.debug_info = {
            "car_spline": graphics.normalized_car_position,
            "heading_deg": math.degrees(car_heading),
            "ideal_heading_deg": math.degrees(ideal_h),
            "heading_error_deg": math.degrees(heading_error),
            "speed": speed_kmh
        }

        # Sincronizzato con GPUSimulator.get_observation in main2.py
        state = torch.tensor([
            speed_kmh * 0.01,
            accel_g,
            -vx * 0.1,  # INVERTITO: Positive = Left nel simulatore, ma vx Positive = Right in AC
            dist_to_center / OUT_OF_BOUNDS_DIST,
            heading_error / math.pi,
            math.sin(car_heading),
            math.cos(car_heading),
            *curvatures
        ], dtype=torch.float32, device=DEVICE)

        return state

    def step(self, action):
        throttle, steer_raw = action

        # Compromesso: 1.8x. Abbastanza per curvare, non troppo per sbandare.
        steer = np.clip(steer_raw * 1.8, -1.0, 1.0)

        self.gamepad.left_joystick_float(x_value_float=float(steer), y_value_float=0.0)

        if throttle >= 0:
            self.gamepad.right_trigger_float(value_float=float(throttle))
            self.gamepad.left_trigger_float(value_float=0.0)
        else:
            self.gamepad.right_trigger_float(value_float=0.0)
            self.gamepad.left_trigger_float(value_float=float(-throttle))

        self.gamepad.update()
        time.sleep(REFRESH_RATE)

        next_state = self._get_state()
        sm = self.acc_sm.read_shared_memory()
        physics = sm.Physics if sm is not None else None
        graphics = sm.Graphics if sm is not None else None

        reward, terminated = self._compute_reward(physics, graphics, next_state)

        return next_state, reward, terminated

    def _compute_reward(self, physics, graphics, state_tensor):
        if graphics is None:
            return 0.0, False

        speed_kmh = getattr(self.asm, "speed", 0.0)
        is_off_track = getattr(self.asm, "numberOfTyresOut", 0) >= 3

        has_damage = (getattr(self.asm, "carDamagefront", 0) > 0 or
                      getattr(self.asm, "carDamageleft", 0) > 0 or
                      getattr(self.asm, "carDamageright", 0) > 0 or
                      getattr(self.asm, "carDamagecentre", 0) > 0 or
                      getattr(self.asm, "carDamagerear", 0) > 0)

        # Ricompensa basata sulla bussola (indice 4 è l'errore di rotta)
        angle_error_norm = abs(state_tensor[4].item())

        reward = (speed_kmh / 30.0) - (angle_error_norm * 2.0)

        terminated = False

        if has_damage:
            reward -= 100.0
            terminated = True

        if is_off_track:
            reward -= 100.0
            terminated = True

        if speed_kmh < 2.0:
            reward -= 2.0

        return reward, terminated

    def reset(self):
        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=0.0)
        self.gamepad.left_trigger_float(value_float=0.0)
        self.gamepad.update()

        self._prev_speed = 0.0

        print("[*] Eseguendo il reboot della sessione...")
        pyautogui.hotkey('ctrl', 'r')
        time.sleep(2)

        target_x = 48
        target_y = 184
        pyautogui.click(x=target_x, y=target_y)
        time.sleep(0.5)
        pyautogui.click(x=target_x, y=target_y)

        time.sleep(5)
        return self._get_state()

    def close(self):
        if hasattr(self, 'asm'):
            self.asm.stop()
        if hasattr(self, 'acc_sm'):
            self.acc_sm.close()


# ========== MAIN TRAINING LOOP ==========
if __name__ == "__main__":
    print("[+] Inizializzazione Ambiente Assetto Corsa...")
    env = AssettoCorsaEnv()

    model = PilotNet(input_dim=11).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if os.path.exists(MODEL_PATH):
        print(f"[+] Trovato modello salvato: {MODEL_PATH}. Caricamento in corso...")
        checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("[+] Nessun modello trovato. Utilizzo modello con pesi casuali.")

    if MODE == "inference":
        model.eval()
    else:
        model.train()

    print(f"[+] Inizio guida con PilotNet ({MODE}) - Premi Ctrl+C per fermare...")

    try:
        episode_count = 0

        while True:
            state = env.reset()
            states_buffer, actions_buffer, rewards_buffer, log_probs_buffer, values_buffer = [], [], [], [], []
            total_reward = 0.0
            step_count = 0
            terminated = False
            aiuto_sterzo = 0.0

            while not terminated:
                state_tensor = state.unsqueeze(0) if state.dim() == 1 else state

                if MODE == "training":
                    action_means, value = model(state_tensor)

                    speed_kmh_current = state_tensor[0, 0].item() * 100.0

                    if episode_count < 50:
                        action_means = action_means.clone()

                        if speed_kmh_current < 35.0:
                            action_means[0, 0] = 0.6

                        # Compromesso maestro: guadagno 1.4, clip 0.7.
                        rotta_error = state_tensor[0, 4].item()
                        aiuto_sterzo = np.clip(rotta_error * 1.4, -0.7, 0.7)

                        peso_maestro = 1.0 - (episode_count / 50.0)
                        action_means[0, 1] = (aiuto_sterzo * peso_maestro) + (action_means[0, 1] * (1.0 - peso_maestro))

                    noise_std = max(0.05, 0.3 * (1.0 - episode_count / 500.0))
                    std = torch.full_like(action_means, noise_std)
                    dist_policy = Normal(action_means, std)
                    action_t = dist_policy.sample()

                    action_t = torch.clamp(action_t, -1.0, 1.0)

                    log_prob = dist_policy.log_prob(action_t).sum(dim=-1)
                    action = action_t.squeeze(0).cpu().numpy()

                    states_buffer.append(state)
                    actions_buffer.append(torch.tensor(action, device=DEVICE))
                    log_probs_buffer.append(log_prob)
                    values_buffer.append(value.squeeze())
                else:
                    with torch.no_grad():
                        action_means, _ = model(state_tensor)
                        action = action_means.squeeze(0).cpu().numpy()

                next_state, reward, terminated = env.step(action)

                if MODE == "training":
                    rewards_buffer.append(reward)

                total_reward += reward
                step_count += 1
                state = next_state

                speed_kmh = getattr(env.asm, "speed", 0.0)

                # ==========================================
                # IL CRUSCOTTO DI DEBUG ESTREMO
                # ==========================================
                if step_count % 10 == 0:
                    d = env.debug_info
                    print(f"\n[🙏 DEBUG GESÙ - STEP {step_count}]")
                    print(f"📍 POS AUTO   : Spline={d['car_spline']:.4f}")
                    print(
                        f"🧭 BUSSOLA    : Auto={d['heading_deg']:6.1f}° | Ideale={d['ideal_heading_deg']:6.1f}° | Errore={d['heading_error_deg']:6.1f}°")
                    print(
                        f"🕹️ AZIONE     : Maestro={aiuto_sterzo:.2f} | Al Gioco={np.clip(action[1] * 3.0, -1, 1):.2f}")
                    print("-" * 50)
                # ==========================================

                if step_count > 250 and speed_kmh < 5.0:
                    print("[!] L'IA si rifiuta di guidare (velocità < 5 km/h). Terminazione forzata!")
                    rewards_buffer[-1] -= 100.0
                    terminated = True

                if step_count > 3000:
                    print("[!] Timeout Episodio! È durato troppo.")
                    terminated = True

            print(f"[!] Episodio {episode_count} terminato in {step_count} step. Reward Totale: {total_reward:.2f}")

            if MODE == "training" and len(rewards_buffer) > 1:
                print("[*] Esecuzione Backpropagation per l'intero episodio...")

                returns = []
                R = 0.0
                for r in reversed(rewards_buffer):
                    R = r + GAMMA * R
                    returns.insert(0, R)

                returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

                values_t = torch.stack(values_buffer)
                log_probs_t = torch.stack(log_probs_buffer)
                advantage = returns - values_t.detach()

                states_t = torch.stack(states_buffer)
                actions_t = torch.stack(actions_buffer)
                action_means_new, values_new = model(states_t)

                std_new = torch.full_like(action_means_new, noise_std)
                dist_new = Normal(action_means_new, std_new)
                new_log_probs = dist_new.log_prob(actions_t).sum(dim=-1)

                actor_loss = -(advantage * new_log_probs).mean()
                critic_loss = nn.MSELoss()(values_new.squeeze(), returns)
                entropy = dist_new.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                print(
                    f"[TRAINING] Loss: {loss.item():.4f} | Actor: {actor_loss.item():.4f} | Critic: {critic_loss.item():.4f}")

            if MODE == "training" and episode_count % 3 == 0:
                torch.save({
                    'epoch': episode_count,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'reward': total_reward,
                }, "ac_pilot_finetuned.pth")
                print(f"[+] Modello salvato: ac_pilot_finetuned.pth")

            episode_count += 1

    except KeyboardInterrupt:
        print("\n[!] Guida interrotta dall'utente.")
    finally:
        env.close()
        print("[+] Sessione terminata.")
