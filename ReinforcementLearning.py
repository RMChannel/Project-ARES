import time
import os
from turtle import Terminator

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from pyaccsharedmemory import accSharedMemory
import vgamepad as vg
import pyautogui
import driver

try:
    from src.racing_line import RacingLine, load_racing_line_for_track
except ImportError:
    from racing_line import RacingLine, load_racing_line_for_track

# ========== COSTANTI ==========


MAX_STEER_DEG = 540.0
TARGET_SPEED_KMH = 150.0  # Velocità che l'IA proverà a mantenere/raggiungere
REFRESH_RATE = 0.05  # 20 Hz (50ms per step)
MODEL_PATH = "ppo_assetto_corsa"




class AssettoCorsaEnv(gym.Env):
    """Ambiente Custom per Assetto Corsa compatibile con Stable Baselines 3"""

    def __init__(self):
        super(AssettoCorsaEnv, self).__init__()

        # Connessione ad AC
        self.asm = self._connect_shared_memory()



        # Racing Line (Traiettoria ideale — auto-detect dal circuito corrente)
        self.racing_line = None
        self._prev_progress = 0.0
        self._last_traj_score = None
        self.racing_line = load_racing_line_for_track(asm=self.asm)
        if self.racing_line:
            print("[+] Racing line auto-rilevata dal circuito corrente")
        else:
            print("[!] Racing line non disponibile per questo circuito — reward traiettoria disabilitata")
            print("    Registrala con: python src/record_racing_line.py --live")

        # Controller virtuale Xbox 360 (Verrà visto da AC come un joypad)
        self.gamepad = vg.VX360Gamepad()

        # ========== SPAZIO DELLE AZIONI ==========
        # Array di 3 valori continui tra -1.0 e 1.0: [Sterzo, Acceleratore, Freno]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # ========== SPAZIO DELLE OSSERVAZIONI (Stato) ==========
        # 5 base + 3 traiettoria (distanza, heading error, progresso) = 8
        obs_size = 8 if self.racing_line is not None else 5
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

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
        """Legge i dati dalla telemetria e li normalizza per la rete neurale."""
        sm = self.asm.read_shared_memory()
        obs_size = 8 if self.racing_line is not None else 5
        if sm is None or sm.Physics is None:
            return np.zeros(obs_size, dtype=np.float32)

        physics = sm.Physics
        graphics = sm.Graphics

        speed = getattr(physics, "speed_kmh", 0.0)
        vx = getattr(physics, "localVelocityX", speed)
        vy = getattr(physics, "localVelocityY", 0.0)
        rpm = getattr(physics, "rpm", 0)
        norm_pos = getattr(graphics, "normalizedCarPosition", 0.0)

        # Normalizzazione
        speed_norm = np.clip(speed / 300.0, 0.0, 1.0)
        rpm_norm = np.clip(rpm / 8000.0, 0.0, 1.0)  # Assumiamo max 8000 rpm

        state = [speed_norm, norm_pos, vx / 100.0, vy / 100.0, rpm_norm]

        # Feature traiettoria (se racing line disponibile)
        if self.racing_line is not None:
            car_coords = getattr(graphics, "carCoordinates", [0.0, 0.0, 0.0])
            steer_angle = getattr(physics, "steerAngle", 0.0)
            traj = self.racing_line.compute_trajectory_score(
                car_coords, car_heading_rad=steer_angle
            )
            self._last_traj_score = traj
            state.append(np.clip(1.0 - traj['distance_norm'], -1.0, 1.0))  # Vicinanza
            state.append(np.clip(traj['heading_norm'], -1.0, 1.0))          # Heading error
            state.append(np.clip(traj['progress'], 0.0, 1.0))              # Progresso

        return np.array(state, dtype=np.float32)

    def step(self, action):
        """Esegue l'azione, aspetta un tick, calcola la ricompensa e restituisce il nuovo stato."""
        steer, throttle, brake = action

        # 1. Applica le azioni al controller virtuale
        self.gamepad.left_joystick_float(x_value_float=float(steer), y_value_float=0.0)

        t_val = float(np.clip((throttle + 1) / 2, 0.0, 1.0))
        b_val = float(np.clip((brake + 1) / 2, 0.0, 1.0))
        self.gamepad.right_trigger_float(value_float=t_val)
        self.gamepad.left_trigger_float(value_float=b_val)
        self.gamepad.update()

        # 2. Aspetta che il gioco processi l'input
        time.sleep(REFRESH_RATE)

        # 3. Leggi il nuovo stato e la memoria condivisa
        next_state = self._get_state()
        sm = self.asm.read_shared_memory()

        # FIX: Evitiamo il crash se la memoria condivisa è momentaneamente inaccessibile
        physics = sm.Physics if sm is not None else None
        graphics = sm.Graphics if sm is not None else None

        # 4. Calcola la Reward
        reward, terminated = self._compute_reward(physics, graphics)
        truncated = False
        info = {}

        return next_state, reward, terminated, truncated, info

    def _compute_reward(self, physics, graphics):
        """La funzione vitale: dice all'IA se sta facendo bene o male."""
        if physics is None or graphics is None:
            return 0.0, False

        speed_kmh = getattr(physics, "speed_kmh", 0.0)
        is_off_track = getattr(physics, "numberOfTyresOut", 0) >= 3  # Penalità se esce di pista
        rpm = getattr(physics,"rpm", 0.0)
        gear = getattr(physics,"gear",0)
        car_damage = getattr(physics, "car_damage", None)
        brake = physics.brake


        reward = 0.0
        terminated = False

        # Premio per la velocità (incoraggia l'IA ad andare avanti)
        reward += speed_kmh

        if car_damage.front > 0 or car_damage.left > 0 or car_damage.right > 0 or car_damage.center > 0 or car_damage.rear > 0:
            reward-=100
            terminated=True

        if brake>=0.70:
            reward-=50

        # Penalità estreme
        if is_off_track:
            reward -= 50.0
            terminated = True  # Fine dell'episodio se esce di pista

        if speed_kmh < 2.0:
            reward -= 10.0  # Penalità per lo stallo

        if rpm < 1000:
            reward -= 10.0

        #Facciamo in modo che aumenti la marcia
        if gear < 2:
            reward -= 5.0

        # ========== REWARD TRAIETTORIA ==========
        if self.racing_line is not None and self._last_traj_score is not None:
            traj = self._last_traj_score

            # 1. Premio per vicinanza alla racing line (+5 max)
            proximity_reward = 20.0 * (1.0 - traj['distance_norm']) - 10
            reward += proximity_reward

            # 2. Penalità per heading error (-2 max)
            heading_penalty = -8.0 * abs(traj['heading_norm'])
            reward += heading_penalty

            # 3. Premio per progresso lungo il tracciato
            current_progress = traj['progress']
            delta_progress = current_progress - self._prev_progress

            # Gestione wrap-around (traguardo: 0.99 → 0.01)
            if delta_progress < -0.5:
                delta_progress += 1.0
            elif delta_progress > 0.5:
                delta_progress = 0.0  # Salto anomalo, ignora

            if delta_progress > 0:
                reward += 1.0 * delta_progress * 100.0

            self._prev_progress = current_progress

            # 4. Penalità forte per distanza eccessiva (>15m)
            if traj['distance'] > RacingLine.MAX_DISTANCE:
                reward -= 30.0

        return reward, terminated

    def reset(self, seed=None, options=None):
        """Riporta l'ambiente allo stato iniziale.
        In AC utilizziamo pyautogui per premere il tasto 'Restart' sessione."""
        super().reset(seed=seed)

        # Resettiamo i controlli prima del reboot
        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=0.0)
        self.gamepad.left_trigger_float(value_float=0.0)
        self.gamepad.update()

        # Reset stato traiettoria
        self._prev_progress = 0.0
        self._last_traj_score = None

        # Logica di reboot presa da reboot.py
        print("[*] Eseguendo il reboot della sessione...")
        pyautogui.hotkey('ctrl', 'r')
        time.sleep(2)

        # Clicca sul pulsante di conferma/restart (coordinate da reboot.py)
        target_x = 1323
        target_y = 925
        pyautogui.click(x=target_x, y=target_y)
        pyautogui.click(x=target_x, y=target_y)
        pyautogui.click(x=target_x, y=target_y)
        
        # Attesa per stabilizzare l'auto e caricamento sessione
        time.sleep(5)

        return self._get_state(), {}

    def close(self):
        self.asm.close()


# ========== MAIN TRAINING LOOP ==========
if __name__ == "__main__":
    print("[+] Inizializzazione Ambiente Assetto Corsa...")
    env = AssettoCorsaEnv()

    if os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"[+] Trovato modello salvato: {MODEL_PATH}.zip. Caricamento in corso...")
        model = PPO.load(MODEL_PATH, env=env, device="cuda")
    else:
        print("[+] Nessun modello trovato. Creazione di un nuovo agente...")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, device="cuda")


    print("[!] Assicurati di essere in pista su Assetto Corsa.")
    print("[!] Vai nelle impostazioni del gioco e seleziona il controller Xbox 360 come input.")
    print("[+] Inizio addestramento (Premi Ctrl+C per fermare e salvare)...")

    try:
        # Avvia l'apprendimento per 100.000 step (circa un'ora e mezza di guida reale)
        model.learn(total_timesteps=10000000)
    except KeyboardInterrupt:
        print("\n[!] Addestramento interrotto dall'utente.")
    finally:
        # Salva il modello addestrato
        model.save("ppo_assetto_corsa")
        print("[+] Modello salvato come 'ppo'_assetto_corsa.zip'.")
        env.close()