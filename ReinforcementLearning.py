import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import vgamepad as vg
import pyautogui

# Importiamo il nuovo driver custom
import driver


distance_done = 0

# ========== COSTANTI ==========
MAX_STEER_DEG = 540.0
TARGET_SPEED_KMH = 150.0
REFRESH_RATE = 0.05
MODEL_PATH = "ppo_assetto_corsa"


class AssettoCorsaEnv(gym.Env):
    """Ambiente Custom per Assetto Corsa compatibile con SB3"""

    def __init__(self):
        super(AssettoCorsaEnv, self).__init__()

        # Inizializza il driver custom
        self.asm = self._connect_shared_memory()

        # Controller virtuale Xbox 360
        self.gamepad = vg.VX360Gamepad()

        # SPAZIO AZIONI: [Sterzo, Acceleratore, Freno] tra -1.0 e 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # SPAZIO OSSERVAZIONI: [speed_norm, vx, vy, rpm_norm] (norm_pos rimosso, serve Graphics SHM)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _connect_shared_memory(self):
        reader = driver.AssettoCorsaData()
        while True:
            try:
                reader.start()
                # Verifica rapida
                reader.update()
                print("[Driver] Connessione Shared Memory OK")
                return reader
            except Exception as e:
                print("[-] In attesa di Assetto Corsa...", end="\r")
                time.sleep(1)

    def _get_state(self):
        """Ottiene lo stato corrente usando la dot notation dal driver."""
        # I valori sono già aggiornati grazie a self.asm.update() chiamato in step()
        speed = getattr(self.asm, "speed", 0.0)
        vx = getattr(self.asm, "localVelocityX", 0.0)
        vy = getattr(self.asm, "localVelocityY", 0.0)
        rpm = getattr(self.asm, "rpm", 0.0)

        # Normalizzazione
        speed_norm = np.clip(speed / 300.0, 0.0, 1.0)
        rpm_norm = np.clip(rpm / 8000.0, 0.0, 1.0)

        return np.array([speed_norm, vx / 100.0, vy / 100.0, rpm_norm], dtype=np.float32)

    def step(self, action):
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

        # 3. Leggi il nuovo stato aggiornando il driver
        self.asm.update()
        next_state = self._get_state()

        # 4. Calcola la Reward
        reward, terminated = self._compute_reward()
        truncated = False
        info = {}

        return next_state, reward, terminated, truncated, info

    def _compute_reward(self):
        """Calcola la ricompensa ottimizzata per accelerazione, direzione e sopravvivenza"""
        speed_kmh = getattr(self.asm, "speed", 0.0)
        rpm = getattr(self.asm, "rpm", 0.0)
        gear = getattr(self.asm, "gear", 0)
        tyres_out = getattr(self.asm, "numberOfTyresOut", 0)
        
        # Velocità locale (Z è longitudinale, X è laterale)
        vz = getattr(self.asm, "localVelocityZ", 0.0)
        
        # Slittamento ruote (per essere "attento" al grip)
        slip = getattr(self.asm, "wheelSlip", [0.0]*4)
        avg_slip = np.mean(np.abs(slip)) if isinstance(slip, list) else 0.0

        # Danni vettura
        dmg_f = getattr(self.asm, "carDamagefront", 0.0)
        dmg_r = getattr(self.asm, "carDamagerear", 0.0)
        dmg_l = getattr(self.asm, "carDamageleft", 0.0)
        dmg_right = getattr(self.asm, "carDamageright", 0.0)

        reward = 0.0
        terminated = False

        # 1. Premio Sopravvivenza (Incentiva a non resettare)
        reward += 10.0

        # 2. Premio Velocità Progressiva (Incentiva ad andare AVANTI veloce)
        # vz è in m/s, lo premiamo molto se positivo
        if vz > 0:
            reward += vz * 10.0 # Premia la velocità in avanti
        else:
            reward -= 20.0 # Penalizza se va all'indietro o è fermo con marcia inserita

        # Aggiungiamo comunque un premio alla velocità scalare per l'accelerazione pura
        reward += speed_kmh * 2.0

        # 3. Penalità Fuoripista
        if tyres_out >= 3:
            print(f"[!] FUORI PISTA! (TyresOut:{tyres_out})")
            reward -= 2000.0
            terminated = True
        elif tyres_out > 0:
            reward -= 200.0

        # 4. Penalità Danni (Massima attenzione)
        if dmg_f > 0 or dmg_r > 0 or dmg_l > 0 or dmg_right > 0:
            print("[!] DANNO RILEVATO!")
            reward -= 5000.0
            terminated = True

        # 5. Controllo Trazione / Attenzione (Slip)
        if avg_slip > 1.0:
            reward -= avg_slip * 5.0 # Penalizza se slitta troppo (spreco energia/perdita controllo)

        # 6. Efficienza marce e RPM
        if speed_kmh > 10:
            if rpm > 6500: reward += 50.0
            if gear >= 2: reward += 20.0
            if rpm < 2500 and gear > 1: reward -= 30.0

        # 7. Penalità Inattività pesante
        if speed_kmh < 2.0:
            reward -= 50.0

        return reward, terminated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Resettiamo i controlli
        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=0.0)
        self.gamepad.left_trigger_float(value_float=0.0)
        self.gamepad.update()

        print("[*] Eseguendo il reboot della sessione...")
        pyautogui.hotkey('ctrl', 'r')
        time.sleep(2)

        target_x = 1335
        target_y = 904
        pyautogui.click(x=target_x, y=target_y, clicks=3, interval=0.1)

        time.sleep(5)

        # Aggiorna subito lo stato prima di restituirlo
        self.asm.update()
        return self._get_state(), {}

    def close(self):
        self.asm.stop()


# ========== MAIN TRAINING LOOP ==========
if __name__ == "__main__":
    print("[+] Inizializzazione Ambiente Assetto Corsa...")
    env = AssettoCorsaEnv()

    if os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"[+] Trovato modello: {MODEL_PATH}.zip. Caricamento in corso...")
        model = PPO.load(MODEL_PATH, env=env, device="cuda")
    else:
        print("[+] Nessun modello trovato. Creazione nuovo agente...")
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0001, device="cuda")

    print("[!] Assicurati di essere in pista.")
    print("[!] Input gioco: Controller Xbox 360.")
    print("[+] Inizio addestramento (Premi Ctrl+C per fermare e salvare)...")

    try:
        model.learn(total_timesteps=100000, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\n[!] Addestramento interrotto.")
    finally:
        model.save(MODEL_PATH)
        print(f"[+] Modello salvato come '{MODEL_PATH}.zip'.")
        env.close()