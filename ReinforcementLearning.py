import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from pyaccsharedmemory import accSharedMemory
import vgamepad as vg

# ========== COSTANTI ==========
MAX_STEER_DEG = 540.0
TARGET_SPEED_KMH = 150.0  # Velocità che l'IA proverà a mantenere/raggiungere
REFRESH_RATE = 0.05  # 20 Hz (50ms per step)


class AssettoCorsaEnv(gym.Env):
    """Ambiente Custom per Assetto Corsa compatibile con Stable Baselines 3"""

    def __init__(self):
        super(AssettoCorsaEnv, self).__init__()

        # Connessione ad AC
        self.asm = self._connect_shared_memory()

        # Controller virtuale Xbox 360 (Verrà visto da AC come un joypad)
        self.gamepad = vg.VX360Gamepad()

        # ========== SPAZIO DELLE AZIONI ==========
        # Array di 3 valori continui tra -1.0 e 1.0: [Sterzo, Acceleratore, Freno]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # ========== SPAZIO DELLE OSSERVAZIONI (Stato) ==========
        # Array di 5 valori continui: [speed_norm, norm_pos, vx, vy, rpm_norm]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

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
        if sm is None or sm.Physics is None:
            return np.zeros(5, dtype=np.float32)

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

        return np.array([speed_norm, norm_pos, vx / 100.0, vy / 100.0, rpm_norm], dtype=np.float32)

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

        reward = 0.0
        terminated = False

        # Premio per la velocità (incoraggia l'IA ad andare avanti)
        reward += speed_kmh / 10.0

        # Penalità estreme
        if is_off_track:
            reward -= 50.0
            terminated = True  # Fine dell'episodio se esce di pista

        if speed_kmh < 2.0:
            reward -= 1.0  # Penalità per lo stallo

        return reward, terminated

    def reset(self, seed=None, options=None):
        """Riporta l'ambiente allo stato iniziale.
        In AC questo è complesso, richiederebbe una mod o la pressione del tasto 'Restart'
        tramite tastiera virtuale. Qui resettiamo semplicemente i controlli."""
        super().reset(seed=seed)

        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=0.0)
        self.gamepad.left_trigger_float(value_float=0.0)
        self.gamepad.update()

        # Attesa per stabilizzare l'auto (idealmente qui c'è un input per riavviare la sessione)
        time.sleep(2)

        return self._get_state(), {}

    def close(self):
        self.asm.close()


# ========== MAIN TRAINING LOOP ==========
if __name__ == "__main__":
    print("[+] Inizializzazione Ambiente Assetto Corsa...")
    env = AssettoCorsaEnv()

    # Inizializziamo l'agente PPO (Proximal Policy Optimization)
    # È l'algoritmo standard per il controllo continuo (guida, robotica)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, device="cuda")

    print("[!] Assicurati di essere in pista su Assetto Corsa.")
    print("[!] Vai nelle impostazioni del gioco e seleziona il controller Xbox 360 come input.")
    print("[+] Inizio addestramento (Premi Ctrl+C per fermare e salvare)...")

    try:
        # Avvia l'apprendimento per 100.000 step (circa un'ora e mezza di guida reale)
        model.learn(total_timesteps=100000)
    except KeyboardInterrupt:
        print("\n[!] Addestramento interrotto dall'utente.")
    finally:
        # Salva il modello addestrato
        model.save("ppo_assetto_corsa")
        print("[+] Modello salvato come 'ppo_assetto_corsa.zip'.")
        env.close()