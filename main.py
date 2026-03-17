"""
main.py - Ambiente RL per Assetto Corsa (Fase 1 - Segui la linea)
=================================================================
ARCHITETTURA:
  - L'agente controlla gas, freno E sterzo (action space 3D)
  - Un controller PD calcola la correzione ideale dello sterzo (pd_steer)
    basata sull'heading error (angolo auto vs tangente della AI line)
  - Lo sterzo finale e' un blend fra quello dell'agente e quello del PD:
      * fuori pista / bordo pista  → il PD prende il controllo
      * in pista e heading ok      → l'agente controlla liberamente
  - L'obiettivo e' far seguire la traiettoria ideale con velocita' ottimale

Tutti i dati vengono letti tramite utils.driver.
"""

import os
import time
import math

import numpy as np
import vgamepad as vg
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from utils.driver import (
    AssettoCorsaData,
    CheckpointSystem,
    get_track_name,
    get_car_position,
    load_ai_line,
    build_kdtree,
    send_reset_to_ac,
)

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------

BASE_TRACK_PATH = "C:/Users/Arjel/Giochi/Assetto Corsa/content/tracks"

# ---------------------------------------------------------------------------
# Parametri di training
# ---------------------------------------------------------------------------

MAX_TYRES_OUT  = 3         # ruote fuori che scatena reset (>= 3)
PENALTY_TYRES  = -200.0   # penalita' piatta per uscita pista
CAR_DAMAGE_MAX = 1         # soglia danno carrozzeria per reset
MAX_DIST_RESET = 800       # distanza (m) dalla linea che forza reset
MAX_RPM_REF    = 8500.0    # RPM max di riferimento
STEP_DELAY     = 0.001     # secondi tra step

# --- Controller PD per la correzione di sicurezza dello sterzo ---
# heading_error (rad) → steer_correction [-1, 1]
# steer = Kp * err + Kd * (err - prev_err)
STEER_KP       = 0.4       # guadagno proporzionale
STEER_KD       = 0.15      # guadagno derivativo (smorzamento oscillazioni)
STEER_MAX      = 1.0       # saturazione uscita PD

# --- Safety blend: quanto il PD sovrascrive l'agente ---
# blend_factor = 0.0 → agente controlla tutto
# blend_factor = 1.0 → PD controlla tutto (override pieno)
# La blend cresce linearmente con dist_norm e |heading_err|
STEER_BLEND_DIST_START = 0.4   # dist/MAX_DIST_RESET oltre cui inizia il blend (40% della zona pericolosa)
STEER_BLEND_DIST_FULL  = 0.85  # dist/MAX_DIST_RESET a cui il PD prende il pieno controllo
STEER_BLEND_HEAD_START = 0.35  # |heading_err|/pi oltre cui l'heading contribuisce al blend
STEER_BLEND_HEAD_FULL  = 0.75  # |heading_err|/pi a cui il PD prende il pieno controllo

# --- Pesi reward ---
W_SPEED        = 1.6     # premia velocita' alta
W_RPM          = 0.6       # premia RPM alti
W_PROGRESS     = 6.0       # premia avanzamento checkpoint
W_BACKTRACK    = 6.0       # penaliza retromarcia
W_LINE         = 100.0      # penaliza distanza dalla linea
LINE_EXP       = 2.5       # esponente penalita' linea
W_HEADING      = 5.0       # penaliza heading error (auto non punta nella dir. giusta)
W_BRAKING      = 5.0       # penaliza entrata troppo veloce in curva
MAX_BRAKE_DECEL= 18.0      # m/s^2 — limite fisico frenata
CORNER_DIST_REF= 200.0     # m ref per normalizzazione curva


# ---------------------------------------------------------------------------
# Ambiente Gymnasium
# ---------------------------------------------------------------------------

class AssettoCorsaEnv(gym.Env):
    """
    Ambiente Fase 1 — Segui la traiettoria.

    Observation (10,):
      [0] speed_norm         velocita' normalizzata     (0..1)
      [1] dist_norm          distanza dalla linea        (0..1+)
      [2] g_lat              accelerazione laterale      (g)
      [3] g_long             accelerazione longitudinale (g)
      [4] rpm_norm           RPM normalizzati            (0..1)
      [5] tyres_out_norm     ruote fuori                 (0..1)
      [6] heading_err_norm   heading error normalizzato  (-1..1)
      [7] corner_dist_norm   distanza prossima curva     (0..1)
      [8] speed_excess_norm  eccesso vel rispetto curva  (0..3)
      [9] blend_factor       quanto il PD sta correggendo(0..1)

    Action (3,):
      [0] throttle  in [0, 1]
      [1] brake     in [0, 1]
      [2] steer     in [-1, 1]  (controllato dall'agente, con safety blend PD)

    Il sistema applica un safety blend: se l'auto si avvicina al bordo o
    ha un alto heading error, il PD sovrascrive gradualmente lo sterzo dell'agente.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # --- Gamepad ---
        self.gamepad = vg.VX360Gamepad()

        # --- Telemetria ---
        self.asm = AssettoCorsaData()
        self.asm.start()

        # --- Pista ---
        self.track_name = get_track_name()
        print(f"[Env] Pista rilevata: {self.track_name}")

        ai_path = os.path.join(BASE_TRACK_PATH, self.track_name, "ai", "fast_lane.ai")
        if not os.path.exists(ai_path):
            raise FileNotFoundError(f"File AI non trovato: {ai_path}")

        self.ai_line    = load_ai_line(ai_path)
        self.kdtree     = build_kdtree(self.ai_line)
        self.checkpoints = CheckpointSystem(self.ai_line, self.kdtree)

        # --- Stato PD controller ---
        self._prev_heading_err = 0.0   # per termine derivativo

        # --- Spazi RL ---
        self.action_space = spaces.Box(
            low  = np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high = np.array([1.0, 1.0,  1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Controller PD (correzione di sicurezza sterzo)
    # ------------------------------------------------------------------

    def _compute_pd_steer(self, heading_err: float) -> float:
        """
        Calcola la correzione PD ideale [-1, 1] basata sull'heading error.
        Questo valore viene usato solo come correzione di sicurezza, non
        come comando diretto.

        heading_err > 0  → auto punta a sinistra della traiettoria → sterza a destra
        heading_err < 0  → auto punta a destra della traiettoria   → sterza a sinistra
        """
        p_term = STEER_KP * heading_err
        d_term = STEER_KD * (heading_err - self._prev_heading_err)
        self._prev_heading_err = heading_err

        return float(np.clip(p_term + d_term, -STEER_MAX, STEER_MAX))

    def _apply_safety_blend(self, agent_steer: float, pd_steer: float,
                            dist: float, heading_err: float) -> tuple[float, float]:
        """
        Combina lo sterzo dell'agente con la correzione PD.
        Ritorna (steer_finale, blend_factor).

        blend_factor = 0  → agente controlla tutto
        blend_factor = 1  → PD sovrascrive completamente
        """
        # Contributo distanza dalla linea
        dist_norm = dist / MAX_DIST_RESET
        blend_dist = float(np.clip(
            (dist_norm - STEER_BLEND_DIST_START) /
            (STEER_BLEND_DIST_FULL - STEER_BLEND_DIST_START),
            0.0, 1.0
        ))

        # Contributo heading error
        head_norm = abs(heading_err) / math.pi
        blend_head = float(np.clip(
            (head_norm - STEER_BLEND_HEAD_START) /
            (STEER_BLEND_HEAD_FULL - STEER_BLEND_HEAD_START),
            0.0, 1.0
        ))

        # Prendi il blend_factor maggiore tra i due contributi
        blend_factor = max(blend_dist, blend_head)

        steer_final = (1.0 - blend_factor) * agent_steer + blend_factor * pd_steer
        steer_final = float(np.clip(steer_final, -1.0, 1.0))
        return steer_final, blend_factor

    # ------------------------------------------------------------------
    # Lettura stato
    # ------------------------------------------------------------------

    def _read_state(self) -> dict:
        """Legge telemetria + posizione + checkpoint info."""
        self.asm.update()
        x, z   = get_car_position()

        #print(x, z)
        dist, _ = self.kdtree.query([x, z])
        cp_info = self.checkpoints.update(x, z)

        # Heading error: differenza angolare auto vs tangente AI line
        # Entrambi in radianti, wrap in [-pi, pi]
        raw_err = self.asm.heading - cp_info["ideal_heading_rad"]
        heading_err = (raw_err + math.pi) % (2 * math.pi) - math.pi

        return {
            "speed"            : float(self.asm.speed),
            "rpm"              : float(self.asm.rpm),
            "g_lat"            : float(self.asm.accGX),
            "g_long"           : float(self.asm.accGY),
            "tyres_out"        : int(self.asm.tyres_out),
            "car_damage"       : float(self.asm.car_damage_total),
            "dist"             : float(dist),
            "heading_err"      : heading_err,         # rad [-pi, pi]
            "ideal_heading_rad": cp_info["ideal_heading_rad"],
            "progress_reward"  : cp_info["progress_reward"],
            "backtrack_penalty": cp_info["backtrack_penalty"],
            "checkpoint_hit"   : cp_info["checkpoint_hit"],
            "corner_dist_m"    : cp_info["corner_dist_m"],
            "corner_speed"     : cp_info["corner_speed"],
        }

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _make_obs(self, state: dict) -> np.ndarray:
        speed_ms        = state["speed"] / 3.6
        corner_speed_ms = state["corner_speed"] / 3.6
        d               = max(state["corner_dist_m"], 1.0)

        required_decel  = max(
            (speed_ms**2 - corner_speed_ms**2) / (2.0 * d), 0.0
        )
        speed_excess = required_decel / MAX_BRAKE_DECEL

        return np.array([
            state["speed"]        / 300.0,             # [0] speed_norm
            state["dist"]         / MAX_DIST_RESET,    # [1] dist_norm
            state["g_lat"],                             # [2] g_lat
            state["g_long"],                            # [3] g_long
            state["rpm"]          / MAX_RPM_REF,        # [4] rpm_norm
            state["tyres_out"]    / 4.0,               # [5] tyres_out_norm
            float(np.clip(state["heading_err"] / math.pi, -1.0, 1.0)),  # [6] heading_err
            state["corner_dist_m"] / CORNER_DIST_REF,  # [7] corner_dist_norm
            float(np.clip(speed_excess, 0.0, 3.0)),    # [8] speed_excess_norm
            state.get("blend_factor", 0.0),            # [9] blend_factor (quanto il PD sta intervenendo)
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, state: dict) -> float:
        """
        Reward Phase 1 — focus su: seguire la linea + velocita' giusta + direzione corretta.
        """
        speed_ms        = state["speed"] / 3.6
        corner_speed_ms = state["corner_speed"] / 3.6
        d               = max(state["corner_dist_m"], 1.0)

        # Positivi
        speed_r    = state["speed"] / 10.0 * W_SPEED
        rpm_r      = (state["rpm"] / MAX_RPM_REF) * W_RPM
        progress_r = state["progress_reward"] * W_PROGRESS

        # Negativi — linea

        #print(state["dist"])
        dist_p      = -(state["dist"] ** LINE_EXP) * W_LINE
        backtrack_p = state["backtrack_penalty"] * W_BACKTRACK

        # Negativi — heading error (penalizza deviation angolare)
        heading_p = -(state["heading_err"] ** 2) * W_HEADING

        # Negativi — braking (entrata curva troppo veloce)
        required_decel = max(
            (speed_ms**2 - corner_speed_ms**2) / (2.0 * d), 0.0
        )
        speed_excess = required_decel / MAX_BRAKE_DECEL
        braking_p = -max((speed_excess - 0.5), 0.0) ** 2 * W_BRAKING

        reward = speed_r + rpm_r + progress_r + dist_p + backtrack_p + heading_p + braking_p

        if state["tyres_out"] >= MAX_TYRES_OUT:
            reward += PENALTY_TYRES
        
        return float(reward)

    # ------------------------------------------------------------------
    # Interfaccia Gymnasium
    # ------------------------------------------------------------------

    def step(self, action):
        # --- Leggi heading error PRIMA di applicare l'azione ---
        self.asm.update()
        x, z = get_car_position()
        cp_info_pre = self.checkpoints.update(x, z)
        raw_err = self.asm.heading - cp_info_pre["ideal_heading_rad"]
        heading_err_pre = (raw_err + math.pi) % (2 * math.pi) - math.pi

        # --- Sterzo: agente + safety blend PD ---
        dist_pre, _ = self.kdtree.query([x, z])
        agent_steer  = float(np.clip(action[2], -1.0, 1.0))   # sterzo dell'agente
        pd_steer     = self._compute_pd_steer(heading_err_pre) # correzione PD
        steer_val, blend_factor = self._apply_safety_blend(
            agent_steer, pd_steer, dist_pre, heading_err_pre
        )

        # --- Applica comandi ---
        self.gamepad.left_joystick_float(x_value_float=steer_val, y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=float(action[0]))   # gas
        self.gamepad.left_trigger_float(value_float=float(action[1]))    # freno
        self.gamepad.update()

        time.sleep(STEP_DELAY)

        # --- Leggi stato post-azione ---
        state  = self._read_state()
        state["blend_factor"] = blend_factor   # aggiunge il blend_factor allo stato
        obs    = self._make_obs(state)
        reward = self._compute_reward(state)

        # --- Log checkpoint ---
        if state["checkpoint_hit"]:
            print(f"[CP] +checkpoint | steer={steer_val:+.3f} "
                  f"(agent={agent_steer:+.3f}, pd={pd_steer:+.3f}, blend={blend_factor:.2f}) | "
                  f"herr={math.degrees(state['heading_err']):+.1f}deg | "
                  f"corner={state['corner_dist_m']:.0f}m@{state['corner_speed']:.0f}km/h")

        # --- Condizioni di terminazione ---
        terminated = False
        if state["tyres_out"] >= MAX_TYRES_OUT:
            print(f"[RESET] {state['tyres_out']} ruote fuori | steer={steer_val:+.3f} (blend={blend_factor:.2f})")
            terminated = True
        if state["car_damage"] >= CAR_DAMAGE_MAX:
            print(f"[RESET] Danno: {state['car_damage']:.1f}")
            terminated = True
        if state["dist"] > MAX_DIST_RESET:
            print(f"[RESET] Dist: {state['dist']:.2f}m")
            terminated = True

        if terminated:
            send_reset_to_ac(2)

        return obs, reward, terminated, False, state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gamepad.reset()
        self.gamepad.update()
        self.checkpoints.reset()
        self._prev_heading_err = 0.0   # resetta termine derivativo PD
        time.sleep(0.5)
        state = self._read_state()
        state["blend_factor"] = 0.0
        return self._make_obs(state), {}

    def close(self):
        self.asm.stop()
        super().close()


# ---------------------------------------------------------------------------
# Entry point — Fine-tuning su AC reale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = AssettoCorsaEnv()

    track        = env.track_name
    model_name   = f"model_{track}"
    model_path   = f"models/{model_name}.zip"

    # Percorsi modelli dal simulatore
    sim_best_path = f"models_sim/ppo_sim_{track}_best/best_model.zip"
    sim_path      = f"models_sim/ppo_sim_{track}.zip"

    os.makedirs("models", exist_ok=True)

    if os.path.exists(model_path):
        # 1) Fine-tuning modello AC reale già esistente
        print(f"[Main] Caricamento modello AC reale: {model_path}")
        model = PPO.load(model_path, env=env)

    elif os.path.exists(sim_best_path):
        # 2) Trasferimento dal BEST model del simulatore
        print(f"[Main] Caricamento best model simulatore: {sim_best_path}")
        print(f"[Main] Fine-tuning su AC reale...")
        model = PPO.load(sim_best_path, env=env)

    elif os.path.exists(sim_path):
        # 3) Trasferimento dal checkpoint finale del simulatore
        print(f"[Main] Caricamento modello simulatore: {sim_path}")
        print(f"[Main] Fine-tuning su AC reale...")
        model = PPO.load(sim_path, env=env)

    else:
        # 4) Nessun modello disponibile — training da zero
        print(f"[Main] Nessun modello trovato — training da zero per: {track}")
        print(f"[Main] Suggerimento: esegui prima 'python train_sim.py' per pre-addestrare nel simulatore")
        model = PPO(
            policy        = "MlpPolicy",
            env           = env,
            verbose       = 0,
            learning_rate = 3e-3,
            n_steps       = 2048,
            batch_size    = 64,
            gamma         = 0.99,
            ent_coef      = 0.01,
            clip_range    = 0.2,
        )

    try:
        model.learn(total_timesteps=1_000_000)
    except KeyboardInterrupt:
        print("\n[Main] Interruzione — salvataggio...")
    finally:
        model.save(model_path)
        env.close()
        print(f"[Main] Modello salvato: {model_path}")