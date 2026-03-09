import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from pyaccsharedmemory import accSharedMemory
import matplotlib.pyplot as plt
from collections import deque
import vgamepad as vg

class ARESEnv(gym.Env):
    """
    Ambiente Custom per A.R.E.S. basato sulle specifiche di Assetto Corsa e il documento LaTeX.
    """
    def __init__(self, use_shared_memory=True, use_gamepad=True):
        super(ARESEnv, self).__init__()
        
        self.use_shared_memory = use_shared_memory
        self.asm = None
        if self.use_shared_memory:
            self._connect_asm()

        self.gamepad = None
        if use_gamepad:
            self.gamepad = vg.VX360Gamepad()
            print("[+] ARESEnv: Gamepad Virtuale inizializzato")

        # --- COSTANTI DI NORMALIZZAZIONE (Dal LaTeX) ---
        self.V_MAX = 300.0        
        self.RPM_MAX = 9000.0     
        self.WS_MAX = 5.0         
        self.T_CENTER = 87.5      
        self.T_RANGE = 27.5       
        self.G_MAX = 5.0          
        self.GEAR_MAX = 8.0
        self.COORD_MAX = 5000.0   
        self.REFRESH_RATE = 0.05 # 20Hz

        # --- SPAZIO DELLE AZIONI ---
        # [gas, brake, clutch, steerAngle]
        # gas, brake, clutch: [0, 1]
        # steerAngle: [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # --- SPAZIO DEGLI STATI (25 Feature) ---
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(25,), dtype=np.float32
        )

    def _connect_asm(self):
        while self.asm is None:
            try:
                self.asm = accSharedMemory()
                print("[+] ARESEnv: Connessione Shared Memory OK")
            except:
                print("[-] ARESEnv: In attesa di Assetto Corsa...", end="\r")
                time.sleep(1)

    def _get_telemetry(self):
        if not self.use_shared_memory or self.asm is None:
            return self._get_dummy_telemetry()
        
        try:
            sm = self.asm.read_shared_memory()
            if sm is None or sm.Physics is None or sm.Graphics is None:
                return self._get_dummy_telemetry()
            
            p = sm.Physics
            g = sm.Graphics
            
            return {
                'speed_kmh': getattr(p, 'speed_kmh', 0.0),
                'normalizedCarPosition': getattr(g, 'normalizedCarPosition', 0.0),
                'localVelocity': getattr(p, 'localVelocity', [0.0, 0.0, 0.0]),
                'rpm': getattr(p, 'rpm', 0),
                'wheelSlip': getattr(p, 'wheelSlip', [0.0]*4),
                'tyreCoreTemp': getattr(p, 'tyreCoreTemperature', [0.0]*4),
                'accG': getattr(p, 'accG', [0.0]*3),
                'gear': getattr(p, 'gear', 1),
                'numberOfTyresOut': getattr(p, 'numberOfTyresOut', 0),
                'steerAngle': getattr(p, 'steerAngle', 0.0),
                'brake': getattr(p, 'brake', 0.0),
                'gas': getattr(p, 'gas', 0.0),
                'surfaceGrip': getattr(g, 'surfaceGrip', 0.0),
                'carCoordinates': getattr(g, 'carCoordinates', [0.0, 0.0, 0.0])
            }
        except:
            return self._get_dummy_telemetry()

    def _normalize_obs(self, raw):
        obs = np.zeros(25, dtype=np.float32)
        
        obs[0] = raw['speed_kmh'] / self.V_MAX
        obs[1] = raw['normalizedCarPosition']
        obs[2] = raw['localVelocity'][0] / self.V_MAX
        obs[3] = raw['localVelocity'][1] / self.V_MAX
        obs[4] = raw['rpm'] / self.RPM_MAX
        
        for i in range(4):
            obs[5+i] = np.clip(raw['wheelSlip'][i] / self.WS_MAX, -1, 1)
            
        for i in range(4):
            obs[9+i] = np.clip((raw['tyreCoreTemp'][i] - self.T_CENTER) / self.T_RANGE, -1, 1)
            
        for i in range(3):
            obs[13+i] = np.clip(raw['accG'][i] / self.G_MAX, -1, 1)
            
        obs[16] = raw['gear'] / self.GEAR_MAX
        obs[17] = raw['numberOfTyresOut'] / 4.0
        obs[18] = raw['steerAngle'] / np.pi
        obs[19] = raw['brake']
        obs[20] = raw['gas']
        obs[21] = raw['surfaceGrip']
        
        for i in range(3):
            obs[22+i] = np.clip(raw['carCoordinates'][i] / self.COORD_MAX, -1, 1)
            
        return obs

    def _compute_reward(self, telemetry):
        v_kmh = telemetry['speed_kmh']
        reward = v_kmh / 10.0
        done = False
        info = {}

        if telemetry['numberOfTyresOut'] >= 3:
            reward -= 50.0
            done = True
            info['reason'] = 'track_limit'

        if v_kmh < 2.0:
            reward -= 1.0
        if telemetry['rpm'] < 1000.0:
            reward -= 10.0
        if telemetry['gear'] < 2:
            reward -= 5.0

        avg_slip = np.mean(np.abs(telemetry['wheelSlip']))
        if avg_slip > 2.0:
            reward -= (avg_slip * 0.5)

        for temp in telemetry['tyreCoreTemp']:
            if temp < 60.0 or temp > 115.0:
                reward -= 0.1

        return reward, done, info

    def step(self, action):
        gas, brake, clutch, steer = action
        
        if self.gamepad:
            self.gamepad.left_joystick_float(x_value_float=float(steer), y_value_float=0.0)
            self.gamepad.right_trigger_float(value_float=float(gas))
            self.gamepad.left_trigger_float(value_float=float(brake))
            # La frizione (clutch) non è mappata di default sui trigger, la saltiamo o mappiamo su tasto
            self.gamepad.update()

        time.sleep(self.REFRESH_RATE)
        
        raw_telemetry = self._get_telemetry()
        obs = self._normalize_obs(raw_telemetry)
        reward, done, info = self._compute_reward(raw_telemetry)
        
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.gamepad:
            self.gamepad.reset()
            self.gamepad.update()
        
        raw_telemetry = self._get_telemetry()
        return self._normalize_obs(raw_telemetry), {}

    def _get_dummy_telemetry(self):
        return {
            'speed_kmh': 0.0, 'normalizedCarPosition': 0.0,
            'localVelocity': [0.0, 0.0, 0.0],
            'rpm': 0, 'wheelSlip': [0.0]*4,
            'tyreCoreTemp': [85.0]*4,
            'accG': [0.0, 0.0, 0.0], 'gear': 1,
            'numberOfTyresOut': 0, 'steerAngle': 0.0,
            'brake': 0.0, 'gas': 0.0, 'surfaceGrip': 0.0,
            'carCoordinates': [0.0, 0.0, 0.0]
        }

    def close(self):
        if self.asm:
            self.asm.close()

class ARESDashboard:
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.x_history = deque(maxlen=max_history)
        self.z_history = deque(maxlen=max_history)
        
        plt.ion()
        self.fig = plt.figure(figsize=(14, 7))
        
        # Mappa del tracciato (X, Z sono float)
        self.ax_map = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
        self.ax_map.set_title("Mappa Tracciato - Coordinate Reali (Float)")
        self.ax_map.set_facecolor('#1a1a1a')
        self.ax_map.grid(True, color='#333333', linestyle='--')
        self.line_path, = self.ax_map.plot([], [], color='cyan', alpha=0.6, linewidth=1.5, label='Traiettoria')
        self.car_dot, = self.ax_map.plot([], [], 'ro', markersize=10, label='Auto')
        
        # Grafico Input
        self.ax_inputs = plt.subplot2grid((2, 3), (0, 2))
        self.ax_inputs.set_title("Input Agente (Real-time)")
        self.ax_inputs.set_ylim(-1.1, 1.1)
        self.bars = self.ax_inputs.bar(['Gas', 'Brake', 'Steer'], [0, 0, 0], color=['#2ecc71', '#e74c3c', '#3498db'])
        
        # Telemetria 25 parametri (Testo)
        self.ax_text = plt.subplot2grid((2, 3), (1, 2))
        self.ax_text.axis('off')
        self.telemetry_text = self.ax_text.text(0.05, 0.95, "", fontsize=9, family='monospace', 
                                              verticalalignment='top', bbox=dict(facecolor='black', alpha=0.1))
        
        plt.tight_layout()

    def update(self, raw, action):
        # Coordinate Float (X = laterale, Z = profondità in AC)
        x = float(raw['carCoordinates'][0])
        z = float(raw['carCoordinates'][2])
        self.x_history.append(x)
        self.z_history.append(z)
        
        # Aggiornamento grafico con float
        self.line_path.set_data(list(self.x_history), list(self.z_history))
        self.car_dot.set_data([x], [z])
        
        # Auto-zoom dinamico basato sui float reali
        if len(self.x_history) > 2:
            margin = 50.0 # metri
            self.ax_map.set_xlim(min(self.x_history) - margin, max(self.x_history) + margin)
            self.ax_map.set_ylim(min(self.z_history) - margin, max(self.z_history) + margin)
        
        # Barre Input
        self.bars[0].set_height(float(action[0]))
        self.bars[1].set_height(float(action[1]))
        self.bars[2].set_height(float(action[3])) # SteerAngle
        
        # Testo Telemetria 25 Parametri (Precisione Float)
        info = (
            f"--- DINAMICA (1-5) ---\n"
            f"Speed: {raw['speed_kmh']:6.2f} km/h\n"
            f"Pos:   {raw['normalizedCarPosition']:.5f}\n"
            f"V-Loc: X:{raw['localVelocity'][0]:.3f} Y:{raw['localVelocity'][1]:.3f}\n"
            f"RPM:   {raw['rpm']:.0f}\n\n"
            f"--- GRIP & TYRES (6-13) ---\n"
            f"Slip:  {raw['wheelSlip'][0]:.2f}|{raw['wheelSlip'][1]:.2f}|{raw['wheelSlip'][2]:.2f}|{raw['wheelSlip'][3]:.2f}\n"
            f"Temp:  {raw['tyreCoreTemp'][0]:.1f}|{raw['tyreCoreTemp'][1]:.1f}|{raw['tyreCoreTemp'][2]:.1f}|{raw['tyreCoreTemp'][3]:.1f}\n\n"
            f"--- FORZE & STATO (14-22) ---\n"
            f"AccG:  X:{raw['accG'][0]:.2f} Y:{raw['accG'][1]:.2f} Z:{raw['accG'][2]:.2f}\n"
            f"Gear:  {raw['gear']} | TyresOut: {raw['numberOfTyresOut']}\n"
            f"Steer: {raw['steerAngle']:.4f} rad\n"
            f"Brake: {raw['brake']:.2f} | Gas: {raw['gas']:.2f}\n"
            f"Grip:  {raw['surfaceGrip']:.3f}\n\n"
            f"--- POSIZIONE (23-25) ---\n"
            f"Coord X: {x:.3f}\n"
            f"Coord Y: {float(raw['carCoordinates'][1]):.3f}\n"
            f"Coord Z: {z:.3f}"
        )
        self.telemetry_text.set_text(info)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
