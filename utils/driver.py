import mmap
import struct
import numpy as np
from scipy.spatial import KDTree
import pyautogui


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _convert_degree_arc_to_percent(value):
    """Converte gradi d'arco in percentuale (0..1)."""
    return max(value / 360.0, 0.0)


# ---------------------------------------------------------------------------
# Costanti Shared Memory Assetto Corsa
# ---------------------------------------------------------------------------

_SM_PHYSICS   = "Local\\acpmf_physics"
_SM_GRAPHICS  = "Local\\acpmf_graphics"
_SM_STATIC    = "Local\\AcTools.Static"

# Offset nella SM graphics per carCoordinates[3] (posizione WORLD della macchina)
# Struttura SPageFileGraphic (MSVC packed, wchar_t = 2 byte su Windows):
#   0  : packetId (int 4)
#   4  : status   (int 4)
#   8  : session  (int 4)
#   12 : currentTime  wchar_t[15] = 30 byte
#   42 : lastTime     wchar_t[15] = 30 byte
#   72 : bestTime     wchar_t[15] = 30 byte
#   102: split        wchar_t[15] = 30 byte
#   132: completedLaps, position, iCurrentTime, iLastTime, iBestTime  (5x int = 20)
#   152: sessionTimeLeft, distanceTraveled (2x float = 8)
#   160: isInPit, currentSectorIndex, lastSectorTime, numberOfLaps (4x int = 16)
#   176: tyreCompound wchar_t[33] = 66 byte  → fine a 242
#   242: 2 byte padding (allineamento float a 4 byte)
#   244: replayTimeMultiplier (float 4)
#   248: normalizedCarPosition (float 4)
#   252: carCoordinates[3]  ← X, Y, Z in world space  ← QUESTO vogliamo
_GRAPHICS_CAR_POS_OFFSET = 252   # byte offset per carCoordinates[3] (X, Y, Z) — float


# ---------------------------------------------------------------------------
# Lettura Track Name da AC_SM_Static
# ---------------------------------------------------------------------------

def get_track_name(fallback: str = "monza") -> str:
    """
    Legge la Shared Memory Static di Assetto Corsa per ottenere il nome della pista.
    Ritorna il fallback se la SM non è disponibile.
    """
    try:
        shm = mmap.mmap(-1, 512, _SM_STATIC, access=mmap.ACCESS_READ)
        # Track name in SM Static è una stringa Unicode (UTF-16-LE) di max 30 char
        # a partire dall'offset 20 (dopo il campo smVersion e acVersion)
        raw = shm[20:80]
        shm.close()
        # Split sul primo null-word e decodifica
        name = raw.split(b'\x00\x00')[0].decode('utf-16-le', errors='ignore').strip()
        return name if name else fallback
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Parsing file AI line (.ai) di Assetto Corsa
# ---------------------------------------------------------------------------

def load_ai_line(filepath: str) -> np.ndarray:
    """
    Legge il file binario fast_lane.ai di AC e restituisce un array numpy (N, 7).
    Ogni riga contiene: [x, y, z, speed, ?, ?, ?].
    """
    points = []
    with open(filepath, "rb") as f:
        num_points = struct.unpack('i', f.read(4))[0]
        for _ in range(num_points):
            data = struct.unpack('fffffff', f.read(28))
            points.append(data)
    return np.array(points, dtype=np.float32)


def build_kdtree(ai_line: np.ndarray) -> KDTree:
    """
    Costruisce un KDTree 2D (X, Z) dalla linea AI per query di distanza rapide.
    """
    return KDTree(ai_line[:, [0, 2]])


# ---------------------------------------------------------------------------
# Sistema a Checkpoint progressivi sulla AI line
# ---------------------------------------------------------------------------

class CheckpointSystem:
    """
    Traccia il progresso dell'agente lungo la AI line come sequenza di checkpoint.

    La AI line è un array (N, 7) dove:
      col 0, 1, 2 = X, Y, Z
      col 3       = velocità target AC per quel punto (km/h)
      col 4,5,6   = dati direzione

    Logica:
      - Ogni step l'agente si trova sull'indice più vicino (nearest_idx).
      - Se nearest_idx avanza rispetto all'ultimo checkpoint → reward di progress.
      - Se l'agente si trova INDIETRO rispetto all'ultimo checkpoint di più di
        BACKTRACK_TOLERANCE punti → penalità (sta andando al contrario).
      - Il corner detector guarda in avanti LOOKAHEAD_POINTS passi e trova
        il primo punto dove la velocità target scende di CORNER_SPEED_DROP_PCT.
        Ritorna: distanza (m) e velocità target (km/h) alla curva.
    """

    # --- Parametri ---
    CHECKPOINT_STEP     = 15    # ogni quanti punti AI viene definito un checkpoint
    BACKTRACK_TOLERANCE = 25    # quanti punti "indietro" tolleriamo prima di penalizzare
    LOOKAHEAD_POINTS    = 120   # punti da guardare avanti per trovare una curva
    CORNER_DROP_PCT     = 0.18  # calo di velocità target per considerare "curva" (18%)

    def __init__(self, ai_line: np.ndarray, kdtree: KDTree):
        self.ai_line  = ai_line
        self.kdtree   = kdtree
        self.n_points = len(ai_line)

        # Precomputa le posizioni 2D (X, Z) per il calcolo distanze inter-punto
        self.positions_xz = ai_line[:, [0, 2]]

        # Precomputa distanza cumulativa lungo la linea (metri)
        diffs = np.diff(self.positions_xz, axis=0)
        seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        self.cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        # Indice dell'ultimo checkpoint raggiunto
        self.last_idx = 0
        self.laps_completed = 0

    def reset(self):
        """Resetta il tracker a inizio pista."""
        self.last_idx = 0
        self.laps_completed = 0

    def get_ideal_heading(self, nearest_idx: int) -> float:
        """
        Calcola l'angolo di direzione ideale (in radianti) della AI line
        al punto nearest_idx, usando la differenza centrale tra il punto
        precedente e quello successivo (tangente smooth).
        L'angolo e' nel piano XZ: atan2(dx, dz).
        """
        n   = self.n_points
        idx_prev = (nearest_idx - 1) % n
        idx_next = (nearest_idx + 1) % n
        dx = float(self.ai_line[idx_next, 0] - self.ai_line[idx_prev, 0])
        dz = float(self.ai_line[idx_next, 2] - self.ai_line[idx_prev, 2])
        return float(np.arctan2(dx, dz))   # rad, range [-pi, pi]


    def update(self, x: float, z: float) -> dict:
        """
        Aggiorna il tracker con la posizione corrente dell'agente.
        Ritorna un dizionario con:
          - nearest_idx      : indice del punto AI piu' vicino
          - ideal_heading_rad: angolo tangente della AI line al punto corrente (rad)
          - progress_reward  : reward positivo se l'agente avanza
          - backtrack_penalty: penalita' se torna indietro
          - checkpoint_hit   : True se ha superato un nuovo checkpoint
          - corner_dist_m    : distanza (m) alla prossima curva
          - corner_speed     : velocita' target (km/h) alla curva
        """
        _, nearest_idx = self.kdtree.query([x, z])

        progress_reward   = 0.0
        backtrack_penalty = 0.0
        checkpoint_hit    = False

        # --- Avanzamento ---
        raw_advance = nearest_idx - self.last_idx
        if raw_advance < -(self.n_points // 2):
            raw_advance += self.n_points

        if raw_advance > 0:
            progress_reward = raw_advance / self.n_points * 10.0
            prev_cp = self.last_idx // self.CHECKPOINT_STEP
            curr_cp = (nearest_idx % self.n_points) // self.CHECKPOINT_STEP
            if curr_cp != prev_cp:
                checkpoint_hit = True
            self.last_idx = nearest_idx % self.n_points

        elif raw_advance < -self.BACKTRACK_TOLERANCE:
            backtrack_penalty = -2.0 * abs(raw_advance) / self.n_points * 10.0

        # --- Heading ideale dalla AI line ---
        ideal_heading_rad = self.get_ideal_heading(nearest_idx)

        # --- Corner detection ---
        corner_dist_m, corner_speed = self._find_next_corner(nearest_idx)

        return {
            "nearest_idx"      : nearest_idx,
            "ideal_heading_rad": ideal_heading_rad,
            "progress_reward"  : float(progress_reward),
            "backtrack_penalty": float(backtrack_penalty),
            "checkpoint_hit"   : checkpoint_hit,
            "corner_dist_m"    : float(corner_dist_m),
            "corner_speed"     : float(corner_speed),
        }


    def _find_next_corner(self, from_idx: int) -> tuple[float, float]:
        """
        Cercail prossimo punto in avanti dove la velocità target scende
        di almeno CORNER_DROP_PCT rispetto alla velocità corrente.
        Ritorna (distanza_metri, velocità_target_al_corner).
        """
        n = self.n_points
        current_speed_target = float(self.ai_line[from_idx % n, 3])

        for offset in range(1, self.LOOKAHEAD_POINTS + 1):
            idx = (from_idx + offset) % n
            future_speed = float(self.ai_line[idx, 3])

            # Evita divisione per zero se AI line ha velocità = 0
            if current_speed_target > 1.0:
                drop = (current_speed_target - future_speed) / current_speed_target
            else:
                drop = 0.0

            if drop >= self.CORNER_DROP_PCT:
                # Distanza cumulativa da from_idx a idx
                d_from = self.cum_dist[from_idx % n]
                d_to   = self.cum_dist[idx]
                dist   = d_to - d_from if d_to >= d_from else (
                    self.cum_dist[-1] - d_from + d_to  # wrap-around
                )
                return dist, future_speed

        # Nessuna curva trovata nel lookahead → siamo in rettilineo lungo
        return float(self.cum_dist[min(from_idx + self.LOOKAHEAD_POINTS, n-1)]
                     - self.cum_dist[from_idx]), current_speed_target


# ---------------------------------------------------------------------------
# Lettura posizione auto da acpmf_graphics
# ---------------------------------------------------------------------------

def get_car_position() -> tuple[float, float]:
    """
    Legge X, Z della posizione auto dalla Shared Memory Graphics di AC.
    Usa il campo carCoordinates[3] (posizione world della macchina).
    Ritorna (0.0, 0.0) se la SM non è disponibile (AC non in esecuzione).
    """
    try:
        shm = mmap.mmap(-1, _GRAPHICS_CAR_POS_OFFSET + 12, _SM_GRAPHICS, access=mmap.ACCESS_READ)
        shm.seek(_GRAPHICS_CAR_POS_OFFSET)
        x, y, z = struct.unpack('fff', shm.read(12))
        shm.close()
        return float(x), float(z)
    except Exception as e:
        print(f"[Driver] WARN get_car_position fallito: {e}")
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Driver principale: lettura telemetria fisica in tempo reale
# ---------------------------------------------------------------------------

class AssettoCorsaData:
    """
    Legge la memoria condivisa acpmf_physics di Assetto Corsa e
    espone tutti i campi telemetrici come attributi (es. self.rpm, self.speed).
    """

    # Nomi campi — stringa originale testata (NON modificare l'ordine)
    FIELDS = (
        'packetId throttle brake fuel gear rpm steerAngle speed velocity1 velocity2 velocity3 '
        'accGX accGY accGZ '
        'wheelSlipFL wheelSlipFR wheelSlipRL wheelSlipRR '
        'wheelLoadFL wheelLoadFR wheelLoadRL wheelLoadRR '
        'wheelsPressureFL wheelsPressureFR wheelsPressureRL wheelsPressureRR '
        'wheelAngularSpeedFL wheelAngularSpeedFR wheelAngularSpeedRL wheelAngularSpeedRR '
        'TyrewearFL TyrewearFR TyrewearRL TyrewearRR '
        'tyreDirtyLevelFL tyreDirtyLevelFR tyreDirtyLevelRL tyreDirtyLevelRR '
        'TyreCoreTempFL TyreCoreTempFR TyreCoreTempRL TyreCoreTempRR '
        'camberRADFL camberRADFR camberRADRL camberRADRR '
        'suspensionTravelFL suspensionTravelFR suspensionTravelRL suspensionTravelRR '
        'drs tc1 heading pitch roll cgHeight '
        'carDamagefront carDamagerear carDamageleft carDamageright carDamagecentre '
        'numberOfTyresOut pitLimiterOn abs1 '
        'kersCharge kersInput automat rideHeightfront rideHeightrear turboBoost ballast '
        'airDensity airTemp roadTemp '
        'localAngularVelX localAngularVelY localAngularVelZ '
        'finalFF performanceMeter engineBrake '
        'ersRecoveryLevel ersPowerLevel ersHeatCharging ersIsCharging kersCurrentKJ '
        'drsAvailable drsEnabled '
        'brakeTempFL brakeTempFR brakeTempRL brakeTempRR clutch '
        'tyreTempI1 tyreTempI2 tyreTempI3 tyreTempI4 '
        'tyreTempM1 tyreTempM2 tyreTempM3 tyreTempM4 '
        'tyreTempO1 tyreTempO2 tyreTempO3 tyreTempO4 '
        'isAIControlled '
        'tyreContactPointFLX tyreContactPointFLY tyreContactPointFLZ '
        'tyreContactPointFRX tyreContactPointFRY tyreContactPointFRZ '
        'tyreContactPointRLX tyreContactPointRLY tyreContactPointRLZ '
        'tyreContactPointRRX tyreContactPointRRY tyreContactPointRRZ '
        'tyreContactNormalFLX tyreContactNormalFLY tyreContactNormalFLZ '
        'tyreContactNormalFRX tyreContactNormalFRY tyreContactNormalFRZ '
        'tyreContactNormalRLX tyreContactNormalRLY tyreContactNormalRLZ '
        'tyreContactNormalRRX tyreContactNormalRRY tyreContactNormalRRZ '
        'tyreContactHeadingFLX tyreContactHeadingFLY tyreContactHeadingFLZ '
        'tyreContactHeadingFRX tyreContactHeadingFRY tyreContactHeadingFRZ '
        'tyreContactHeadingRLX tyreContactHeadingRLY tyreContactHeadingRLZ '
        'tyreContactHeadingRRX tyreContactHeadingRRY tyreContactHeadingRRZ '
        'brakeBias localVelocityX localVelocityY localVelocityZ '
        'P2PActivation P2PStatus currentMaxRpm '
        'mz1 mz2 mz3 mz4 fx1 fx2 fx3 fx4 fy1 fy2 fy3 fy4 '
        'slipRatio1 slipRatio2 slipRatio3 slipRatio4 '
        'slipAngle1 slipAngle2 slipAngle3 slipAngle4 '
        'tcinAction absInAction '
        'suspensionDamage1 suspensionDamage2 suspensionDamage3 suspensionDamage4 '
        'tyreTemp1 tyreTemp2 tyreTemp3 tyreTemp4 waterTemp '
        'brakePressureFL brakePressureFR brakePressureRL brakePressureRR '
        'frontBrakeCompound rearBrakeCompound '
        'padLifeFL padLifeFR padLifeRL padLifeRR '
        'discLifeFL discLifeFR discLifeRL discLifeRR'
    ).split()

    # Layout struct originale testato — corrisponde 1:1 ai FIELDS sopra
    _LAYOUT = 'ifffiiffffffff 4f fffffffffffffffffffffffffffffffffffffffffffiifffiffffffffffffiiiiifiifffffffffffffffffiffffffffffffffffffffffffffffffffffffffffiifffffffffffffffffffffiifffffffffffffiiffffffff'

    # Gruppi di campi da raggruppare in liste [FL, FR, RL, RR]
    _ARRAY_GROUPS = [
        'wheelSlip', 'wheelLoad', 'wheelsPressure', 'wheelAngularSpeed',
        'Tyrewear', 'tyreDirtyLevel', 'TyreCoreTemp', 'camberRAD',
        'suspensionTravel', 'brakeTemp', 'brakePressure', 'padLife', 'discLife',
    ]

    def __init__(self):
        print('[Driver] Inizializzazione AssettoCorsaData...')
        self._fields = self.FIELDS
        self._size = struct.calcsize(self._LAYOUT)
        self._mmap = None

        # Pre-inizializza tutti gli attributi a 0
        for field in self._fields:
            setattr(self, field, 0.0)

    # ------------------------------------------------------------------
    # Connessione
    # ------------------------------------------------------------------

    def start(self):
        """Apre la connessione alla Shared Memory acpmf_physics."""
        print('[Driver] Connessione alla physics SM...')
        if not self._mmap:
            self._mmap = mmap.mmap(-1, self._size, _SM_PHYSICS, access=mmap.ACCESS_READ)

    def stop(self):
        """Chiude la connessione alla Shared Memory."""
        print('[Driver] Chiusura connessione SM...')
        if self._mmap:
            self._mmap.close()
        self._mmap = None

    # ------------------------------------------------------------------
    # Aggiornamento dati
    # ------------------------------------------------------------------

    def update(self):
        """
        Legge la SM physics e aggiorna tutti gli attributi.
        Dopo la chiamata è possibile usare: self.rpm, self.speed, self.numberOfTyresOut, ecc.
        """
        if not self._mmap:
            return

        self._mmap.seek(0)
        raw = self._mmap.read(self._size)
        unpacked = struct.unpack(self._LAYOUT, raw)

        data = {self._fields[i]: v for i, v in enumerate(unpacked)}

        # Raggruppa i campi FL/FR/RL/RR in liste
        for group in self._ARRAY_GROUPS:
            data[group] = []
            for suffix in ('FL', 'FR', 'RL', 'RR'):
                key = group + suffix
                if key in data:
                    data[group].append(_convert_degree_arc_to_percent(data.pop(key)))

        # Imposta attributi dinamicamente
        for key, value in data.items():
            setattr(self, key, value)

    # ------------------------------------------------------------------
    # Proprietà di comodo
    # ------------------------------------------------------------------

    @property
    def car_damage_total(self) -> float:
        """Somma dei 5 valori di danno carrozzeria (0..500 circa). 0 = nessun danno."""
        return (
            getattr(self, 'carDamagefront',  0.0)
            + getattr(self, 'carDamagerear',  0.0)
            + getattr(self, 'carDamageleft',  0.0)
            + getattr(self, 'carDamageright', 0.0)
            + getattr(self, 'carDamagecentre',0.0)
        )

    @property
    def tyres_out(self) -> int:
        """Numero di ruote fuori dalla pista (0..4)."""
        return int(self.numberOfTyresOut)

    @property
    def speed_ms(self) -> float:
        """Velocità in m/s (velocità AC è in km/h)."""
        return self.speed / 3.6


# ---------------------------------------------------------------------------
# Controllo sessione AC (Ctrl+R = restart + click bottone)
# ---------------------------------------------------------------------------

# Coordinata schermo del bottone "Restart" in AC.
# Usa mouse_pos.py per trovarla, poi impostala qui come (X, Y).
# Se None, viene saltato il click e si usa solo Ctrl+R.
AC_RESTART_CLICK_POS: tuple[int, int] | None = None   # es: (960, 540)

AC_RESTART_CLICK_POS = (57,182)

def send_reset_to_ac(delay: float = 0.3) -> None:
    """
    Invia Ctrl+R ad Assetto Corsa per terminare la sessione corrente,
    poi clicca AC_RESTART_CLICK_POS (se configurato) per confermare il restart.

    Come trovare le coordinate:
        python mouse_pos.py
        -> muovi il mouse sul bottone, premi SPAZIO, copia i valori qui sopra.
    """
    pyautogui.hotkey('ctrl', 'r')

    if AC_RESTART_CLICK_POS is not None:
        import time as _t
        _t.sleep(delay)                          # attendi che si apra il menu
        pyautogui.click(*AC_RESTART_CLICK_POS)
        _t.sleep(1)
        pyautogui.click(*AC_RESTART_CLICK_POS)
        _t.sleep(1)
        pyautogui.click(*AC_RESTART_CLICK_POS)
        print(f"[Driver] Click restart @ {AC_RESTART_CLICK_POS}")

if __name__ == '__main__':
    import time

    print(f"Pista corrente: {get_track_name()}")

    reader = AssettoCorsaData()
    reader.start()
    try:
        while True:
            reader.update()
            x, z = get_car_position()
            print(
                f"RPM: {reader.rpm:6.0f} | "
                f"Speed: {reader.speed:6.1f} km/h | "
                f"Gear: {reader.gear} | "
                f"TyresOut: {reader.tyres_out} | "
                f"Pos: ({x:.1f}, {z:.1f})"
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()