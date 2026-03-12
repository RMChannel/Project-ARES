"""
racing_line.py — Modulo per il tracking della traiettoria ideale di gara.

Carica una racing line (array di punti 3D + normalizedPosition) e fornisce
metodi per calcolare in tempo reale:
  - Distanza laterale (cross-track error) dall'auto alla racing line
  - Heading error tra la direzione dell'auto e la tangente della racing line
  - Punteggio complessivo di traiettoria per il sistema di reward
"""

import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path


class RacingLine:
    """
    Gestisce la traiettoria ideale di gara.

    Attributi:
        points : np.ndarray (N, 3)   — coordinate mondo [x, y, z] della racing line
        norm_pos : np.ndarray (N,)   — posizione normalizzata 0→1 lungo il tracciato
        tangents : np.ndarray (N, 3) — vettori tangenti unitari lungo la racing line
        tree : cKDTree               — struttura per nearest-neighbor queries O(log n)
    """

    # Distanza massima (metri) oltre la quale la penalità è massima
    MAX_DISTANCE = 15.0

    def __init__(self, filepath: str | Path):
        """
        Carica la racing line da file .npz.

        Il file deve contenere:
            - 'points': array (N, 3) con coordinate [x, y, z]
            - 'norm_pos': array (N,) con posizione normalizzata 0→1
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Racing line non trovata: {filepath}")

        data = np.load(filepath)
        self.points = data['points'].astype(np.float64)     # (N, 3)
        self.norm_pos = data['norm_pos'].astype(np.float64)  # (N,)

        if len(self.points) < 3:
            raise ValueError("La racing line deve contenere almeno 3 punti")

        # Calcola i vettori tangenti lungo la racing line
        self.tangents = self._compute_tangents(self.points)

        # Costruisci il KD-Tree per nearest-neighbor veloce
        # Usiamo solo X e Z (piano orizzontale) — Y è l'altitudine
        self.tree = cKDTree(self.points[:, [0, 2]])

        print(f"[+] RacingLine caricata: {len(self.points)} punti da {filepath.name}")

    @staticmethod
    def _compute_tangents(points: np.ndarray) -> np.ndarray:
        """
        Calcola i vettori tangenti unitari per ogni punto della racing line.
        Usa differenze finite centrali (circolari per chiudere il circuito).
        """
        N = len(points)
        tangents = np.zeros_like(points)

        for i in range(N):
            # Differenza circolare: il punto successivo meno il precedente
            next_idx = (i + 1) % N
            prev_idx = (i - 1) % N
            tangent = points[next_idx] - points[prev_idx]

            norm = np.linalg.norm(tangent)
            if norm > 1e-8:
                tangent /= norm

            tangents[i] = tangent

        return tangents

    def nearest_point(self, car_x: float, car_z: float):
        """
        Trova il punto più vicino sulla racing line (nel piano XZ).

        Args:
            car_x: coordinata X dell'auto (mondo)
            car_z: coordinata Z dell'auto (mondo)

        Returns:
            idx: indice del punto più vicino
            distance: distanza in metri (piano XZ)
        """
        distance, idx = self.tree.query([car_x, car_z])
        return idx, float(distance)

    def compute_heading_error(self, idx: int, car_heading_rad: float) -> float:
        """
        Calcola l'errore di heading tra l'auto e la tangente della racing line.

        Args:
            idx: indice del punto più vicino sulla racing line
            car_heading_rad: angolo di heading dell'auto in radianti (yaw)

        Returns:
            heading_error: differenza angolare in radianti [-π, π]
        """
        tangent = self.tangents[idx]

        # Angolo della tangente della racing line nel piano XZ
        racing_heading = np.arctan2(tangent[0], tangent[2])

        # Differenza angolare normalizzata in [-π, π]
        error = racing_heading - car_heading_rad
        error = (error + np.pi) % (2 * np.pi) - np.pi

        return float(error)

    def get_progress(self, idx: int) -> float:
        """Ritorna la posizione normalizzata (0→1) del punto idx sulla racing line."""
        return float(self.norm_pos[idx])

    def compute_trajectory_score(self, car_coords, car_heading_rad: float = None):
        """
        Calcola il punteggio complessivo della traiettoria.

        Args:
            car_coords: [x, y, z] coordinate mondo dell'auto
            car_heading_rad: heading dell'auto in radianti (opzionale)

        Returns:
            dict con:
                'distance'       : distanza dal racing line (m)
                'distance_norm'  : distanza normalizzata [0, 1] (clipped a MAX_DISTANCE)
                'heading_error'  : errore heading in radianti (0 se non fornito)
                'heading_norm'   : heading error normalizzato [-1, 1]
                'progress'       : posizione normalizzata sul tracciato [0, 1]
                'nearest_idx'    : indice del punto più vicino
        """
        car_x = float(car_coords[0])
        car_z = float(car_coords[2])

        # Punto più vicino
        idx, distance = self.nearest_point(car_x, car_z)

        # Distanza normalizzata (0 = sulla racing line, 1 = MAX_DISTANCE o più)
        distance_norm = min(distance / self.MAX_DISTANCE, 1.0)

        # Heading error
        heading_error = 0.0
        heading_norm = 0.0
        if car_heading_rad is not None:
            heading_error = self.compute_heading_error(idx, car_heading_rad)
            heading_norm = heading_error / np.pi  # Normalizzato in [-1, 1]

        # Progresso lungo il tracciato
        progress = self.get_progress(idx)

        return {
            'distance': distance,
            'distance_norm': distance_norm,
            'heading_error': heading_error,
            'heading_norm': heading_norm,
            'progress': progress,
            'nearest_idx': idx,
        }


# ========== GESTIONE MULTI-CIRCUITO ==========

# Directory dove vengono salvate le racing line per ogni circuito
RACING_LINES_DIR = Path(__file__).parent / "data" / "racing_lines"


def detect_track_name(asm=None) -> str | None:
    """
    Rileva il nome del circuito dalla shared memory di Assetto Corsa.

    Args:
        asm: istanza accSharedMemory già connessa (opzionale, ne crea una se None)

    Returns:
        Nome del circuito (es. 'monza', 'spa') o None se non disponibile
    """
    close_after = False
    if asm is None:
        try:
            from pyaccsharedmemory import accSharedMemory
            asm = accSharedMemory()
            close_after = True
        except Exception:
            return None

    try:
        sm = asm.read_shared_memory()
        if sm is None:
            return None

        # Il nome del circuito è in sm.Statics.track
        statics = getattr(sm, 'Statics', None) or getattr(sm, 'Static', None)
        if statics is None:
            return None

        track_name = getattr(statics, 'track', None)
        if track_name:
            # Normalizza: Assetto Corsa usa array a lunghezza fissa con padding \x00
            # Decodificando l'array utf-16 potrebbero esserci null bytes
            track_name_str = str(track_name)
            if '\x00' in track_name_str:
                track_name_str = track_name_str.split('\x00')[0]
            track_name = track_name_str.strip().lower().replace(' ', '_')
            return track_name

        return None
    except Exception:
        return None
    finally:
        if close_after and asm:
            try:
                asm.close()
            except Exception:
                pass


def get_racing_line_path(track_name: str) -> Path:
    """
    Restituisce il percorso del file .npz della racing line per un circuito.

    Args:
        track_name: nome del circuito (es. 'monza', 'spa', 'imola')

    Returns:
        Path al file .npz (potrebbe non esistere ancora)
    """
    return RACING_LINES_DIR / f"{track_name}.npz"


def load_racing_line_for_track(track_name: str = None, asm=None) -> RacingLine | None:
    """
    Carica la racing line per un circuito specifico.
    Se track_name non è fornito, lo rileva dalla shared memory.

    Args:
        track_name: nome del circuito (opzionale, auto-detected se None)
        asm: istanza accSharedMemory (opzionale)

    Returns:
        RacingLine caricata, o None se il file non esiste
    """
    if track_name is None:
        track_name = detect_track_name(asm)
        if track_name is None:
            print("[!] Impossibile rilevare il circuito dalla shared memory")
            return None

    filepath = get_racing_line_path(track_name)

    if not filepath.exists():
        print(f"[!] Racing line non trovata per '{track_name}': {filepath}")
        print(f"    Registrala con: python src/record_racing_line.py --live --track {track_name}")
        return None

    return RacingLine(filepath)


def list_available_racing_lines() -> list[str]:
    """Restituisce la lista dei circuiti per cui esiste una racing line."""
    if not RACING_LINES_DIR.exists():
        return []
    return [f.stem for f in RACING_LINES_DIR.glob("*.npz")]


def extract_racing_line_from_parquet(parquet_path: str | Path, output_path: str | Path,
                                      best_lap_only: bool = True):
    """
    Estrae la racing line dal dataset Kaggle Monza (parquet).

    Args:
        parquet_path: percorso al file .parquet con colonne x_world, y_world, z_world
        output_path: percorso di output per il file .npz
        best_lap_only: se True, usa solo il giro più veloce
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    print(f"[*] Dataset caricato: {len(df)} righe")

    if best_lap_only and 'lap' in df.columns and 't_lap' in df.columns:
        # Trova il giro con il tempo migliore
        lap_times = df.groupby('lap')['t_lap'].max()
        best_lap = lap_times.idxmin()
        df = df[df['lap'] == best_lap].copy()
        print(f"[*] Usando giro migliore: lap {best_lap} ({lap_times[best_lap]:.2f}s)")

    # Ordina per step_idx
    if 'step_idx' in df.columns:
        df = df.sort_values('step_idx').reset_index(drop=True)

    # Estrai coordinate
    points = df[['x_world', 'y_world', 'z_world']].values.astype(np.float64)

    # Calcola posizione normalizzata dalla distanza cumulativa
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative[-1]

    if total_length > 0:
        norm_pos = cumulative / total_length
    else:
        norm_pos = np.linspace(0, 1, len(points))

    # Salva
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, points=points, norm_pos=norm_pos)
    print(f"[+] Racing line salvata: {len(points)} punti → {output_path}")
    print(f"    Lunghezza tracciato: {total_length:.1f} m")

    return points, norm_pos
