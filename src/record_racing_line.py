"""
record_racing_line.py — Registra la traiettoria ideale da Assetto Corsa
oppure la estrae da un dataset. Supporta multi-circuito.

Uso:
  1) Registrazione live (auto-detect del circuito):
     python src/record_racing_line.py --live

  2) Registrazione live con nome circuito esplicito:
     python src/record_racing_line.py --live --track monza

  3) Estrazione da dataset Kaggle:
     python src/record_racing_line.py --from-kaggle src/data/Kaggle_Monza/parquet/kaggle_monza_gt3.parquet --track monza

  4) Lista circuiti disponibili:
     python src/record_racing_line.py --list
"""

import argparse
import time
import numpy as np
from pathlib import Path


def record_live(track_name: str = None, output_path: str = None,
                duration_laps: int = 1, sample_rate: float = 0.05):
    """
    Registra la traiettoria guidando manualmente in Assetto Corsa.
    Usa la shared memory per leggere carCoordinates e normalizedCarPosition.

    Args:
        track_name: nome del circuito (auto-detected se None)
        output_path: percorso output (auto-generato da track_name se None)
    """
    from pyaccsharedmemory import accSharedMemory
    from racing_line import detect_track_name, get_racing_line_path

    print("[*] Connessione alla shared memory di Assetto Corsa...")
    asm = None
    while asm is None:
        try:
            asm = accSharedMemory()
            print("[+] Connesso!")
        except Exception:
            print("[-] In attesa di Assetto Corsa...", end="\r")
            time.sleep(1)

    # Auto-detect del circuito se non specificato
    if track_name is None:
        track_name = detect_track_name(asm)
        if track_name:
            print(f"[+] Circuito rilevato automaticamente: {track_name}")
        else:
            print("[!] Impossibile rilevare il circuito. Usa --track <nome>")
            asm.close()
            return

    # Determina il percorso di output
    if output_path is None:
        output_path = str(get_racing_line_path(track_name))
    print(f"[*] Output: {output_path}")

    points = []
    norm_positions = []
    laps_completed = 0
    last_norm_pos = 0.0
    recording = False

    print(f"\n[*] Circuito: {track_name}")
    print(f"[*] Pronto! Guida un giro completo del circuito.")
    print(f"[*] La registrazione inizia quando attraversi il traguardo.")
    print(f"[*] Premi Ctrl+C per fermare manualmente.\n")

    try:
        while laps_completed < duration_laps:
            sm = asm.read_shared_memory()
            if sm is None or sm.Graphics is None:
                time.sleep(sample_rate)
                continue

            graphics = sm.Graphics
            norm_pos = getattr(graphics, 'normalized_car_position', 0.0)
            coords_list = getattr(graphics, 'car_coordinates', None)
            car_ids = getattr(graphics, 'car_id', None)
            player_id = getattr(graphics, 'player_car_id', 0)

            coords = [0.0, 0.0, 0.0]
            if coords_list and car_ids:
                try:
                    # Trova l'indice corretto per il player usando l'array car_id
                    idx = car_ids.index(player_id)
                    car = coords_list[idx]
                    coords = [float(car.x), float(car.y), float(car.z)]
                except (ValueError, IndexError):
                    if len(coords_list) > player_id:
                        car = coords_list[player_id]
                        coords = [float(car.x), float(car.y), float(car.z)]

            # Rileva inizio giro (attraversamento traguardo: norm_pos torna vicino a 0)
            if not recording and norm_pos < 0.05:
                recording = True
                print(f"[+] Registrazione giro {laps_completed + 1} iniziata!")

            if recording:
                points.append(coords)
                norm_positions.append(float(norm_pos))

                # Stampa progresso ogni 5%
                if int(norm_pos * 20) != int(last_norm_pos * 20):
                    print(f"    Progresso: {norm_pos * 100:.1f}%  |  Punti: {len(points)}")

                # Rileva fine giro
                if last_norm_pos > 0.95 and norm_pos < 0.05:
                    laps_completed += 1
                    print(f"[+] Giro {laps_completed} completato! ({len(points)} punti)")

                    if laps_completed < duration_laps:
                        recording = True  # Continua col prossimo giro

            last_norm_pos = norm_pos
            time.sleep(sample_rate)

    except KeyboardInterrupt:
        print(f"\n[!] Registrazione interrotta manualmente. Punti raccolti: {len(points)}")

    if len(points) < 10:
        print("[!] Troppi pochi punti, registrazione non salvata.")
        asm.close()
        return

    # Salva
    points_arr = np.array(points, dtype=np.float64)
    norm_pos_arr = np.array(norm_positions, dtype=np.float64)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, points=points_arr, norm_pos=norm_pos_arr)

    print(f"\n[+] Racing line per '{track_name}' salvata: {len(points_arr)} punti → {output}")
    print(f"    Range X: [{points_arr[:, 0].min():.1f}, {points_arr[:, 0].max():.1f}]")
    print(f"    Range Z: [{points_arr[:, 2].min():.1f}, {points_arr[:, 2].max():.1f}]")

    asm.close()


def extract_from_kaggle(parquet_path: str, track_name: str, output_path: str = None):
    """Estrae la racing line da un dataset parquet."""
    from racing_line import extract_racing_line_from_parquet, get_racing_line_path

    if output_path is None:
        output_path = str(get_racing_line_path(track_name))

    extract_racing_line_from_parquet(parquet_path, output_path, best_lap_only=True)


def main():
    parser = argparse.ArgumentParser(
        description="Registra o estrai la racing line per Project ARES (multi-circuito)"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--live', action='store_true',
                      help='Registra la traiettoria guidando manualmente in AC')
    mode.add_argument('--from-kaggle', type=str, metavar='PARQUET_PATH',
                      help='Estrai la traiettoria da un dataset (.parquet)')
    mode.add_argument('--list', action='store_true',
                      help='Mostra i circuiti per cui esiste una racing line')

    parser.add_argument('--track', type=str, default=None,
                        help='Nome del circuito (es. monza, spa, imola). Auto-rilevato in modalità --live')
    parser.add_argument('--output', type=str, default=None,
                        help='Percorso di output .npz (default: src/data/racing_lines/<track>.npz)')
    parser.add_argument('--laps', type=int, default=1,
                        help='Numero di giri da registrare in modalità live (default: 1)')

    args = parser.parse_args()

    if args.list:
        from racing_line import list_available_racing_lines
        tracks = list_available_racing_lines()
        if tracks:
            print(f"Racing line disponibili ({len(tracks)}):")
            for t in sorted(tracks):
                print(f"  • {t}")
        else:
            print("Nessuna racing line disponibile. Registrane una con --live o --from-kaggle")
        return

    if args.live:
        record_live(track_name=args.track, output_path=args.output, duration_laps=args.laps)
    else:
        if args.track is None:
            parser.error("--track è obbligatorio con --from-kaggle (es. --track monza)")
        extract_from_kaggle(args.from_kaggle, args.track, output_path=args.output)


if __name__ == "__main__":
    main()
