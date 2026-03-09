import sys
import time
import numpy as np
from src.environment import ARESEnv, ARESDashboard

def run_map_mode():
    """Modalità solo Mappa e Telemetria (25 parametri)"""
    print("[+] Avvio A.R.E.S. Map Mode...")
    env = ARESEnv(use_shared_memory=True, use_gamepad=False)
    dashboard = ARESDashboard(max_history=2000)
    
    try:
        print("[!] Visualizzazione dati telemetrici attiva. Premi Ctrl+C per uscire.")
        while True:
            # Leggiamo i dati reali
            raw = env._get_telemetry()
            
            # Simuliamo un'azione nulla (o leggiamo quella dell'utente se in pista)
            current_action = [raw['gas'], raw['brake'], 0.0, raw['steerAngle']]
            
            # Aggiorniamo la dashboard con tutti i parametri
            dashboard.update(raw, current_action)
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n[-] Chiusura Map Mode.")
    finally:
        env.close()

def run_coach_mode():
    """Modalità AI Coach (Precedente main.py)"""
    # ... qui andrebbe il codice originale del coach se vuoi mantenerlo ...
    print("[+] Avvio A.R.E.S. AI Coach...")
    # (Inserire qui la logica LSTM se necessario)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "map":
        run_map_mode()
    else:
        print("Utilizzo:")
        print("  python main.py map    -> Visualizza mappa e 25 parametri telemetrici")
        print("  python main.py        -> Avvia AI Coach (default)")
        # Per ora facciamo partire la mappa come default se non specificato
        run_map_mode()
