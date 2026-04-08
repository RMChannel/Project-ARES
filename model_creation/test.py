import re
import sys
from pathlib import Path

LOG_FILE = "print.txt"
PATTERN  = re.compile(
    r"Epoca\s+(\d+)\s*\|\s*Reward:\s*([-\d.]+)\s*\|\s*Std:\s*([\d.]+)\s*\|\s*Loss:\s*([\d.]+)"
)

# --- soglie diagnostiche ---
STD_COLLAPSED   = 0.08   # std troppo bassa → agente smette di esplorare
STD_EXPLODED    = 1.5    # std troppo alta  → policy caotica
LOSS_EXPLODED   = 20.0   # loss fuori controllo
FLAT_WINDOW     = 50     # epoche consecutive senza miglioramento significativo
FLAT_THRESHOLD  = 0.005  # delta reward minimo per considerarlo "miglioramento"
CRASH_REWARD    = -8.0   # reward così basso → l'agente crasha quasi sempre


def load_log(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                records.append({
                    "epoch":  int(m.group(1)),
                    "reward": float(m.group(2)),
                    "std":    float(m.group(3)),
                    "loss":   float(m.group(4)),
                })
    return records


def diagnose(records):
    if not records:
        print("⚠  Nessuna riga valida trovata in print.txt.")
        return

    n      = len(records)
    last   = records[-1]
    first  = records[0]

    problems = []
    warnings = []
    ok       = []

    # --- 1. Std collassata ---
    if last["std"] < STD_COLLAPSED:
        problems.append(
            f"STD COLLASSATA ({last['std']:.3f} < {STD_COLLAPSED}): "
            f"l'agente ha smesso di esplorare troppo presto. "
            f"Potrebbe essere bloccato in un ottimo locale."
        )
    elif last["std"] > STD_EXPLODED:
        problems.append(
            f"STD ESPLOSA ({last['std']:.3f} > {STD_EXPLODED}): "
            f"la policy è caotica, l'agente non ha ancora imparato nulla di stabile."
        )
    else:
        ok.append(f"Std nella norma ({last['std']:.3f})")

    # --- 2. Loss esplosa ---
    if last["loss"] > LOSS_EXPLODED:
        problems.append(
            f"LOSS ESPLOSA ({last['loss']:.4f} > {LOSS_EXPLODED}): "
            f"possibile instabilità numerica o learning rate troppo alto."
        )
    else:
        ok.append(f"Loss stabile ({last['loss']:.4f})")

    # --- 3. Reward troppo bassa (crash loop) ---
    if last["reward"] < CRASH_REWARD:
        problems.append(
            f"REWARD MOLTO NEGATIVA ({last['reward']:.3f}): "
            f"l'agente crasha quasi sempre. Controlla TRACK_HALF_WIDTH e la posizione di spawn."
        )

    # --- 4. Trend reward (miglioramento nel tempo) ---
    if n >= FLAT_WINDOW:
        window_old = records[-(FLAT_WINDOW)]["reward"]
        window_new = last["reward"]
        delta      = window_new - window_old
        if abs(delta) < FLAT_THRESHOLD:
            warnings.append(
                f"REWARD PIATTA: nelle ultime {FLAT_WINDOW} epoche è cambiata solo di "
                f"{delta:+.4f}. L'agente potrebbe essere in stallo."
            )
        elif delta > 0:
            ok.append(
                f"Reward in crescita nelle ultime {FLAT_WINDOW} epoche "
                f"({window_old:.3f} → {window_new:.3f}, Δ{delta:+.3f})"
            )
        else:
            warnings.append(
                f"REWARD IN CALO nelle ultime {FLAT_WINDOW} epoche "
                f"({window_old:.3f} → {window_new:.3f}, Δ{delta:+.3f}). "
                f"Possibile regressione della policy."
            )

    # --- 5. Fase di apprendimento attuale ---
    r = last["reward"]
    if r < -5:
        fase = "1 – agente casuale (crasha continuamente)"
    elif r < 0.5:
        fase = "2 – impara a stare in pista"
    elif r < 2.0:
        fase = "3 – accelera nella direzione giusta"
    elif r < 5.0:
        fase = "4 – guida accettabile"
    else:
        fase = "5 – guida aggressiva / racing line"

    # --- 6. Velocità di apprendimento globale ---
    total_delta = last["reward"] - first["reward"]
    epoche_tot  = last["epoch"] - first["epoch"]
    if epoche_tot > 0:
        rate = total_delta / epoche_tot
    else:
        rate = 0.0

    # --- STAMPA REPORT ---
    sep = "─" * 60
    print(sep)
    print(f"  ANALISI LOG  —  {n} record  |  epoca {first['epoch']} → {last['epoch']}")
    print(sep)
    print(f"  Ultima epoca : {last['epoch']}")
    print(f"  Reward       : {last['reward']:.3f}")
    print(f"  Std          : {last['std']:.3f}")
    print(f"  Loss         : {last['loss']:.4f}")
    print(f"  Fase stimata : {fase}")
    print(f"  Δ reward/ep  : {rate:+.5f}  (totale {total_delta:+.3f} in {epoche_tot} epoche)")
    print(sep)

    if problems:
        print("  ❌  PROBLEMI RILEVATI:")
        for p in problems:
            print(f"      • {p}")
    if warnings:
        print("  ⚠   ATTENZIONE:")
        for w in warnings:
            print(f"      • {w}")
    if ok:
        print("  ✅  OK:")
        for o in ok:
            print(f"      • {o}")

    if not problems and not warnings:
        print("  Tutto nella norma. Continua il training.")

    print(sep)

    # --- Suggerimenti mirati ---
    tips = []
    if last["std"] < STD_COLLAPSED:
        tips.append("Prova ad aumentare il coefficiente di entropy (0.01 → 0.02) nel loss.")
    if last["std"] > STD_EXPLODED and last["epoch"] > 200:
        tips.append("La std non scende: controlla che il reward non sia sempre vicino a zero.")
    if last["loss"] > LOSS_EXPLODED:
        tips.append("Riduci il learning rate (es. 3e-4 → 1e-4) o abbassa max_grad_norm.")
    if last["reward"] < CRASH_REWARD:
        tips.append("Aumenta TRACK_HALF_WIDTH o riduci la crash penalty temporaneamente.")
    if tips:
        print("  💡  SUGGERIMENTI:")
        for t in tips:
            print(f"      • {t}")
        print(sep)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(LOG_FILE)
    if not path.exists():
        print(f"File non trovato: {path}")
        sys.exit(1)

    records = load_log(path)
    diagnose(records)


if __name__ == "__main__":
    main()
