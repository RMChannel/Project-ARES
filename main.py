from pyaccsharedmemory import accSharedMemory
import time
import sys

REFRESH_RATE = 0.05  # 20Hz


def connect_shared_memory():
    try:
        asm = accSharedMemory()
        print("[+] Connessione Shared Memory OK")
        return asm
    except Exception as e:
        print(f"[!] Errore connessione Shared Memory: {e}")
        return None


def main():
    asm = None

    while asm is None:
        asm = connect_shared_memory()
        if asm is None:
            time.sleep(1)

    print("[?] In attesa che ACC sia in pista...")

    try:
        while True:

            try:
                sm = asm.read_shared_memory()
            except Exception:
                print("\n[!] Connessione persa. Reconnect...")
                asm.close()
                time.sleep(1)
                asm = connect_shared_memory()
                continue

            if sm is None:
                print("[-] ACC non rilevato...                   ", end="\r")
                time.sleep(1)
                continue

            graphics = sm.Graphics
            physics = sm.Physics

            if graphics is None or physics is None:
                time.sleep(0.2)
                continue

            # Status sessione
            try:
                status_val = int(graphics.status)
            except:
                status_val = getattr(graphics.status, 'value', 0)

            if status_val < 2:
                print("[-] Non in sessione LIVE...               ", end="\r")
                time.sleep(0.5)
                continue

            # Lettura dati
            speed = getattr(physics, "speed_kmh", 0.0)
            rpm = getattr(physics, "rpm", 0)
            gear_raw = getattr(physics, "gear", 1)

            # Se RPM è troppo basso, probabilmente non sei ancora attivo
            if rpm < 500:
                print("[-] Auto non ancora attiva...             ", end="\r")
                time.sleep(0.2)
                continue

            # Conversione marcia
            gear_display = gear_raw - 1
            if gear_display == -1:
                gear_str = "R"
            elif gear_display == 0:
                gear_str = "N"
            else:
                gear_str = str(gear_display)

            # Filtro micro velocità
            if speed < 0.5:
                speed = 0.0

            print(
                f"[LIVE] "
                f"Velocità: {speed:6.1f} km/h | "
                f"RPM: {int(rpm):5} | "
                f"Marcia: {gear_str}     ",
                end="\r"
            )

            time.sleep(REFRESH_RATE)

    except KeyboardInterrupt:
        print("\n[!] Interrotto dall'utente.")
    finally:
        if asm:
            asm.close()
        print("[-] Risorse rilasciate.")


if __name__ == "__main__":
    main()