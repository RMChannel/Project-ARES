import turtle
import read_ai
import utils.driver as driver

# 1. Inizializziamo il lettore dei dati fisici dal tuo driver
rd = driver.AssettoCorsaData()
rd.start()


def draw_circuit(points):
    if not points:
        print("No points found to draw.")
        return

    screen = turtle.Screen()
    screen.title("Assetto Corsa - Live Telemetry Viewer")
    screen.bgcolor("black")

    # Spegniamo temporaneamente le animazioni per disegnare la pista istantaneamente
    screen.tracer(0)

    # 2. Setup delle coordinate (X orizzontale, Z verticale)
    min_x = min(p.x for p in points)
    max_x = max(p.x for p in points)
    min_y = min(p.z for p in points)
    max_y = max(p.z for p in points)

    pad_x = (max_x - min_x) * 0.1
    pad_y = (max_y - min_y) * 0.1
    screen.setworldcoordinates(min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y)

    # 3. Disegno della traiettoria (AI Line)
    pen = turtle.Turtle()
    pen.speed(0)
    pen.color("cyan")
    pen.pensize(2)
    pen.hideturtle()

    pen.penup()
    pen.goto(points[0].x, points[0].z)
    pen.pendown()

    for p in points[1:]:
        pen.goto(p.x, p.z)

    print(f"Finished drawing {len(points)} points!")

    # Aggiorniamo lo schermo per mostrare la pista completa
    screen.update()

    # 4. Setup del cursore dell'auto (Utente)
    user = turtle.Turtle()
    user.shape("circle")
    user.color("red")
    user.shapesize(0.5, 0.5)  # Rende il pallino un po' più piccolo
    user.penup()  # Importante: non vogliamo che l'auto lasci una scia (a meno che tu non lo voglia)
    user.speed(0)

    # Riattiviamo gli aggiornamenti dello schermo per vedere l'auto muoversi
    screen.tracer(1)

    # 5. Funzione di Update in Tempo Reale
    def update_car():
        # Aggiorniamo i dati fisici (se ti servono velocità, rpm, ecc.)
        rd.update()

        # Preleviamo la posizione usando la funzione standalone del tuo driver
        x, z = driver.get_car_position()

        # Muoviamo l'auto solo se AC sta effettivamente inviando dati
        if x != 0.0 or z != 0.0:
            user.goto(x, z)

        # Richiama questa funzione ogni 50 millisecondi (circa 20 FPS)
        screen.ontimer(update_car, 50)

    # Facciamo partire il loop
    update_car()

    # Sostituiamo exitonclick con mainloop per mantenere vivo l'aggiornamento
    screen.mainloop()


# --- EXECUTION ---
if __name__ == "__main__":
    try:
        # Load the points
        lista_punti = read_ai.get_data("files_ai/fast_lane.ai")

        # Start the drawing process
        draw_circuit(lista_punti)
    finally:
        # Assicuriamoci che la memoria condivisa venga chiusa correttamente
        # anche se chiudi la finestra bruscamente
        rd.stop()