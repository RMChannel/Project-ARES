import multiprocessing as mp
import queue
import time
import turtle
import numpy as np

def render_worker(track_points, pos_queue, num_agents):
    """Gira su un processo separato per non bloccare la GPU"""
    screen = turtle.Screen()
    screen.title("Project ARES - Async Training Visualizer")
    screen.bgcolor("black")
    screen.tracer(0, 0)

    min_x, min_z = track_points.min(axis=0)
    max_x, max_z = track_points.max(axis=0)

    pad_x = (max_x - min_x) * 0.1
    pad_z = (max_z - min_z) * 0.1
    screen.setworldcoordinates(min_x - pad_x, min_z - pad_z, max_x + pad_x, max_z + pad_z)

    track_pen = turtle.Turtle()
    track_pen.speed("fastest")
    track_pen.color("cyan")
    track_pen.pensize(2)
    track_pen.hideturtle()
    track_pen.penup()
    track_pen.goto(track_points[0][0], track_points[0][1])
    track_pen.pendown()
    for p in track_points[1:]:
        track_pen.goto(p[0], p[1])
    track_pen.goto(track_points[0][0], track_points[0][1])

    agents = []
    colors = ["red", "green", "blue", "yellow", "magenta", "white"]
    for i in range(num_agents):
        t = turtle.Turtle()
        t.shape("circle")
        t.shapesize(0.3)
        t.color(colors[i % len(colors)])
        t.penup()
        agents.append(t)

    screen.update()

    while True:
        try:
            pos_np = None
            # Svuota la coda per prendere solo l'ultimo frame
            while not pos_queue.empty():
                pos_np = pos_queue.get_nowait()

            if pos_np is not None:
                for i in range(num_agents):
                    agents[i].goto(pos_np[i][0], pos_np[i][1])
                screen.update()
            else:
                time.sleep(0.01)
                screen.update()
        except queue.Empty:
            pass
        except Exception:
            break

class AsyncVisualizer:
    def __init__(self, track_tensor, num_agents=20, num_instances=4096):
        self.num_agents = min(num_agents, num_instances)
        self.pos_queue = mp.Queue(maxsize=3)

        track_points = track_tensor.cpu().numpy()

        self.process = mp.Process(
            target=render_worker,
            args=(track_points, self.pos_queue, self.num_agents),
            daemon=True
        )
        self.process.start()

    def update(self, agent_positions):
        pos_np = agent_positions[:self.num_agents].detach().cpu().numpy()
        try:
            self.pos_queue.put_nowait(pos_np)
        except queue.Full:
            pass
