"""
main3.py – Ambiente semplificato: Segui la polyline
====================================================
Una macchina virtuale (1.5m x 3m) deve imparare a seguire un percorso
definito da una lista di waypoints modificabili (TRACK_POINTS).
Ambiente completamente su GPU con PPO mini-batch.
Visualizzatore Pygame asincrono che mostra i migliori agenti.

Osservazione (7):
  0  speed_norm           velocità normalizzata (0..~1)
  1  lateral_offset       offset laterale dalla linea / TRACK_HALF_WIDTH
  2  heading_error        errore angolare rispetto al segmento / pi
  3  sin(heading)
  4  cos(heading)
  5  lateral_vel          velocità laterale normalizzata
  6  forward_vel          velocità in avanti normalizzata

Azione (2):
  0  throttle  [-1, 1]  (negativo = freno)
  1  steer     [-1, 1]
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import multiprocessing as mp
import queue
from torch.distributions import Normal

# ========================= CONFIGURAZIONE =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_INSTANCES     = 512
STEPS_PER_EPOCH   = 512
MINI_BATCH_SIZE   = 4096
LR                = 3e-4
GAMMA             = 0.99
EPS_CLIP          = 0.2
K_EPOCHS          = 3
EPOCHS            = 10000
SAVE_PATH         = "pilot_model_v3.pth"

# --- Fisiche ---
CAR_WIDTH         = 1.5    # metri
CAR_LENGTH        = 3.0    # metri
TRACK_HALF_WIDTH  = 5.0    # metri – limiti laterali dalla linea
MAX_SPEED_KMH     = 200.0  # velocità massima raggiungibile
STEER_RATE        = 3.0    # rad/s a steer=1.0
MAX_ACCEL         = 12.0   # m/s² accelerazione massima
MAX_BRAKE         = 10.0   # m/s² frenata massima
DRAG_COEFF        = 0.005  # attrito aerodinamico

# ========================= TRACCIATO (MODIFICA QUI) =========================
# Lista di waypoints [x, y] in metri. Aggiungi/modifica/rimuovi punti a piacere.
# Il percorso segue i segmenti nell'ordine in cui sono elencati.
TRACK_POINTS = [
    [0.0,    0.0],
    [500.0,  0.0],
    [1000.0, 0.0],
    [1500.0, 0.0],
    [2000.0, 0.0],
]

# --- Visualizzazione ---
VISUALIZE         = True
NUM_VIS_AGENTS    = 5      # top 5 migliori agenti
NUM_VIS_AGENTS    = 10     # migliori agenti da mostrare


# ========================= UTILS TRACCIATO =========================
def build_track(points_list):
    """
    Da una lista di punti crea:
      - track: (M, 2) tensor dei waypoints
      - seg_dirs: (M-1, 2) direzioni unitarie dei segmenti
      - seg_normals: (M-1, 2) normali (perpendicolare sinistra)
      - seg_headings: (M-1,) heading di ogni segmento
      - seg_lengths: (M-1,) lunghezza di ogni segmento
      - cum_lengths: (M-1,) distanza cumulativa fino all'inizio di ogni segmento
      - total_length: lunghezza totale del percorso
    """
    track = torch.tensor(points_list, dtype=torch.float32, device=DEVICE)
    M = len(track)
    assert M >= 2, "Servono almeno 2 waypoints"

    diffs = track[1:] - track[:-1]                       # (M-1, 2)
    seg_lengths = torch.norm(diffs, dim=1)                # (M-1,)
    seg_dirs    = diffs / seg_lengths.unsqueeze(1)        # (M-1, 2) unitari
    seg_normals = torch.stack([-seg_dirs[:, 1], seg_dirs[:, 0]], dim=1)  # (M-1, 2)
    seg_headings = torch.atan2(diffs[:, 1], diffs[:, 0])  # (M-1,)

    cum_lengths = torch.zeros(M - 1, device=DEVICE)
    cum_lengths[1:] = torch.cumsum(seg_lengths[:-1], dim=0)
    total_length = seg_lengths.sum().item()

    return track, seg_dirs, seg_normals, seg_headings, seg_lengths, cum_lengths, total_length


# ========================= RETE NEURALE =========================
class PilotNet(nn.Module):
    """
    Input: 7, Hidden: 128→128→64, Actor: Tanh + learnable log_std, Critic: lineare
    """
    def __init__(self, input_dim=7):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
        )
        self.actor_mean    = nn.Sequential(nn.Linear(64, 2), nn.Tanh())
        self.actor_log_std = nn.Parameter(torch.zeros(2))
        self.critic        = nn.Linear(64, 1)

    def forward(self, x):
        feat = self.common(x)
        return self.actor_mean(feat), self.critic(feat)

    def act(self, x):
        mean, value = self.forward(x)
        std   = self.actor_log_std.exp().clamp(0.05, 1.0)
        dist  = Normal(mean, std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), logprob.detach(), value.detach()

    def evaluate(self, x, action):
        mean, value = self.forward(x)
        std     = self.actor_log_std.exp().clamp(0.05, 1.0)
        dist    = Normal(mean, std)
        logprobs = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return logprobs, value, entropy


# ========================= SIMULATORE GPU =========================
class GPUSimulator:
    """
    Simula N macchine in parallelo su GPU.
    Il percorso è una polyline definita da TRACK_POINTS.
    Ogni agente sa su quale segmento si trova; osservazioni e reward
    sono calcolati rispetto al segmento più vicino.
    """
    def __init__(self, num_instances, track_data):
        self.N = num_instances
        self.dt = 1.0 / 30.0  # 30 Hz

        (self.track, self.seg_dirs, self.seg_normals,
         self.seg_headings, self.seg_lengths,
         self.cum_lengths, self.total_length) = track_data

        self.n_segments = len(self.seg_lengths)

        # Reward tracking cumulativo per ogni agente
        self.cumulative_reward = torch.zeros(num_instances, device=DEVICE)

        self.reset()

    def _find_nearest_segment(self):
        """
        Per ogni agente trova il segmento più vicino e la proiezione su esso.
        Ritorna: seg_idx, lateral_offset (con segno), forward_on_seg (0..seg_len)
        """
        N = self.N
        S = self.n_segments

        # Vettore da ogni inizio-segmento ad ogni agente: (N, S, 2)
        starts = self.track[:-1]                     # (S, 2)
        to_car = self.pos.unsqueeze(1) - starts.unsqueeze(0)  # (N, S, 2)

        # Proiezione sul segmento
        dirs = self.seg_dirs.unsqueeze(0)            # (1, S, 2)
        forward_proj = (to_car * dirs).sum(dim=2)    # (N, S)
        forward_proj_clamped = torch.maximum(forward_proj, torch.zeros_like(forward_proj))
        forward_proj_clamped = torch.minimum(forward_proj_clamped, self.seg_lengths.unsqueeze(0))

        # Punto più vicino su ogni segmento
        closest = starts.unsqueeze(0) + forward_proj_clamped.unsqueeze(2) * dirs  # (N, S, 2)
        dist_sq = ((self.pos.unsqueeze(1) - closest) ** 2).sum(dim=2)  # (N, S)

        # Segmento con distanza minima
        seg_idx = dist_sq.argmin(dim=1)              # (N,)

        # Raccogli i valori dal segmento scelto
        batch_idx = torch.arange(N, device=DEVICE)
        forward_on_seg = forward_proj_clamped[batch_idx, seg_idx]

        # Offset laterale con segno
        normals = self.seg_normals[seg_idx]           # (N, 2)
        to_car_on_seg = to_car[batch_idx, seg_idx]    # (N, 2)
        lateral_offset = (to_car_on_seg * normals).sum(dim=1)  # (N,)

        return seg_idx, lateral_offset, forward_on_seg

    def reset(self, mask=None):
        """Spawn: posizioni casuali lungo il primo 20% del percorso."""
        if mask is None:
            n = self.N
        else:
            n = int(mask.sum().item())

        # Sceglie una posizione casuale lungo il percorso (primi 20%)
        max_dist = self.total_length * 0.2
        t = torch.rand(n, device=DEVICE) * max_dist  # distanza dall'inizio

        # Trova il segmento corrispondente
        seg_idx = torch.zeros(n, dtype=torch.long, device=DEVICE)
        for i in range(self.n_segments - 1):
            seg_idx = torch.where(t >= self.cum_lengths[i + 1], 
                                  torch.tensor(i + 1, device=DEVICE), seg_idx)
        local_t = t - self.cum_lengths[seg_idx]

        # Posizione
        starts = self.track[:-1][seg_idx]       # (n, 2)
        dirs   = self.seg_dirs[seg_idx]         # (n, 2)
        pos    = starts + local_t.unsqueeze(1) * dirs

        # Heading allineato al segmento + piccola perturbazione
        heading = self.seg_headings[seg_idx]
        heading = heading + (torch.rand(n, device=DEVICE) - 0.5) * 0.3

        # Velocità iniziale
        speed_init = torch.rand(n, device=DEVICE) * 10.0  # 0-10 m/s
        vel = torch.zeros((n, 2), device=DEVICE)
        vel[:, 0] = torch.cos(heading) * speed_init
        vel[:, 1] = torch.sin(heading) * speed_init

        if mask is None:
            self.pos     = pos
            self.heading = heading
            self.vel     = vel
            self.speed   = speed_init * 3.6
            self.cumulative_reward = torch.zeros(n, device=DEVICE)
        else:
            self.pos[mask]     = pos
            self.heading[mask] = heading
            self.vel[mask]     = vel
            self.speed[mask]   = speed_init * 3.6
            self.cumulative_reward[mask] = 0.0

    def get_observation(self):
        seg_idx, lat_offset, fwd_on_seg = self._find_nearest_segment()

        # Heading error rispetto al segmento corrente
        ideal_heading = self.seg_headings[seg_idx]
        heading_err = ideal_heading - self.heading
        heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi

        # Velocità proiettata sul segmento
        dirs = self.seg_dirs[seg_idx]
        normals = self.seg_normals[seg_idx]
        forward_vel = (self.vel * dirs).sum(dim=1)
        lateral_vel = (self.vel * normals).sum(dim=1)

        obs = torch.stack([
            self.speed / MAX_SPEED_KMH,                          # [0]
            (lat_offset / TRACK_HALF_WIDTH).clamp(-1.5, 1.5),   # [1]
            heading_err / np.pi,                                  # [2]
            torch.sin(self.heading),                              # [3]
            torch.cos(self.heading),                              # [4]
            lateral_vel * 0.05,                                   # [5]
            forward_vel * 0.05,                                   # [6]
        ], dim=1)

        return obs, seg_idx, lat_offset, fwd_on_seg, heading_err

    def _get_total_progress(self, seg_idx, fwd_on_seg):
        """Progresso totale lungo il percorso in metri."""
        return self.cum_lengths[seg_idx] + fwd_on_seg

    def step_with_reward(self, actions):
        throttle = actions[:, 0].clamp(-1, 1)
        steer    = actions[:, 1].clamp(-1, 1)

        # --- Heading ---
        self.heading += steer * STEER_RATE * self.dt
        self.heading  = (self.heading + np.pi) % (2 * np.pi) - np.pi

        # --- Accelerazione ---
        accel = torch.where(throttle >= 0,
                            throttle * MAX_ACCEL,
                            throttle * MAX_BRAKE)
        self.vel[:, 0] += torch.cos(self.heading) * accel * self.dt
        self.vel[:, 1] += torch.sin(self.heading) * accel * self.dt

        # --- Drag ---
        speed_ms = torch.norm(self.vel, dim=1)
        drag_force = DRAG_COEFF * speed_ms
        drag_decel = torch.where(speed_ms > 0.1,
                                 drag_force / speed_ms.clamp(min=0.1),
                                 torch.zeros_like(speed_ms))
        self.vel *= (1.0 - drag_decel * self.dt).unsqueeze(1).clamp(min=0)

        # --- Posizione ---
        self.pos += self.vel * self.dt
        self.speed = torch.norm(self.vel, dim=1) * 3.6

        # --- Osservazione ---
        obs, seg_idx, lat_offset, fwd_on_seg, heading_err = self.get_observation()

        # --- Progresso totale ---
        total_progress = self._get_total_progress(seg_idx, fwd_on_seg)

        # Velocità in avanti lungo il segmento
        dirs = self.seg_dirs[seg_idx]
        forward_vel = (self.vel * dirs).sum(dim=1)

        # --- Terminazione ---
        out_of_bounds = lat_offset.abs() > TRACK_HALF_WIDTH
        past_end   = total_progress > self.total_length - 1.0
        behind_start = total_progress < -10.0
        dones = out_of_bounds | past_end | behind_start

        # --- Reward ---
        speed_reward    = (forward_vel * 3.6 / MAX_SPEED_KMH).clamp(-0.5, 1.0)
        dist_penalty    = -(lat_offset.abs() / TRACK_HALF_WIDTH).clamp(0, 1) * 0.5
        heading_penalty = -(heading_err.abs() / np.pi) * 0.3

        reward = speed_reward + dist_penalty + heading_penalty
        reward[past_end]      += 50.0
        reward[out_of_bounds] -= 5.0
        reward[behind_start]  -= 5.0

        # Accumula reward per ranking
        self.cumulative_reward += reward

        # --- Reset agenti morti ---
        if dones.any():
            self.reset(mask=dones)
            obs_new, _, _, _, _ = self.get_observation()
            obs[dones] = obs_new[dones]

        # --- Metriche ---
        self._last_lat_dist    = lat_offset.abs()
        self._last_progress    = total_progress
        self._last_past_end    = past_end

        return obs, dones, reward

    def get_best_agent_indices(self, n=10):
        """Ritorna gli indici degli N agenti con reward cumulativo più alto."""
        n = min(n, self.N)
        _, indices = torch.topk(self.cumulative_reward, n)
        return indices


# ========================= PYGAME VISUALIZER =========================
def pygame_render_worker(track_points_np, half_width, data_queue, num_vis):
    """Processo separato: visualizza il tracciato e i migliori agenti con Pygame."""
    import pygame

    pygame.init()
    WIDTH, HEIGHT = 1200, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Project ARES – Best Agents Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)
    big_font = pygame.font.SysFont("monospace", 18, bold=True)

    # --- Camera: calcola bounds del tracciato ---
    tp = track_points_np
    min_x, min_y = tp.min(axis=0) - half_width * 2
    max_x, max_y = tp.max(axis=0) + half_width * 2

    # Assicura un'area minima visibile
    range_x = max(max_x - min_x, 100)
    range_y = max(max_y - min_y, 50)

    # Scala per fittare la finestra con margini
    margin = 60
    scale_x = (WIDTH - 2 * margin) / range_x
    scale_y = (HEIGHT - 2 * margin) / range_y
    scale = min(scale_x, scale_y)

    # Centro
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    def world_to_screen(wx, wy):
        sx = int(margin + (wx - min_x) * scale)
        sy = int(HEIGHT - margin - (wy - min_y) * scale)
        return sx, sy

    # Colori agenti
    agent_colors = [
        (0, 255, 100),    # verde brillante (best)
        (255, 255, 0),    # giallo
        (255, 100, 255),  # magenta
        (0, 200, 255),    # ciano
        (255, 150, 0),    # arancio
        (255, 80, 80),    # rosso chiaro
        (150, 255, 150),  # verde chiaro
        (200, 200, 255),  # blu chiaro
        (255, 200, 100),  # pesca
        (200, 100, 255),  # viola
    ]

    # Pre-calcola punti del tracciato in screen coords
    track_screen = [world_to_screen(p[0], p[1]) for p in tp]

    # Bordi (offset perpendicolare)
    border_left  = []
    border_right = []
    for i in range(len(tp) - 1):
        dx = tp[i+1][0] - tp[i][0]
        dy = tp[i+1][1] - tp[i][1]
        length = max(np.sqrt(dx*dx + dy*dy), 0.001)
        nx, ny = -dy / length, dx / length
        for t_val in [0.0, 1.0]:
            px = tp[i][0] + t_val * dx
            py = tp[i][1] + t_val * dy
            border_left.append(world_to_screen(px + nx * half_width,
                                               py + ny * half_width))
            border_right.append(world_to_screen(px - nx * half_width,
                                                py - ny * half_width))

    # Stato
    latest_data = None
    epoch_info = ""

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Leggi ultimo dato dalla coda
        try:
            while not data_queue.empty():
                latest_data = data_queue.get_nowait()
        except queue.Empty:
            pass

        # --- RENDER ---
        screen.fill((15, 15, 25))

        # Griglia sottile
        grid_color = (30, 30, 45)
        for gx in range(int(min_x), int(max_x) + 1, max(1, int(range_x / 10))):
            sx, _ = world_to_screen(gx, 0)
            pygame.draw.line(screen, grid_color, (sx, 0), (sx, HEIGHT), 1)
        for gy in range(int(min_y), int(max_y) + 1, max(1, int(range_y / 10))):
            _, sy = world_to_screen(0, gy)
            pygame.draw.line(screen, grid_color, (0, sy), (WIDTH, sy), 1)

        # Bordi della pista (rosso scuro)
        if len(border_left) > 1:
            pygame.draw.lines(screen, (120, 30, 30), False, border_left, 2)
            pygame.draw.lines(screen, (120, 30, 30), False, border_right, 2)

        # Linea centrale (ciano)
        if len(track_screen) > 1:
            pygame.draw.lines(screen, (0, 180, 220), False, track_screen, 3)

        # Waypoints
        for i, sp in enumerate(track_screen):
            pygame.draw.circle(screen, (0, 220, 255), sp, 5)
            label = font.render(f"{i}", True, (100, 100, 130))
            screen.blit(label, (sp[0] + 8, sp[1] - 8))

        # Agenti
        if latest_data is not None:
            positions, headings, speeds, rewards, epoch_info = latest_data

            for i in range(len(positions)):
                px, py = positions[i]
                sx, sy = world_to_screen(px, py)
                heading = headings[i]
                spd = speeds[i]
                rew = rewards[i]
                color = agent_colors[i % len(agent_colors)]

                # Corpo auto (rettangolo ruotato)
                car_half_l = CAR_LENGTH * scale / 2
                car_half_w = CAR_WIDTH * scale / 2
                cos_h, sin_h = np.cos(-heading), np.sin(-heading)  # -heading per screen Y flip

                corners = []
                for dx, dy in [(-car_half_l, -car_half_w),
                               ( car_half_l, -car_half_w),
                               ( car_half_l,  car_half_w),
                               (-car_half_l,  car_half_w)]:
                    rx = dx * cos_h - dy * sin_h
                    ry = dx * sin_h + dy * cos_h
                    corners.append((int(sx + rx), int(sy + ry)))
                pygame.draw.polygon(screen, color, corners, 2)

                # Freccia direzione
                arrow_len = 15
                ax = sx + arrow_len * np.cos(-heading)
                ay = sy + arrow_len * np.sin(-heading)
                pygame.draw.line(screen, color, (sx, sy), (int(ax), int(ay)), 2)

                # Label
                label = font.render(f"#{i+1} {spd:.0f}km/h R:{rew:.1f}", True, color)
                screen.blit(label, (sx + 12, sy - 12))

        # HUD
        title = big_font.render("PROJECT ARES – Best Agents", True, (0, 220, 255))
        screen.blit(title, (10, 10))
        if epoch_info:
            info_surf = font.render(epoch_info, True, (180, 180, 200))
            screen.blit(info_surf, (10, 35))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


class AsyncPygameVisualizer:
    def __init__(self, track_points_np, num_vis=10):
        self.num_vis = num_vis
        self.data_queue = mp.Queue(maxsize=3)
        self.process = mp.Process(
            target=pygame_render_worker,
            args=(track_points_np, TRACK_HALF_WIDTH,
                  self.data_queue, self.num_vis),
            daemon=True,
        )
        self.process.start()

    def update(self, sim, epoch_info_str=""):
        """Invia le posizioni dei migliori agenti al visualizzatore."""
        best_idx = sim.get_best_agent_indices(self.num_vis)
        positions = sim.pos[best_idx].detach().cpu().numpy()
        headings  = sim.heading[best_idx].detach().cpu().numpy()
        speeds    = sim.speed[best_idx].detach().cpu().numpy()
        rewards   = sim.cumulative_reward[best_idx].detach().cpu().numpy()

        try:
            self.data_queue.put_nowait(
                (positions, headings, speeds, rewards, epoch_info_str)
            )
        except queue.Full:
            pass


# ========================= ROLLOUT =========================
def _rollout(sim, model, visualizer, steps, epoch_info=""):
    memory_states    = []
    memory_actions   = []
    memory_logprobs  = []
    memory_rewards   = []
    memory_terminals = []
    memory_values    = []
    epoch_reward     = 0.0

    total_dist_sum   = 0.0
    total_fwd_sum    = 0.0
    total_completed  = 0
    total_steps_n    = 0

    model.eval()
    with torch.no_grad():
        for step in range(steps):
            obs, _, _, _, _ = sim.get_observation()
            action, logprob, value = model.act(obs)
            _, dones, reward = sim.step_with_reward(action)

            if visualizer and step % 5 == 0:
                visualizer.update(sim, epoch_info)

            memory_states.append(obs)
            memory_actions.append(action)
            memory_logprobs.append(logprob)
            memory_rewards.append(reward)
            memory_terminals.append(dones)
            memory_values.append(value.squeeze())
            epoch_reward += reward.mean().item()

            total_dist_sum  += sim._last_lat_dist.mean().item()
            total_fwd_sum   += sim._last_progress.mean().item()
            total_completed += sim._last_past_end.sum().item()
            total_steps_n   += 1

    metrics = {
        'avg_lat_dist': total_dist_sum / max(total_steps_n, 1),
        'avg_fwd':      total_fwd_sum / max(total_steps_n, 1),
        'completed':    int(total_completed),
    }

    return (memory_states, memory_actions, memory_logprobs,
            memory_rewards, memory_terminals, memory_values,
            epoch_reward, metrics)


# ========================= TRAINING =========================
def train():
    track_data = build_track(TRACK_POINTS)
    total_length = track_data[-1]

    print(f"Tracciato: {len(TRACK_POINTS)} waypoints, {total_length:.0f}m totali")
    print(f"Waypoints: {TRACK_POINTS}")

    sim   = GPUSimulator(NUM_INSTANCES, track_data)
    model = PilotNet(input_dim=7).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    visualizer = None
    if VISUALIZE:
        track_np = np.array(TRACK_POINTS, dtype=np.float32)
        visualizer = AsyncPygameVisualizer(track_np, num_vis=NUM_VIS_AGENTS)

    start_epoch = 0
    if os.path.exists(SAVE_PATH):
        try:
            ckpt = torch.load(SAVE_PATH, weights_only=False, map_location=DEVICE)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
                optimizer.load_state_dict(ckpt['optimizer_state'])
                start_epoch = ckpt.get('epoch', 0) + 1
                print(f"Checkpoint caricato: ripresa dall'epoca {start_epoch}")
            else:
                model.load_state_dict(ckpt)
                print("Modello legacy caricato.")
        except Exception as e:
            print(f"Errore caricamento: {e}. Inizio da zero.")

    sim.reset()

    for epoch in range(start_epoch, EPOCHS):
        # Reset cumulative reward per ranking a inizio epoca
        sim.cumulative_reward.zero_()

        # --- Epoch info string per il visualizzatore ---
        epoch_info = f"Epoca {epoch}"

        # --- Rollout ---
        (memory_states, memory_actions, memory_logprobs,
         memory_rewards, memory_terminals, memory_values,
         epoch_reward, metrics) = _rollout(sim, model, visualizer,
                                           STEPS_PER_EPOCH, epoch_info)

        # --- Adaptive entropy ---
        cur_std = model.actor_log_std.exp().mean().item()
        if cur_std > 0.5:
            ent_coef = 0.005
        elif cur_std < 0.15:
            ent_coef = 0.02
        else:
            ent_coef = 0.01

        # --- Discounted returns ---
        returns = []
        discounted_reward = torch.zeros(NUM_INSTANCES, device=DEVICE)
        for r, d in zip(reversed(memory_rewards), reversed(memory_terminals)):
            discounted_reward = r + GAMMA * discounted_reward * (~d).float()
            returns.insert(0, discounted_reward)

        returns      = torch.stack(returns).detach()
        old_states   = torch.stack(memory_states).detach().view(-1, 7)
        old_actions  = torch.stack(memory_actions).detach().view(-1, 2)
        old_logprobs = torch.stack(memory_logprobs).detach().view(-1)
        old_values   = torch.stack(memory_values).detach().view(-1)

        advantages = returns.view(-1) - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PPO Update ---
        n_total = old_states.shape[0]
        model.train()
        total_loss = 0.0
        n_updates  = 0

        for _ in range(K_EPOCHS):
            perm = torch.randperm(n_total, device=DEVICE)
            for start in range(0, n_total, MINI_BATCH_SIZE):
                idx = perm[start : start + MINI_BATCH_SIZE]

                logprobs, state_values, dist_entropy = model.evaluate(
                    old_states[idx], old_actions[idx]
                )
                ratios = torch.exp(logprobs - old_logprobs[idx])
                mb_adv = advantages[idx]
                surr1  = ratios * mb_adv
                surr2  = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * mb_adv

                loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * nn.MSELoss()(state_values.squeeze(),
                                         returns.view(-1)[idx])
                    - ent_coef * dist_entropy
                ).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                n_updates  += 1

        # --- Log ---
        if epoch % 5 == 0:
            avg_reward = epoch_reward / STEPS_PER_EPOCH
            with torch.no_grad():
                avg_speed = sim.speed.mean().item()
                best_reward = sim.cumulative_reward.max().item()
            print(
                f"Epoca {epoch:4d} | "
                f"Reward: {avg_reward:7.3f} | "
                f"Std: {cur_std:.3f} | "
                f"Loss: {total_loss / max(n_updates, 1):.4f} | "
                f"LatDist: {metrics['avg_lat_dist']:.2f}m | "
                f"AvgFwd: {metrics['avg_fwd']:.0f}m | "
                f"Speed: {avg_speed:.1f}km/h | "
                f"BestR: {best_reward:.1f} | "
                f"Completed: {metrics['completed']}"
            )
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'reward': avg_reward,
            }, SAVE_PATH)


# ========================= MAIN =========================
if __name__ == "__main__":
    mp.freeze_support()
    print(f"Dispositivo: {DEVICE} | Istanze: {NUM_INSTANCES}")
    try:
        train()
    except KeyboardInterrupt:
        print("\nAddestramento interrotto. Modello salvato.")
